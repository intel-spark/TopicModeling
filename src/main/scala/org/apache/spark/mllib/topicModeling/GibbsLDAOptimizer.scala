/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.topicModeling

import java.util.Random

import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.{HashPartitioner, Logging, Partitioner}
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoRegistrator
import org.apache.spark.storage.StorageLevel

import breeze.linalg.{Vector => BV, DenseVector => BDV, SparseVector => BSV, StorageVector,
  sum => brzSum, norm => brzNorm, DenseMatrix => BDM, Matrix => BM, CSCMatrix => BSM}

import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV,
  DenseMatrix => SDM, SparseMatrix => SSM, Vector => SV, Matrix}

import GibbsLDAOptimizer._

import scala.reflect.ClassTag

/**
 *
 * Adapted from Spark PRs(#1405 and #4807) and JIRA SPARK-5556 (https://github.com/witgo/spark/tree/lda_Gibbs,
 * https://github.com/EntilZha/spark/tree/LDA-Refactor, https://github.com/witgo/zen/tree/lda_opt/ml, etc.),
 * with several extensions (e.g., support for MLlib interface, predict and in-place state update) added
 */
class GibbsLDAOptimizer private[topicModeling](
  private var alphaAS: Float,
  private var storageLevel: StorageLevel,
  private var sampler:GibbsLDASampler = new GibbsLDAAliasSampler,
  var edgePartitioner:String = "none",
  var printPerplexity:Boolean = false)
  extends LDAOptimizer with Serializable with Logging {

  def this() = this(0.1f, StorageLevel.MEMORY_AND_DISK)

  @transient private var corpus:Graph[VD, ED] = null
  private var alpha = 0.01f
  private var beta = 0.01f
  private var numTopics = 0
  private var numTerms = 0
  private var numTokens = 0L
  private var checkpointInterval = Int.MaxValue

  /**
   * Initializer for the optimizer. LDA passes the common parameters to the optimizer and
   * the internal structure can be initialized properly.
   */
  override def initialize(docs: RDD[(Long, SV)], lda: LDA): LDAOptimizer = {
    alpha = lda.getAlpha.toFloat
    beta = lda.getBeta.toFloat
    numTopics = lda.getK
    numTerms = docs.first()._2.size
    seed = lda.getSeed.toInt
    setCheckpointInterval(lda.getCheckpointInterval)
    corpus = initializeCorpus(docs, lda.getK, storageLevel, edgePartitioner)
    numTokens = corpus.edges.map(e => e.attr.size.toDouble).sum().toLong
    totalTopicCounter = collectTotalTopicCounter(corpus, numTopics, numTokens)
    this
  }

  private var lastSampledCorpus:Option[Graph[VD, ED]] = None

  /**
   * run an iteration
   * @return
   */
  def next(): LDAOptimizer = {

    gibbsSampling()

    if (printPerplexity) {
      println(s"Perplexity of $innerIter-th is ${perplexity}")
    }
    this
  }

  private def gibbsSampling(): Unit = {
    val sampledCorpus = sampler.sampleTokens(corpus, totalTopicCounter, innerIter + seed,
      numTokens, numTopics, numTerms, alpha, alphaAS, beta)
    sampledCorpus.edges.persist(storageLevel)
    sampledCorpus.edges.count()

    val counterCorpus = updateCounter(sampledCorpus, numTopics)
    checkpoint(counterCorpus, innerIter, checkpointInterval)
    counterCorpus.vertices.persist(storageLevel)
    counterCorpus.vertices.count()

    totalTopicCounter = collectTotalTopicCounter(counterCorpus, numTopics, numTokens)

    corpus.edges.unpersist(false)
    corpus.vertices.unpersist(false)
    lastSampledCorpus.map(_.edges.unpersist(false))
    lastSampledCorpus = Some(sampledCorpus)
    corpus = counterCorpus

    innerIter += 1
  }

  /**
   * get model of current iteration
   * @param iterationTimes
   * @return
   */
  override def getLDAModel(iterationTimes: Array[Double]): LDAModel = {
    val ldaModel:GibbsLDAModel = saveModel(1)
    ldaModel
  }

  def setCheckpointInterval(cpInterval: Int): this.type = {
    this.checkpointInterval = cpInterval
    this
  }

  def getCheckpointInterval(): Int = this.checkpointInterval

  def setAlphaAS(alphaAS: Float): this.type = {
    this.alphaAS = alphaAS
    this
  }

  def getAlphaAS():Float = this.alphaAS

  def setStorageLevel(newStorageLevel: StorageLevel): this.type = {
    this.storageLevel = newStorageLevel
    this
  }

  def getStorageLevel(): StorageLevel = this.storageLevel

  def setSeed(newSeed: Int): this.type = {
    this.seed = newSeed
    this
  }

  def getSeed():Int = this.seed

  def setSampler(sampler:GibbsLDASampler): this.type = {
    this.sampler = sampler
    this
  }

  def getSampler(): GibbsLDASampler = {
    this.sampler
  }

  def setSampler(sampler:String): this.type = {
    this.sampler =
      sampler.toLowerCase match {
        case "alias" => new GibbsLDAAliasSampler
        case "sparse" => new GibbsLDASparseSampler
        case "light" => new GibbsLDALightSampler
        case "fast" => new GibbsLDAFastSampler
        case _ =>
          throw new IllegalArgumentException(s"Only alias, sparse, light are supported but got $sampler.")
      }
    this
  }

  def getCorpus = corpus

  @transient private var seed = new Random().nextInt()
  @transient private var innerIter = 1
  @transient private var totalTopicCounter: BDV[Count] = null

  private def termVertices = corpus.vertices.filter(t => t._1 >= 0)

  private def docVertices = corpus.vertices.filter(t => t._1 < 0)

  // scalastyle:off
  /**
   * p(w)=\sum_{k}{p(k|d)*p(w|k)}=
   * \sum_{k}{\frac{{n}_{kw}+{\beta }_{w}} {{n}_{k}+\bar{\beta }} \frac{{n}_{kd}+{\alpha }_{k}} {\sum{{n}_{k}}+\bar{\alpha }}}=
   * \sum_{k} \frac{{\alpha }_{k}{\beta }_{w}  + {n}_{kw}{\alpha }_{k} + {n}_{kd}{\beta }_{w} + {n}_{kw}{n}_{kd}}{{n}_{k}+\bar{\beta }} \frac{1}{\sum{{n}_{k}}+\bar{\alpha }}}
   * \exp^{-(\sum{\log(p(w))})/N}
   * N is total token number within the corpus
   */
  // scalastyle:on
  def perplexity(): Double = {
    val totalTopicCounter = this.totalTopicCounter
    val numTopics = this.numTopics
    val numTerms = this.numTerms
    val alpha = this.alpha
    val beta = this.beta
    val totalSize = brzSum(totalTopicCounter)
    var totalProb = 0D

    // \frac{{\alpha }_{k}{\beta }_{w}}{{n}_{k}+\bar{\beta }}
    totalTopicCounter.activeIterator.foreach { case (topic, cn) =>
      totalProb += alpha * beta / (cn + numTerms * beta)
    }

    val termProb = corpus.mapVertices { (vid, counter) =>
      val probDist = BSV.zeros[Double](numTopics)
      if (vid >= 0) {
        val termTopicCounter = counter
        // \frac{{n}_{kw}{\alpha }_{k}}{{n}_{k}+\bar{\beta }}
        termTopicCounter.activeIterator.foreach { case (topic, cn) =>
          probDist(topic) = cn * alpha /
            (totalTopicCounter(topic) + numTerms * beta)
        }
      } else {
        val docTopicCounter = counter
        // \frac{{n}_{kd}{\beta }_{w}}{{n}_{k}+\bar{\beta }}
        docTopicCounter.activeIterator.foreach { case (topic, cn) =>
          probDist(topic) = cn * beta /
            (totalTopicCounter(topic) + numTerms * beta)
        }
      }
      probDist.compact()
      (counter, probDist)
    }.mapTriplets { triplet =>
      val (termTopicCounter, termProb) = triplet.srcAttr
      val (docTopicCounter, docProb) = triplet.dstAttr
      val docSize = docTopicCounter.sum
      val docTermSize = triplet.attr.length
      var prob = 0D

      // \frac{{n}_{kw}{n}_{kd}}{{n}_{k}+\bar{\beta}}
      docTopicCounter.activeIterator.foreach { case (topic, cn) =>
        prob += cn * termTopicCounter(topic) /
          (totalTopicCounter(topic) + numTerms * beta)
      }
      prob += brzSum(docProb) + brzSum(termProb) + totalProb
      prob += prob / (docSize + numTopics * alpha)

      docTermSize * Math.log(prob)
    }.edges.map(t => t.attr).sum()

    math.exp(-1 * termProb / totalSize)
  }

  /**
   * Save the term-topic related model
   * @param totalIter
   */
  def saveModel(totalIter: Int = 1): GibbsLDAModel = {
    var termTopicCounter: RDD[(VertexId, VD)] = null
    for (iter <- 1 to totalIter) {
      logInfo(s"Save TopicModel (Iteration $iter/$totalIter)")
      var previousTermTopicCounter = termTopicCounter
      gibbsSampling()
      val newTermTopicCounter = termVertices
      termTopicCounter = Option(termTopicCounter).map(_.join(newTermTopicCounter).map {
        case (term, (a, b)) =>
          var c: VD = null
          if(a.isInstanceOf[BSV[Count]] && b.isInstanceOf[BSV[Count]]) {
            c = a.asInstanceOf[BSV[Count]] :+ b.asInstanceOf[BSV[Count]]
          } else if(a.isInstanceOf[BDV[Count]] && b.isInstanceOf[BDV[Count]]){
            c = a.asInstanceOf[BDV[Count]] :+ b.asInstanceOf[BDV[Count]]
          } else if(a.isInstanceOf[BDV[Count]]) {
            c = a.asInstanceOf[BDV[Count]] :+ b.asInstanceOf[BSV[Count]].toDenseVector
          } else {
            c = a.asInstanceOf[BSV[Count]].toDenseVector :+ b.asInstanceOf[BDV[Count]]
          }
          (term, c)
      }).getOrElse(newTermTopicCounter)

      termTopicCounter.persist(storageLevel).count()
      Option(previousTermTopicCounter).foreach(_.unpersist(blocking = false))
      previousTermTopicCounter = termTopicCounter
    }
    val rand = new Random()
    val ttc = termTopicCounter.mapValues(c => {
      if (c.isInstanceOf[BDV[Count]]) {
        val dv = c.asInstanceOf[BDV[Count]]
        val nc = new BDV[Count](dv.data.map (v => {
          val mid = v.toDouble / totalIter
          val l = math.floor(mid)
          if (rand.nextDouble() > mid - l) {
            l
          } else {
            l + 1
          }
        }.asInstanceOf[Count]))
        nc.asInstanceOf[StorageVector[Count]]
      } else {
        val sv = c.asInstanceOf[BSV[Count]]
        val nc = new BSV[Count](sv.index.slice(0, sv.used), sv.data.slice(0, sv.used).map(v => {
          val mid = v.toDouble / totalIter
          val l = math.floor(mid)
          if (rand.nextDouble() > mid - l) {
            l
          } else {
            l + 1
          }
        }.asInstanceOf[Count]), c.length)
        nc
      }
    })

    ttc.persist(storageLevel)
    val gtc = ttc.map(_._2).aggregate(BDV.zeros[Count](numTopics))(_ :+= _, _ :+= _)
    new GibbsLDAModel(gtc, ttc, numTerms, alpha, beta, alphaAS)
  }
}

object GibbsLDAOptimizer {

  private[topicModeling] type DocId = VertexId
  private[topicModeling] type WordId = VertexId
  private[topicModeling] type Count = Int
  private[topicModeling] type ED = Array[Count]
  private[topicModeling] type VD = StorageVector[Count]

  def checkpoint(corpus: Graph[VD, ED], innerIter: Int, checkpointInterval: Int): Unit = {
    if (innerIter % checkpointInterval == 0 && corpus.edges.sparkContext.getCheckpointDir.isDefined) {
      corpus.checkpoint()
    }
  }

  def collectTotalTopicCounter(graph: Graph[VD, ED], numTopics: Int, numTokens: Long): BDV[Count] = {
    val globalTopicCounter = collectGlobalCounter(graph, numTopics)
    assert(brzSum(globalTopicCounter) == numTokens)
    globalTopicCounter
  }

  def updateCounter(graph: Graph[VD, ED], numTopics: Int): Graph[VD, ED] = {
    val newCounter = graph.aggregateMessages[BSV[Count]](ctx => {
      val topics = ctx.attr
      val vector = BSV.zeros[Count](numTopics)
      for (topic <- topics) {
        vector(topic) += 1
      }
      ctx.sendToDst(vector)
      ctx.sendToSrc(vector)
    }, _ + _, TripletFields.EdgeOnly)
    .mapValues(sparseVector => {
      val storageVector:VD =
        if (sparseVector.activeSize > sparseVector.length / 2) {
          sparseVector.toDenseVector
        } else {
          sparseVector
        }
      storageVector
    })
    // GraphImpl.fromExistingRDDs(newCounter, graph.edges)
    GraphImpl(newCounter, graph.edges)
  }

  def collectGlobalCounter(graph: Graph[VD, ED], numTopics: Int): BDV[Count] = {
    graph.vertices.filter(t => t._1 >= 0).map(_._2).
      aggregate(BDV.zeros[Count](numTopics))((a, b) => {
      a :+= b
    }, _ :+= _)
  }

  def initializeCorpus(
    docs: RDD[(GibbsLDAOptimizer.DocId, SV)],
    numTopics: Int,
    storageLevel: StorageLevel, edgePartitioner:String): Graph[VD, ED] = {
    val edges = docs.mapPartitionsWithIndex((pid, iter) => {
      val gen = new Random(pid)
      iter.flatMap {
        case (docId, doc) =>
          initializeEdges(gen, doc, docId, numTopics)
      }
    })
    edges.persist(storageLevel)
    val corpus: Graph[VD, ED] = edgePartitioner match {
      case "none" =>
        Graph.fromEdges(edges, null, storageLevel, storageLevel)
      case "degree" =>
        val degreeCorpus = Graph.fromEdges(edges, null, storageLevel, storageLevel)
        val degrees = degreeCorpus.outerJoinVertices(degreeCorpus.degrees) { (vid, data, deg) => deg.getOrElse(0) }
        val numPartitions = edges.partitions.size
        val partitionStrategy = new DBHPartitioner(numPartitions)
        val newEdges = degrees.triplets.map { e =>
          (partitionStrategy.getPartition(e), Edge(e.srcId, e.dstId, e.attr))
        }.partitionBy(new HashPartitioner(numPartitions)).map(_._2)
        Graph.fromEdges(newEdges, null, storageLevel, storageLevel)
      case "docIdCluster" =>
        val newEdges = edges.map { e =>
          (e.dstId, Edge(e.srcId, e.dstId, e.attr))
        }.partitionBy(new HashPartitioner(docs.partitions.size)).map(_._2)
        Graph.fromEdges(newEdges, null, storageLevel, storageLevel)

      case _ =>
        throw new IllegalArgumentException(s"invalid values of edgePartitioner are none, degree and docIdCluster, but got $edgePartitioner")
    }

    val resultCorpus = updateCounter(corpus, numTopics).cache()
    resultCorpus.vertices.count()
    resultCorpus.edges.count()
    corpus.unpersist(false)
    edges.unpersist(false)
    docs.unpersist(false)
    resultCorpus
  }

  private def initializeEdges(
     gen: Random,
     doc: SV,
     docId: DocId,
     numTopics: Int): Iterator[Edge[ED]] = {
    assert(docId >= 0)
    val newDocId: DocId = -(docId + 1L)

    doc.toBreeze.activeIterator.filter(_._2 > 0).map {case (termId, counter) =>
      val topics = new Array[Int](counter.toInt)
      for (i <- 0 until counter.toInt) {
        topics(i) = gen.nextInt(numTopics)
      }
      Edge(termId, newDocId, topics)
    }
  }
}

class GibbsLDAModel (
    private[topicModeling] val gtc: BDV[Count],
    private[topicModeling] val ttc: RDD[(VertexId, VD)],
    val numTerms: Int,
    val alpha: Float,
    val beta: Float,
    val alphaAS: Float) extends org.apache.spark.mllib.topicModeling.LDAModel with Serializable {

  @transient private lazy val numTopics = gtc.size
  @transient private lazy val numTokens = brzSum(gtc).toLong

  /** Number of topics */
  def k: Int = numTopics

  /** Vocabulary size (number of terms or terms in the vocabulary) */
  def vocabSize: Int = numTerms

  /**
   * Inferred topics, where each topic is represented by a distribution over terms.
   * This is a matrix of size vocabSize x k, where each column is a topic.
   * No guarantees are given about the ordering of the topics.
   */
  def topicsMatrix: Matrix = {
    val matrix = BDM.zeros[Double](numTerms, numTopics)

    val ttcArray = Array.fill(numTerms.toInt) {
      BSV.zeros[Count](numTopics)
    }
    ttc.collect().foreach { case (termId, vector) =>
      ttcArray(termId.toInt) :+= vector
    }

    for (termId <- 0 until numTerms) {
      val sv = ttcArray(termId)
      for (topicId <- 0 until numTopics) {
        matrix(termId, topicId) = sv(topicId)
      }
    }
    GibbsLDAOptimizerUtils.fromBreezeMatrix(matrix)
  }

  /**
   * Return the topics described by weighted terms.
   *
   * This limits the number of terms per topic.
   * This is approximate; it may not return exactly the top-weighted terms for each topic.
   * To get a more precise set of top terms, increase maxTermsPerTopic.
   *
   * @param maxTermsPerTopic  Maximum number of terms to collect for each topic.
   * @return  Array over topics.  Each topic is represented as a pair of matching arrays:
   *          (term indices, term weights in topic).
   *          Each topic's terms are sorted in order of decreasing weight.
   */
  def describeTopics(maxTermsPerTopic: Int): Array[(Array[Int], Array[Double])] = {
    val ttcArray = Array.fill(numTerms.toInt) {
      BSV.zeros[Count](numTopics)
    }
    ttc.collect().foreach { case (termId, vector) =>
      ttcArray(termId.toInt) :+= vector
    }

    (0 until numTopics).map(topicId => {
      val terms = (0 until numTerms).map(termId => (termId, ttcArray(termId)(topicId).toDouble))
        .sortBy(_._2)(Ordering[Double].reverse)
        .take(maxTermsPerTopic)
        .map(_._1).toArray
      val weights = terms.map(ttcArray(_)(topicId).toDouble)
      val wsum = weights.sum
      if (wsum > 1E-5) {
        (0 until weights.length).foreach(weights(_) /= wsum)
      }
      (terms, weights)
    }).toArray
  }

  /**
   * For each document in the training set, return the distribution over topics for that document
   * ("theta_doc").
   *
   * @return  RDD of (document ID, topic distribution) pairs
   */
  def predict(documents: RDD[(Long, SV)],
              optimizer: GibbsLDAOptimizer,
              totalIter: Int = 25,
              burnIn: Int = 22): RDD[(Long, SV)] = {

    val previousCorpus: Graph[VD, ED] = initializeCorpus(documents, numTopics,
      optimizer.getStorageLevel(), optimizer.edgePartitioner)

    var corpus = previousCorpus.outerJoinVertices(ttc) { (vid, c, v) =>
      if (vid >= 0) {
        assert(v.isDefined)
      }
      v.getOrElse(c)
    }
    corpus.persist(optimizer.getStorageLevel)
    corpus.vertices.count()
    corpus.edges.count()
    previousCorpus.edges.unpersist(blocking = false)
    previousCorpus.vertices.unpersist(blocking = false)

    var docTopicCounter: RDD[(VertexId, VD)] = null
    for(i <- 1 to totalIter) {
      val previousCorpus = corpus

      val sampledCorpus = optimizer.getSampler().sampleTokens(corpus, gtc, i + optimizer.getSeed(),
        numTokens, numTopics, numTerms, alpha, alphaAS, beta)
      sampledCorpus.persist(optimizer.getStorageLevel)

      corpus = updateCounter(sampledCorpus, numTopics)
      checkpoint(corpus, i, optimizer.getCheckpointInterval)
      corpus.persist(optimizer.getStorageLevel)

      previousCorpus.edges.unpersist(false)
      previousCorpus.vertices.unpersist(false)

      sampledCorpus.edges.unpersist(false)
      sampledCorpus.vertices.unpersist(false)

      if (i > burnIn) {
        var previousDocTopicCounter = docTopicCounter
        val newTermTopicCounter = corpus.vertices.filter(t => t._1 < 0)
        docTopicCounter = Option(docTopicCounter).map(_.join(newTermTopicCounter).map {
          case (docId, (a, b)) => {
            var c: VD = null
            if(a.isInstanceOf[BSV[Count]] && b.isInstanceOf[BSV[Count]]) {
              c = a.asInstanceOf[BSV[Count]] :+ b.asInstanceOf[BSV[Count]]
            } else if(a.isInstanceOf[BDV[Count]] && b.isInstanceOf[BDV[Count]]){
              c = a.asInstanceOf[BDV[Count]] :+ b.asInstanceOf[BDV[Count]]
            } else if(a.isInstanceOf[BDV[Count]]) {
              c = a.asInstanceOf[BDV[Count]] :+ b.asInstanceOf[BSV[Count]].toDenseVector
            } else {
              c = a.asInstanceOf[BSV[Count]].toDenseVector :+ b.asInstanceOf[BDV[Count]]
            }
            (docId, c)
          }
        }).getOrElse(newTermTopicCounter)

        docTopicCounter.persist(optimizer.getStorageLevel).count()
        Option(previousDocTopicCounter).foreach(_.unpersist(blocking = false))
        previousDocTopicCounter = docTopicCounter
      }
    }
    docTopicCounter.map { case (docId, v) =>
      if(v.isInstanceOf[BDV[Count]]) {
        val dv = v.asInstanceOf[BDV[Count]]
        val norm = brzNorm(dv, 1)
        (docId, GibbsLDAOptimizerUtils.fromBreezeConv[Double](dv.map(_.toDouble) / norm ))
      } else {
        val sv = v.asInstanceOf[BSV[Count]]
        val norm = brzNorm(sv, 1)
        (docId, GibbsLDAOptimizerUtils.fromBreezeConv[Double](sv.map(_.toDouble) / norm))
      }
    }
  }
}

private[topicModeling] object GibbsLDAOptimizerUtils {

  private def _conv[T1: ClassTag, T2: ClassTag](data: Array[T1]): Array[T2] = {
    data.map(_.asInstanceOf[T2]).array
  }

  def fromBreezeConv[T: ClassTag](breezeVector: BV[T]): SV = {
    implicit val conv: Array[T] => Array[Double] = _conv[T, Double]

    breezeVector match {
      case v: BDV[T] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new SDV(v.data)
        } else {
          new SDV(v.toArray) // Can't use underlying array directly, so make a new one
        }
      case v: BSV[T] =>
        if (v.index.length == v.used) {
          new SSV(v.length, v.index, _conv[T, Double](v.data))
        } else {
          new SSV(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[T] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

  def binarySearchInterval(
                            index: Array[Float],
                            key: Float,
                            begin: Int,
                            end: Int,
                            greater: Boolean): Int = {
    if (begin == end) {
      return if (greater) end else begin - 1
    }
    var b = begin
    var e = end - 1

    var mid: Int = (e + b) >> 1
    while (b <= e) {
      mid = (e + b) >> 1
      val v = index(mid)
      if (scala.math.abs(key-v)<=1e-6) {
        return mid
      }
      else if (v > key) {
        e = mid - 1
      }
      else {
        b = mid + 1
      }
    }
    val v = index(mid)
    mid = if ((greater && v >= key) || (!greater && v <= key)) {
      mid
    }
    else if (greater) {
      mid + 1
    }
    else {
      mid - 1
    }

    if (greater) {
      if (mid < end) assert(index(mid) >= key)
      if (mid > 0) assert(index(mid - 1) <= key)
    } else {
      if (mid > 0) assert(index(mid) <= key)
      if (mid < end - 1) assert(index(mid + 1) >= key)
    }
    mid
  }

  /**
   * Creates a Matrix instance from a breeze matrix.
   * @param breeze a breeze matrix
   * @return a Matrix instance
   */
  def fromBreezeMatrix(breeze: BM[Double]): Matrix = {
    breeze match {
      case dm: BDM[Double] =>
        require(dm.majorStride == dm.rows,
          "Do not support stride size different from the number of rows.")
        new SDM(dm.rows, dm.cols, dm.data)
      case sm: BSM[Double] =>
        new SSM(sm.rows, sm.cols, sm.colPtrs, sm.rowIndices, sm.data)
      case _ =>
        throw new UnsupportedOperationException(
          s"Do not support conversion from type ${breeze.getClass.getName}.")
    }
  }

  /**
   * Creates a vector instance from a breeze vector.
   */
  def fromBreezeVector(breezeVector: BV[Double]): SV = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new SDV(v.data)
        } else {
          new SDV(v.toArray)  // Can't use underlying array directly, so make a new one
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SSV(v.length, v.index, v.data)
        } else {
          new SSV(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }
}

/**
 * Degree-Based Hashing, the paper:
 * http://nips.cc/Conferences/2014/Program/event.php?ID=4569
 * @param partitions
 */
private class DBHPartitioner(partitions: Int) extends Partitioner {
  val mixingPrime: Long = 1125899906842597L

  def numPartitions = partitions

  def getPartition(key: Any): Int = {
    val edge = key.asInstanceOf[EdgeTriplet[Int, ED]]
    val srcDeg = edge.srcAttr
    val dstDeg = edge.dstAttr
    val srcId = edge.srcId
    val dstId = edge.dstId
    val minId = if (srcDeg < dstDeg) srcId else dstId
    getPartition(minId)
  }

  def getPartition(idx: Int): PartitionID = {
    (math.abs(idx * mixingPrime) % partitions).toInt
  }

  def getPartition(idx: Long): PartitionID = {
    (math.abs(idx * mixingPrime) % partitions).toInt
  }

  override def equals(other: Any): Boolean = other match {
    case h: DBHPartitioner =>
      h.numPartitions == numPartitions
    case _ =>
      false
  }

  override def hashCode: Int = numPartitions
}

private[topicModeling] class LDAKryoRegistrator extends KryoRegistrator {
  def registerClasses(kryo: com.esotericsoftware.kryo.Kryo) {
    val gkr = new GraphKryoRegistrator
    gkr.registerClasses(kryo)

    kryo.register(classOf[BSV[GibbsLDAOptimizer.Count]])
    kryo.register(classOf[BSV[Double]])

    kryo.register(classOf[BDV[GibbsLDAOptimizer.Count]])
    kryo.register(classOf[BDV[Double]])

    kryo.register(classOf[SV])
    kryo.register(classOf[SSV])
    kryo.register(classOf[SDV])

    kryo.register(classOf[GibbsLDAOptimizer.ED])
    kryo.register(classOf[GibbsLDAOptimizer.VD])

    kryo.register(classOf[Random])
    kryo.register(classOf[GibbsLDAOptimizer])
    kryo.register(classOf[GibbsLDAModel])
  }
}
