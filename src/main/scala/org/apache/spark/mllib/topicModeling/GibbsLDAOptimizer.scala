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
  sum => brzSum, DenseMatrix => BDM, Matrix => BM, CSCMatrix => BSM}

import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, DenseMatrix => SDM, SparseMatrix => SSM, Vector => SV, Matrix}

import GibbsLDAOptimizer._
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
  override def initialize(docs: RDD[(Long, SV)], lda: org.apache.spark.mllib.topicModeling.LDA): LDAOptimizer = {
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

    val sampledCorpus = sampler.sampleTokens(corpus, totalTopicCounter, innerIter + seed,
      numTokens, numTopics, numTerms, alpha, alphaAS, beta)
    sampledCorpus.edges.persist(storageLevel)

    val counterCorpus = updateCounter(sampledCorpus, numTopics)
    checkpoint(counterCorpus, innerIter, checkpointInterval)
    counterCorpus.vertices.persist(storageLevel)
    totalTopicCounter = collectTotalTopicCounter(counterCorpus, numTopics, numTokens)

    corpus.edges.unpersist(false)
    corpus.vertices.unpersist(false)
    lastSampledCorpus.map(_.edges.unpersist(false))
    lastSampledCorpus = Some(sampledCorpus)
    corpus = counterCorpus

    innerIter += 1

    if (printPerplexity) {
      println(s"Perplexity of $innerIter-th is ${perplexity}")
    }
    this
  }

  /**
   * get model of current iteration
   * @param iterationTimes
   * @return
   */
  def getLDAModel(iterationTimes: Array[Double]): org.apache.spark.mllib.topicModeling.LDAModel = {
    val ldaModel:GibbsLDAModel = GibbsLDAModel(numTopics, numTerms, alpha, beta)
    termVertices.collect().foreach { case (term, counter) =>
      ldaModel.merge(term.toInt, counter)
    }
    ldaModel.ttc.foreach(_.compact())
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
          val bsv = BSV.zeros[Int](doc.size)
          // TODO should not iterate the vector like this, however, no other iterface available
          val array = doc.toArray
          for (i <- 0 until array.size) {
            val count = array(i).toInt
            if (count != 0)
              bsv(i) = count
          }
          initializeEdges(gen, bsv, docId, numTopics)
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
      case "even" =>
        val edgeCnt = edges.count
        val partitions = edges.partitions.length
        val edgesPerPart = edgeCnt / partitions
        val newEdges = edges.zipWithIndex.map{ case (edge, idx) =>
          (idx / edgesPerPart, edge)
        }.partitionBy(new HashPartitioner(partitions)).map(_._2)
        Graph.fromEdges(newEdges, null, storageLevel, storageLevel)
      case _ =>
        throw new IllegalArgumentException(s"invalid values of edgePartitioner are none, degree and even, but got $edgePartitioner")
    }

    // corpus = corpus.partitionBy(PartitionStrategy.EdgePartition2D)
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
    doc: BSV[Int],
    docId: DocId,
    numTopics: Int): Array[Edge[ED]] = {
    assert(docId >= 0)
    val newDocId: DocId = -(docId + 1L)
    val edges =
      doc.activeIterator.filter(_._2 > 0).map { case (termId, counter) =>
        val topics = new Array[Int](counter)
        for (i <- 0 until counter) {
          topics(i) = gen.nextInt(numTopics)
        }
        Edge(termId, newDocId, topics)
      }.toArray
    assert(edges.length > 0)
    edges
  }
}

class GibbsLDAModel (
//                      private[topicModel] val gtc: BDV[Double],
//                      private[topicModel] val ttc: Array[BSV[Double]],
                      private[topicModeling] val gtc: BDV[Count],
                      private[topicModeling] val ttc: Array[BSV[Count]],
                      val alpha: Float,
                      val beta: Float,
                      val alphaAS: Float) extends org.apache.spark.mllib.topicModeling.LDAModel with Serializable {

  @transient private lazy val numTopics = gtc.size
  @transient private lazy val numTerms = ttc.size
  @transient private lazy val numTokens = brzSum(gtc).toLong

//  def this(topicCounts: SDV, topicTermCounts: Array[SSV], alpha: Double, beta: Double) {
//    this(new BDV[Double](topicCounts.toArray), topicTermCounts.map(t =>
//def this(topicCounts: SDV, topicTermCounts: Array[SSV], alpha: Float, beta: Float) {
//  this(new BDV[Count](topicCounts.toArray), topicTermCounts.map(t =>
//      new BSV(t.indices, t.values, t.size)), alpha, beta, alpha)
//  }

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
    for (termId <- 0 until numTerms) {
      for (topicId <- 0 until numTopics) {
        matrix(termId, topicId) = ttc(termId)(topicId)
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
    (0 until numTopics).map(topicId => {
      val terms = (0 until numTerms).map(termId => (termId, ttc(termId)(topicId).toDouble))
        .sortBy(_._2)(Ordering[Double].reverse)
        .take(maxTermsPerTopic)
        .map(_._1).toArray
      val weights = terms.map(ttc(_)(topicId).toDouble)
      val wsum = weights.sum
      if (wsum > 1E-5) {
        (0 until weights.length).foreach(weights(_) /= wsum)
      }
      (terms, weights)
    }).toArray
  }

//  def globalTopicCounter = GibbsLDAOptimizerUtils.fromBreezeVector(gtc)
//
//  def topicTermCounter = ttc.map(t => GibbsLDAOptimizerUtils.fromBreezeVector(t))

  private[topicModeling] def merge(term: Int, counter: BV[Int]) = {
    counter.activeIterator.foreach { case (topic, cn) =>
      mergeOne(term, topic, cn)
    }
    this
  }

  private[topicModeling] def mergeOne(term: Int, topic: Int, inc: Int) = {
    gtc(topic) += inc
    ttc(term)(topic) += inc
    this
  }

  private[topicModeling] def merge(other: GibbsLDAModel) = {
    gtc :+= other.gtc
    for (i <- 0 until ttc.length) {
      ttc(i) :+= other.ttc(i)
    }
    this
  }

  /**
   * For each document in the training set, return the distribution over topics for that document
   * ("theta_doc").
   *
   * @return  RDD of (document ID, topic distribution) pairs
   */
  def predict(documents: RDD[(Long, SV)],
              optimizer: GibbsLDAOptimizer,
              totalIter: Int = 25): RDD[(Long, VD)] = {

    val previousCorpus: Graph[VD, ED] = initializeCorpus(documents, numTopics, optimizer.getStorageLevel(), optimizer.edgePartitioner)

    var corpus: Graph[VD, ED] = previousCorpus.mapVertices {(id, attr) =>
      val newAttr = if(id > 0) Some(ttc(id.toInt)).getOrElse(attr) else attr
      newAttr
    }

    corpus.vertices.count()
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
    }

    corpus.vertices.filter(t => t._1 < 0)
  }
}

object GibbsLDAModel {
  /**
   * create GibbsLDAModel
   * @param numTopics number of topics
   * @param numTerms number of terms (vocabulary)
   * @param alpha alpha
   * @param beta beta
   * @return
   */
  def apply(numTopics: Int, numTerms: Int, alpha: Float = 0.1f, beta: Float = 0.01f) = {
    val gtc = BDV.zeros[Count](numTopics)
    val ttc = (0 until numTerms).map(_ => BSV.zeros[Count](numTopics)).toArray
    new GibbsLDAModel(gtc, ttc, alpha, beta, alpha)
  }
}

private[topicModeling] object GibbsLDAOptimizerUtils {
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
      if (v < key) {
        b = mid + 1
      }
      else if (v > key) {
        e = mid - 1
      }
      else {
        return mid
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
