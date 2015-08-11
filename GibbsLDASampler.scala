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

import java.lang.ref.SoftReference
import org.apache.spark.Logging
import org.apache.spark.graphx.impl.GraphImpl
import org.apache.spark.graphx.{TripletFields, VertexId, Graph}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, sum => brzSum, StorageVector}

import java.util.{Random, PriorityQueue => JPriorityQueue}
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.util.collection.AppendOnlyMap
import scala.reflect.ClassTag

trait GibbsLDASampler {
  type VD = GibbsLDAOptimizer.VD
  type ED = GibbsLDAOptimizer.ED
  type Count = GibbsLDAOptimizer.Count

  def sampleTokens(graph: Graph[GibbsLDAOptimizer.VD, ED],
                   totalTopicCounter: BDV[Count],
                   innerIter: Long,
                   numTokens: Long,
                   numTopics: Int,
                   numTerms: Int,
                   alpha: Float,
                   alphaAS: Float,
                   beta: Float): Graph[VD, ED]
}

private[topicModeling] class GibbsAliasTable(initUsed: Int) extends Serializable {

  private var _l: Array[Int] = new Array[Int](initUsed)
  private var _h: Array[Int] = new Array[Int](initUsed)
  private var _p: Array[Float] = new Array[Float](initUsed)
  private var _used = initUsed

  def l: Array[Int] = _l

  def h: Array[Int] = _h

  def p: Array[Float] = _p

  def used: Int = _used

  def length: Int = size

  def size: Int = l.length

  def sampleAlias(gen: Random): Int = {
    val bin = gen.nextInt(_used)
    val prob = _p(bin)
    if (_used * prob > gen.nextFloat()) {
      _l(bin)
    } else {
      _h(bin)
    }
  }

  private[GibbsAliasTable] def reset(newSize: Int): this.type = {
    if (_l.length < newSize) {
      _l = new Array[Int](newSize)
      _h = new Array[Int](newSize)
      _p = new Array[Float](newSize)
    }
    _used = newSize
    this
  }
}

private[topicModeling] object GibbsAliasTable {
  @transient private lazy val tableOrdering = new scala.math.Ordering[(Int, Float)] {
    override def compare(x: (Int, Float), y: (Int, Float)): Int = {
      Ordering.Double.compare(x._2, y._2)
    }
  }
  @transient private lazy val tableReverseOrdering = tableOrdering.reverse

  def generateAlias(sv: BV[Float]): GibbsAliasTable = {
    val used = sv.activeSize
    val sum = brzSum(sv)
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, sum, used)
  }

  def generateAlias(probs: Iterator[(Int, Float)], sum: Float, used: Int): GibbsAliasTable = {
    val table = new GibbsAliasTable(used)
    generateAlias(probs, sum, used, table)
  }

  def generateAlias(
                     probs: Iterator[(Int, Float)],
                     sum: Float,
                     used: Int,
                     table: GibbsAliasTable): GibbsAliasTable = {
    table.reset(used)
    val pMean = 1.0f / used
    val lq = new JPriorityQueue[(Int, Float)](used, tableOrdering)
    val hq = new JPriorityQueue[(Int, Float)](used, tableReverseOrdering)

    probs.slice(0, used).foreach { pair =>
      val i = pair._1
      val pi = pair._2 / sum
      if (pi < pMean) {
        lq.add((i, pi))
      } else {
        hq.add((i, pi))
      }
    }

    var offset = 0
    while (!lq.isEmpty & !hq.isEmpty) {
      val (i, pi) = lq.remove()
      val (h, ph) = hq.remove()
      table.l(offset) = i
      table.h(offset) = h
      table.p(offset) = pi
      val pd = ph - (pMean - pi)
      if (pd >= pMean) {
        hq.add((h, pd))
      } else {
        lq.add((h, pd))
      }
      offset += 1
    }
    while (!hq.isEmpty) {
      val (h, ph) = hq.remove()
      assert(ph - pMean < 1e-6)
      table.l(offset) = h
      table.h(offset) = h
      table.p(offset) = ph
      offset += 1
    }

    while (!lq.isEmpty) {
      val (i, pi) = lq.remove()
      assert(pMean - pi < 1e-6)
      table.l(offset) = i
      table.h(offset) = i
      table.p(offset) = pi
      offset += 1
    }
    table
  }

  def generateAlias(sv: BV[Float], sum: Float): GibbsAliasTable = {
    val used = sv.activeSize
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, sum, used)
  }

  def generateAlias(sv: BV[Float], sum: Float, table: GibbsAliasTable): GibbsAliasTable = {
    val used = sv.activeSize
    val probs = sv.activeIterator.slice(0, used)
    generateAlias(probs, sum, used, table)
  }

}

class GibbsLDAAliasSampler extends GibbsLDASampler with Logging with Serializable{
  def sampleTokens(graph: Graph[GibbsLDAOptimizer.VD, ED],
                   totalTopicCounter: BDV[Count],
                   innerIter: Long,
                   numTokens: Long,
                   numTopics: Int,
                   numTerms: Int,
                   alpha: Float,
                   alphaAS: Float,
                   beta: Float): Graph[VD, ED] = {
    val parts = graph.edges.partitions.size

    val newGraph = graph.mapTriplets(
      (pid, iter) => {
        val gen = new XORShiftRandom(parts * innerIter + pid)
        // table is a per term data structure
        // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
        // so, use below simple cache to avoid calculating table each time
        val lastTable = new GibbsAliasTable(numTopics)
        var lastVid: VertexId = -1
        var lastWSum = 0.0f
        val dv = tDense(totalTopicCounter, numTokens, numTerms, alpha, alphaAS, beta)
        val dData = new Array[Float](numTopics.toInt)
        val t = GibbsAliasTable.generateAlias(dv._2, dv._1)
        val tSum = dv._1

        iter.map {
          triplet =>
            val termId = triplet.srcId
            val docId = triplet.dstId
            val termTopicCounter = triplet.srcAttr
            val docTopicCounter = triplet.dstAttr
            val topics = triplet.attr

            for (i <- 0 until topics.length) {
              val currentTopic = topics(i)
              docTopicCounter.synchronized {
                termTopicCounter.synchronized {
                  dSparse(totalTopicCounter, termTopicCounter, docTopicCounter, dData,
                    currentTopic, numTokens, numTerms, alpha, alphaAS, beta)
                  if (lastVid != termId || gen.nextDouble() < 1e-4) {
                    lastWSum = wordTable(lastTable, totalTopicCounter, termTopicCounter,
                      termId, numTokens, numTerms, alpha, alphaAS, beta)
                    lastVid = termId
                  }
                  val newTopic = tokenSampling(gen, t, tSum, lastTable, termTopicCounter, lastWSum,
                    docTopicCounter, dData, currentTopic)

                  if (newTopic != currentTopic) {
                    topics(i) = newTopic
                    docTopicCounter(currentTopic) -= 1
                    docTopicCounter(newTopic) += 1
                    // if (docTopicCounter(currentTopic) == 0) docTopicCounter.compact()

                    termTopicCounter(currentTopic) -= 1
                    termTopicCounter(newTopic) += 1
                    // if (termTopicCounter(currentTopic) == 0) termTopicCounter.compact()

                    totalTopicCounter(currentTopic) -= 1
                    totalTopicCounter(newTopic) += 1
                  }
                }
              }
            }
            topics
        }
      }, TripletFields.All)
    GraphImpl(newGraph.vertices.mapValues(t => null), newGraph.edges)
  }

  // scalastyle:off
  /**
   * the formula is
   * t = \frac{{\beta }_{w} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} ) } {({n}_{k}^{-di}+\bar{\beta}) ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  // scalastyle:on
  private def tDense(
                      totalTopicCounter: BDV[Count],
                      numTokens: Long,
                      numTerms: Int,
                      alpha: Float,
                      alphaAS: Float,
                      beta: Float): (Float, BDV[Float]) = {
    val numTopics = totalTopicCounter.length
    val t = BDV.zeros[Float](numTopics)
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1F + alphaAS * numTopics
    val betaSum = numTerms * beta
    var sum = 0.0f
    for (topic <- 0 until numTopics) {
      val last = beta * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      t(topic) = last
      sum += last
    }
    (sum, t)
  }

  // scalastyle:off
  /**
   * the formula is:
   * d =  \frac{{n}_{kd} ^{-di}({\sum{n}_{k}^{-di} + \bar{\acute{\alpha}}})({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta})({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * =  \frac{{n}_{kd} ^{-di}({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta}) }
   */
  // scalastyle:on
  private def dSparse(
                       totalTopicCounter: BDV[Count],
                       termTopicCounter: VD,
                       docTopicCounter: VD,
                       d: Array[Float],
                       currentTopic: Int,
                       numTokens: Long,
                       numTerms: Int,
                       alpha: Float,
                       alphaAS: Float,
                       beta: Float): Unit = {
    val data = docTopicCounter.data
    val used = docTopicCounter.activeSize

    // val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    var sum = 0.0f
    for (i <- 0 until used) {
      val topic = docTopicCounter.indexAt(i)
      val count = data(i)
      val adjustment = if (currentTopic == topic) -1F else 0
      val last = (count + adjustment) * (termTopicCounter(topic) + adjustment + beta) /
        (totalTopicCounter(topic) + adjustment + betaSum)
      // val lastD = (count + adjustment) * termSum * (termTopicCounter(topic) + adjustment + beta) /
      //  ((totalTopicCounter(topic) + adjustment + betaSum) * termSum)

      sum += last
      d(i) = sum
    }
  }

  private def wordTable(
                         table: GibbsAliasTable,
                         totalTopicCounter: BDV[Count],
                         termTopicCounter: VD,
                         termId: VertexId,
                         numTokens: Long,
                         numTerms: Int,
                         alpha: Float,
                         alphaAS: Float,
                         beta: Float): Float = {
    val sv = wSparse(totalTopicCounter, termTopicCounter,
      numTokens, numTerms, alpha, alphaAS, beta)
    GibbsAliasTable.generateAlias(sv._2, sv._1, table)
    sv._1
  }

  // scalastyle:off
  /**
   * use both Gibbs sampler and Metropolis Hastings sampler
   * Complexity is O(1)
   * Use formula (3) from the Gibbs sampler paper: Rethinking LDA: Why Priors Matter
   * \frac{{n}_{kw}^{-di}+{\beta }_{w}}{{n}_{k}^{-di}+\bar{\beta}} \frac{{n}_{kd} ^{-di}+ \bar{\alpha} \frac{{n}_{k}^{-di} + \acute{\alpha}}{\sum{n}_{k} +\bar{\acute{\alpha}}}}{\sum{n}_{kd}^{-di} +\bar{\alpha}}
   * = t + w + d
   * t the global part
   * t = \frac{{\beta }_{w} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} ) } {({n}_{k}^{-di}+\bar{\beta}) ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * w: term related
   * w = \frac{ {n}_{kw}^{-di} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} )}{({n}_{k}^{-di}+\bar{\beta})({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * d: product of doc and term
   * d =  \frac{{n}_{kd}^{-di}({\sum{n}_{k}^{-di} + \bar{\acute{\alpha}}})({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta})({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * =  \frac{{n}_{kd ^{-di}({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta}) }
   * where:
   * \bar{\beta}=\sum_{w}{\beta}_{w}
   * \bar{\alpha}=\sum_{k}{\alpha}_{k}
   * \bar{\acute{\alpha}}=\bar{\acute{\alpha}}=\sum_{k}\acute{\alpha}
   * {n}_{kd} number of tokens in document d that are assigned to topic k
   * {n}_{kw} number of tokens with word w (across all docs) that are assigned to topic k
   * {n}_{k} number of tokens across all docs that are assigned to topic k
   * -di substract topic of current token
   */
  // scalastyle:on
  private def tokenSampling(gen: Random,
                            t: GibbsAliasTable,
                            tSum: Float,
                            w: GibbsAliasTable,
                            termTopicCounter: VD,
                            wSum: Float,
                            docTopicCounter: VD,
                            dData: Array[Float],
                            currentTopic: Int): Int = {
    val used = docTopicCounter.activeSize
    val dSum = dData(used - 1)
    val distSum = tSum + wSum + dSum
    val genSum = gen.nextFloat() * distSum
    if (genSum < dSum) {
      val dGenSum = gen.nextFloat() * dSum
      val pos = GibbsLDAOptimizerUtils.binarySearchInterval(dData, dGenSum, 0, used, true)
      docTopicCounter.indexAt(pos)
    } else if (genSum < (dSum + wSum)) {
      sampleSV(gen, w, termTopicCounter, currentTopic)
    } else {
      t.sampleAlias(gen)
    }
  }

  // scalastyle:off
  /**
   * the formula is:
   * w = \frac{ {n}_{kw}^{-di} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} )}{({n}_{k}^{-di}+\bar{\beta}) ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  // scalastyle:on
  private def wSparse(
                       totalTopicCounter: BDV[Count],
                       termTopicCounter: VD,
                       numTokens: Long,
                       numTerms: Int,
                       alpha: Float,
                       alphaAS: Float,
                       beta: Float): (Float, BSV[Float]) = {
    val numTopics = totalTopicCounter.length
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1F + alphaAS * numTopics
    val betaSum = numTerms * beta
    val w = BSV.zeros[Float](numTopics)
    var sum = 0.0f
    termTopicCounter.activeIterator.filter(_._2 > 0).foreach { t =>
      val topic = t._1
      val count = t._2
      val last = count * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      w(topic) = last
      sum += last
    }
    (sum, w)
  }

  private def sampleSV(gen: Random, table: GibbsAliasTable, sv: VD, currentTopic: Int): Int = {
    val docTopic = table.sampleAlias(gen)
    if (docTopic == currentTopic) {
      val svCounter = sv(currentTopic)
      // processing method is not appropriate here,
      // if we drop topic of current sampled token
      // svCounter == 1 && table.length > 1, however topic of current sampled token contains other tokens
      // svCounter > 1 && gen.nextDouble() < 1.0 / svCounter, probability that sampled topic belongs to current topic is 1/svCounter
      if ((svCounter == 1 && table.used > 1) ||
        (svCounter > 1 && gen.nextDouble() < 1.0 / svCounter)) {
        return sampleSV(gen, table, sv, currentTopic)
      }
    }
    docTopic
  }
}

class GibbsLDAFastSampler extends GibbsLDASampler with Serializable with Logging {
  def sampleTokens(graph: Graph[GibbsLDAOptimizer.VD, ED],
                   totalTopicCounter: BDV[Count],
                   innerIter: Long,
                   numTokens: Long,
                   numTopics: Int,
                   numTerms: Int,
                   alpha: Float,
                   alphaAS: Float,
                   beta: Float): Graph[VD, ED] = {
    val parts = graph.edges.partitions.size
    val nweGraph = graph.mapTriplets(
      (pid, iter) => {
        val gen = new Random(parts * innerIter + pid)
        // table is a per term data structure
        // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
        // so, use below simple cache to avoid calculating table each time
        val lastTable = new GibbsAliasTable(numTopics.toInt)
        var lastVid: VertexId = -1
        var lastWSum = 0.0f
        val dv = tDense(totalTopicCounter, numTokens, numTerms, alpha, alphaAS, beta)
        val dData = new Array[Float](numTopics.toInt)
        val t = GibbsAliasTable.generateAlias(dv._2, dv._1)
        val tSum = dv._1
        iter.map {
          triplet =>
            val termId = triplet.srcId
            val docId = triplet.dstId
            val termTopicCounter = triplet.srcAttr
            val docTopicCounter = triplet.dstAttr
            val topics = triplet.attr.clone()
            for (i <- 0 until topics.length) {
              val currentTopic = topics(i)
              dSparse(totalTopicCounter, termTopicCounter, docTopicCounter, dData,
                currentTopic, numTokens, numTerms, alpha, alphaAS, beta)
              if (lastVid != termId) {
                lastWSum = wordTable(lastTable, totalTopicCounter, termTopicCounter,
                  termId, numTokens, numTerms, alpha, alphaAS, beta)
                lastVid = termId
              }
              val newTopic = tokenSampling(gen, t, tSum, lastTable, termTopicCounter, lastWSum,
                docTopicCounter, dData, currentTopic)

              if (newTopic != currentTopic) {
                topics(i) = newTopic
              }
            }

            topics
        }
      }, TripletFields.All)
    GraphImpl(nweGraph.vertices.mapValues(t => null), nweGraph.edges)

  }

  private def tokenSampling(
                             gen: Random,
                             t: GibbsAliasTable,
                             tSum: Float,
                             w: GibbsAliasTable,
                             termTopicCounter: VD,
                             wSum: Float,
                             docTopicCounter: VD,
                             dData: Array[Float],
                             currentTopic: Int): Int = {
    val used = docTopicCounter.activeSize
    val dSum = dData(docTopicCounter.activeSize - 1)
    val distSum = tSum + wSum + dSum
    val genSum = gen.nextFloat() * distSum
    if (genSum < dSum) {
      val dGenSum = gen.nextFloat() * dSum
      val pos = binarySearchInterval(dData, dGenSum, 0, used, true)
      docTopicCounter.indexAt(pos)
    } else if (genSum < (dSum + wSum)) {
      sampleSV(gen, w, termTopicCounter, currentTopic)
    } else {
      t.sampleAlias(gen)
    }
  }

  /**
   * dense part in the decomposed sampling formula:
   * t = \frac{{\beta }_{w} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} ) } {({n}_{k}^{-di}+\bar{\beta})
   * ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  private def tDense(
                      totalTopicCounter: BDV[Count],
                      numTokens: Long,
                      numTerms: Int,
                      alpha: Float,
                      alphaAS: Float,
                      beta: Float): (Float, BDV[Float]) = {
    val numTopics = totalTopicCounter.length
    val t = BDV.zeros[Float](numTopics)
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1F + alphaAS * numTopics
    val betaSum = numTerms * beta
    var sum = 0.0f
    for (topic <- 0 until numTopics) {
      val last = beta * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      t(topic) = last
      sum += last
    }
    (sum, t)
  }

  /**
   * word related sparse part in the decomposed sampling formula:
   * w = \frac{ {n}_{kw}^{-di} \bar{\alpha} ( {n}_{k}^{-di} + \acute{\alpha} )}{({n}_{k}^{-di}+\bar{\beta})
   * ({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   */
  private def wSparse(
                       totalTopicCounter: BDV[Count],
                       termTopicCounter: VD,
                       numTokens: Long,
                       numTerms: Int,
                       alpha: Float,
                       alphaAS: Float,
                       beta: Float): (Float, BSV[Float]) = {
    val numTopics = totalTopicCounter.length
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1F + alphaAS * numTopics
    val betaSum = numTerms * beta
    val w = BSV.zeros[Float](numTopics)
    var sum = 0.0f
    termTopicCounter.activeIterator.filter(_._2 > 0).foreach { t =>
      val topic = t._1
      val count = t._2
      val last = count * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      w(topic) = last
      sum += last
    }
    (sum, w)
  }

  /**
   * doc related sparse part in the decomposed sampling formula:
   * d =  \frac{{n}_{kd} ^{-di}({\sum{n}_{k}^{-di} + \bar{\acute{\alpha}}})({n}_{kw}^{-di}+{\beta}_{w})}
   * {({n}_{k}^{-di}+\bar{\beta})({\sum{n}_{k}^{-di} +\bar{\acute{\alpha}}})}
   * =  \frac{{n}_{kd} ^{-di}({n}_{kw}^{-di}+{\beta}_{w})}{({n}_{k}^{-di}+\bar{\beta}) }
   */
  private def dSparse(
                       totalTopicCounter: BDV[Count],
                       termTopicCounter: VD,
                       docTopicCounter: VD,
                       d: Array[Float],
                       currentTopic: Int,
                       numTokens: Long,
                       numTerms: Int,
                       alpha: Float,
                       alphaAS: Float,
                       beta: Float): Unit = {
    val data = docTopicCounter.data
    val used = docTopicCounter.activeSize

    // val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    var sum = 0.0f
    for (i <- 0 until used) {
      val topic = docTopicCounter.indexAt(i)
      val count = data(i)
      val adjustment = if (currentTopic == topic) -1F else 0
      val last = (count + adjustment) * (termTopicCounter(topic) + adjustment + beta) /
        (totalTopicCounter(topic) + adjustment + betaSum)
      // val lastD = (count + adjustment) * termSum * (termTopicCounter(topic) + adjustment + beta) /
      //  ((totalTopicCounter(topic) + adjustment + betaSum) * termSum)
      sum += last
      d(i) = sum
    }
  }

  private def wordTable(
                         table: GibbsAliasTable,
                         totalTopicCounter: BDV[Count],
                         termTopicCounter: VD,
                         termId: VertexId,
                         numTokens: Long,
                         numTerms: Int,
                         alpha: Float,
                         alphaAS: Float,
                         beta: Float): Float = {
    val sv = wSparse(totalTopicCounter, termTopicCounter,
      numTokens, numTerms, alpha, alphaAS, beta)
    GibbsAliasTable.generateAlias(sv._2, sv._1, table)
    sv._1
  }

  private def sampleSV(
                        gen: Random,
                        table: GibbsAliasTable,
                        sv: VD,
                        currentTopic: Int,
                        currentTopicCounter: Int = 0,
                        numSampling: Int = 0): Int = {
    val docTopic = table.sampleAlias(gen)
    if (docTopic == currentTopic && numSampling < 16) {
      val svCounter = if (currentTopicCounter == 0) sv(currentTopic) else currentTopicCounter
      // TODO: not sure it is correct or not?
      // discard it if the newly sampled topic is current topic
      if ((svCounter == 1 && table.used > 1) ||
        /* the sampled topic that contains current token and other tokens */
        (svCounter > 1 && gen.nextDouble() < 1.0 / svCounter)
      /* the sampled topic has 1/svCounter probability that belongs to current token */ ) {
        return sampleSV(gen, table, sv, currentTopic, svCounter, numSampling + 1)
      }
    }
    docTopic
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
}

class GibbsLDALightSampler extends GibbsLDASampler with Logging with Serializable {

  private def sampleSV(
                        gen: Random,
                        table: GibbsAliasTable,
                        sv: VD,
                        currentTopic: Int,
                        currentTopicCounter: Int = 0,
                        numSampling: Int = 0): Int = {
    val docTopic = table.sampleAlias(gen)
    if (docTopic == currentTopic && numSampling < 16) {
      val svCounter = if (currentTopicCounter == 0) sv(currentTopic) else currentTopicCounter
      // TODO: not sure it is correct or not?
      // discard it if the newly sampled topic is current topic
      if ((svCounter == 1 && table.used > 1) ||
        /* the sampled topic that contains current token and other tokens */
        (svCounter > 1 && gen.nextDouble() < 1.0 / svCounter)
      /* the sampled topic has 1/svCounter probability that belongs to current token */ ) {
        return sampleSV(gen, table, sv, currentTopic, svCounter, numSampling + 1)
      }
    }
    docTopic
  }

  def sampleTokens(graph: Graph[GibbsLDAOptimizer.VD, ED],
                   totalTopicCounter: BDV[Count],
                   innerIter: Long,
                   numTokens: Long,
                   numTopics: Int,
                   numTerms: Int,
                   alpha: Float,
                   alphaAS: Float,
                   beta: Float): Graph[VD, ED] = {
    val parts = graph.edges.partitions.length
    val nweGraph = graph.mapTriplets(
      (pid, iter) => {
        val gen = new Random(parts * innerIter + pid)
        val docTableCache = new AppendOnlyMap[VertexId, SoftReference[(Float, GibbsAliasTable)]]()

        // table is a per term data structure
        // in GraphX, edges in a partition are clustered by source IDs (term id in this case)
        // so, use below simple cache to avoid calculating table each time
        val lastTable = new GibbsAliasTable(numTopics.toInt)
        var lastVid: VertexId = -1
        var lastWSum = 0.0f

        val p = tokenTopicProb(totalTopicCounter, beta, alpha,
          alphaAS, numTokens, numTerms) _
        val dPFun = docProb(totalTopicCounter, alpha, alphaAS, numTokens) _
        val wPFun = wordProb(totalTopicCounter, numTerms, beta) _

        var dD: GibbsAliasTable = null
        var dDSum: Float = 0.0f
        var wD: GibbsAliasTable = null
        var wDSum: Float = 0.0f

        iter.map {
          triplet =>
            val termId = triplet.srcId
            val docId = triplet.dstId
            val termTopicCounter = triplet.srcAttr
            val docTopicCounter = triplet.dstAttr
            val topics = triplet.attr

            if (dD == null || gen.nextDouble() < 1e-6) {
              var dv = dDense(totalTopicCounter, alpha, alphaAS, numTokens)
              dDSum = dv._1
              dD = GibbsAliasTable.generateAlias(dv._2, dDSum)

              dv = wDense(totalTopicCounter, numTerms, beta)
              wDSum = dv._1
              wD = GibbsAliasTable.generateAlias(dv._2, wDSum)
            }
            val (dSum, d) = docTopicCounter.synchronized {
              docTable(x => x == null || x.get() == null || gen.nextDouble() < 1e-2,
                docTableCache, docTopicCounter, docId)
            }
            val (wSum, w) = termTopicCounter.synchronized {
              if (lastVid != termId || gen.nextDouble() < 1e-4) {
                lastWSum = wordTable(lastTable, totalTopicCounter, termTopicCounter, termId, numTerms, beta)
                lastVid = termId
              }
              (lastWSum, lastTable)
            }
            for (i <- topics.indices) {
              var docProposal = gen.nextDouble() < 0.5
              var maxSampling = 8
              while (maxSampling > 0) {
                maxSampling -= 1
                docProposal = !docProposal
                val currentTopic = topics(i)
                var proposalTopic = -1
                val q = if (docProposal) {
                  if (gen.nextFloat() < dDSum / (dSum - 1.0f + dDSum)) {
                    proposalTopic = dD.sampleAlias(gen)
                  }
                  else {
                    proposalTopic = docTopicCounter.synchronized {
                      sampleSV(gen, d, docTopicCounter, currentTopic)
                    }
                  }
                  dPFun
                } else {
                  val table = if (gen.nextDouble() < wSum / (wSum + wDSum)) w else wD
                  proposalTopic = table.sampleAlias(gen)
                  wPFun
                }

                val newTopic = docTopicCounter.synchronized {
                  termTopicCounter.synchronized {
                    tokenSampling(gen, docTopicCounter, termTopicCounter, docProposal,
                      currentTopic, proposalTopic, q, p)
                  }
                }

                assert(newTopic >= 0 && newTopic < numTopics)
                if (newTopic != currentTopic) {
                  topics(i) = newTopic
                  docTopicCounter.synchronized {
                    docTopicCounter(currentTopic) -= 1
                    docTopicCounter(newTopic) += 1
                  }
                  termTopicCounter.synchronized {
                    termTopicCounter(currentTopic) -= 1
                    termTopicCounter(newTopic) += 1
                  }
                  totalTopicCounter(currentTopic) -= 1
                  totalTopicCounter(newTopic) += 1
                }
              }
            }
            topics
        }
      }, TripletFields.All)
    GraphImpl(nweGraph.vertices.mapValues(t => null), nweGraph.edges)
  }

  // scalastyle:off
  private def tokenTopicProb(
                              totalTopicCounter: BDV[Count],
                              beta: Float,
                              alpha: Float,
                              alphaAS: Float,
                              numTokens: Long,
                              numTerms: Int)(docTopicCounter: VD,
                                                termTopicCounter: VD,
                                                topic: Int,
                                                isAdjustment: Boolean): Float = {
    val numTopics = docTopicCounter.length
    val adjustment = if (isAdjustment) -1 else 0
    val ratio = (totalTopicCounter(topic) + adjustment + alphaAS) /
      (numTokens - 1 + alphaAS * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    // constant terms are removed (docLen - 1 + alpha * numTopics)
    (termTopicCounter(topic) + adjustment + beta) *
      (docTopicCounter(topic) + adjustment + asPrior) /
      (totalTopicCounter(topic) + adjustment + (numTerms * beta))

    // original formula: Rethinking LDA: Why Priors Matter formula (3)
    // val docLen = brzSum(docTopicCounter)
    // (termTopicCounter(topic) + adjustment + beta) * (docTopicCounter(topic) + adjustment + asPrior) /
    //   ((totalTopicCounter(topic) + adjustment + (numTerms * beta)) * (docLen - 1 + alpha * numTopics))
  }

  // scalastyle:on

  private def wordProb(
                        totalTopicCounter: BDV[Count],
                        numTerms: Int,
                        beta: Float)(termTopicCounter: VD, topic: Int, isAdjustment: Boolean): Float = {
    (termTopicCounter(topic) + beta) / (totalTopicCounter(topic) + beta * numTerms)
  }

  private def docProb(
                       totalTopicCounter: BDV[Count],
                       alpha: Float,
                       alphaAS: Float,
                       numTokens: Long)(docTopicCounter: VD, topic: Int, isAdjustment: Boolean): Float = {
    val adjustment = if (isAdjustment) -1 else 0
    val numTopics = totalTopicCounter.length
    val ratio = (totalTopicCounter(topic) + alphaAS) /
      (numTokens - 1 + alphaAS * numTopics)
    val asPrior = ratio * (alpha * numTopics)
    docTopicCounter(topic) + adjustment + asPrior
  }

  /**
   * \frac{{n}_{kw}}{{n}_{k}+\bar{\beta}}
   */
  private def wSparse(
                       totalTopicCounter: BDV[Count],
                       termTopicCounter: VD,
                       numTerms: Int,
                       beta: Float): (Float, BV[Float]) = {
    val numTopics = termTopicCounter.length
    val termSum = beta * numTerms
    val w = BSV.zeros[Float](numTopics)

    var sum = 0.0f
    termTopicCounter.activeIterator.foreach { t =>
      val topic = t._1
      val count = t._2
      if (count > 0) {
        val last = count / (totalTopicCounter(topic) + termSum)
        w(topic) = last
        sum += last
      }
    }
    (sum, w)
  }

  /**
   * \frac{{\beta}_{w}}{{n}_{k}+\bar{\beta}}
   */
  private def wDense(
                      totalTopicCounter: BDV[Count],
                      numTerms: Int,
                      beta: Float): (Float, BV[Float]) = {
    val numTopics = totalTopicCounter.length
    val t = BDV.zeros[Float](numTopics)
    val termSum = beta * numTerms
    var sum = 0.0f
    for (topic <- 0 until numTopics) {
      val last = beta / (totalTopicCounter(topic) + termSum)
      t(topic) = last
      sum += last
    }
    (sum, t)
  }

  private def dSparse(docTopicCounter: VD): (Float, BV[Float]) = {
    val numTopics = docTopicCounter.length
    val d = BSV.zeros[Float](numTopics)
    var sum = 0.0f
    docTopicCounter.activeIterator.foreach { t =>
      val topic = t._1
      val count = t._2
      if (count > 0) {
        val last = count
        d(topic) = last
        sum += last
      }
    }
    (sum, d)
  }


  private def dDense(
                      totalTopicCounter: BDV[Count],
                      alpha: Float,
                      alphaAS: Float,
                      numTokens: Long): (Float, BV[Float]) = {
    val numTopics = totalTopicCounter.length
    val asPrior = BDV.zeros[Float](numTopics)

    var sum = 0.0f
    for (topic <- 0 until numTopics) {
      val ratio = (totalTopicCounter(topic) + alphaAS) /
        (numTokens - 1 + alphaAS * numTopics)
      val last = ratio * (alpha * numTopics)
      asPrior(topic) = last
      sum += last
    }
    (sum, asPrior)
  }

  private def docTable(
                        updateFunc: SoftReference[(Float, GibbsAliasTable)] => Boolean,
                        cacheMap: AppendOnlyMap[VertexId, SoftReference[(Float, GibbsAliasTable)]],
                        docTopicCounter: VD,
                        docId: VertexId): (Float, GibbsAliasTable) = {
    val cacheD = cacheMap(docId)
    if (!updateFunc(cacheD)) {
      cacheD.get
    } else {
      docTopicCounter.synchronized {
        val sv = dSparse(docTopicCounter)
        val d = (sv._1, GibbsAliasTable.generateAlias(sv._2, sv._1))
        cacheMap.update(docId, new SoftReference(d))
        d
      }
    }
  }

  private def wordTable(
                         table: GibbsAliasTable,
                         totalTopicCounter: BDV[Count],
                         termTopicCounter: VD,
                         termId: VertexId,
                         numTerms: Int,
                         beta: Float): Float = {
    val sv = wSparse(totalTopicCounter, termTopicCounter, numTerms, beta)
    GibbsAliasTable.generateAlias(sv._2, sv._1, table)
    sv._1
  }

  // scalastyle:off
  /**
   * use both Gibbs sampler and Metropolis Hastings sampler
   * Complexity is O(1)
   * 1. use term related portions of the Gibbs sampler LDA formula
   * LightLDA: Big Topic Models on Modest Compute Clusters, formula(6):
   * ( \frac{{n}_{kd}^{-di}+{\beta }_{w}}{{n}_{k}^{-di}+\bar{\beta }} )
   * 2. use probability sampled from step 1 as Proposal q(.), use Metropolis Hastings sampler Sampling asymmetric transcendental equation
   * reference paper: Rethinking LDA: Why Priors Matter, formula(3)
   * \frac{{n}_{kw}^{-di}+{\beta }_{w}}{{n}_{k}^{-di}+\bar{\beta}} \frac{{n}_{kd} ^{-di}+ \bar{\alpha} \frac{{n}_{k}^{-di} + \acute{\alpha}}{\sum{n}_{k} +\bar{\acute{\alpha}}}}{\sum{n}_{kd}^{-di} +\bar{\alpha}}
   *
   * where
   * \bar{\beta}=\sum_{w}{\beta}_{w}
   * \bar{\alpha}=\sum_{k}{\alpha}_{k}
   * \bar{\acute{\alpha}}=\bar{\acute{\alpha}}=\sum_{k}\acute{\alpha}
   * {n}_{kd} number of tokens in document d that are assigned to topic k
   * {n}_{kw} number of tokens with word w (across all docs) that are assigned to topic k
   * {n}_{k} number of tokens across all docs that are assigned to topic k
   */
  // scalastyle:on
  def tokenSampling(
                     gen: Random,
                     docTopicCounter: VD,
                     termTopicCounter: VD,
                     docProposal: Boolean,
                     currentTopic: Int,
                     proposalTopic: Int,
                     q: (VD, Int, Boolean) => Float,
                     p: (VD, VD, Int, Boolean) => Float): Int = {
    if (proposalTopic == currentTopic) return proposalTopic
    val cp = p(docTopicCounter, termTopicCounter, currentTopic, true)
    val np = p(docTopicCounter, termTopicCounter, proposalTopic, false)
    val vd = if (docProposal) docTopicCounter else termTopicCounter
    val cq = q(vd, currentTopic, true)
    val nq = q(vd, proposalTopic, false)

    val pi = (np * cq) / (cp * nq)
    if (gen.nextDouble() < 1e-32) {
      println(s"Pi: ${pi}")
      println(s"($np * $cq) / ($cp * $nq)")
    }

    if (gen.nextDouble() < math.min(1.0, pi)) proposalTopic else currentTopic
  }
}

class GibbsLDASparseSampler extends GibbsLDASampler with Logging with Serializable {
  def sampleTokens(graph: Graph[GibbsLDAOptimizer.VD, ED],
                   totalTopicCounter: BDV[Count],
                   innerIter: Long,
                   numTokens: Long,
                   numTopics: Int,
                   numTerms: Int,
                   alpha: Float,
                   alphaAS: Float,
                   beta: Float): Graph[VD, ED] = {
    val parts = graph.edges.partitions.size
    val newGraph = graph.mapTriplets(
      (pid, iter) => {
        val gen = new XORShiftRandom(parts * innerIter + pid)
        val d = BDV.zeros[Float](numTopics)
        var lastTermId:VertexId = -1
        var lastWordTable:BSV[Float] = null
        var tCache: BDV[Float] = null
        iter.map {
          triplet =>
            val termId = triplet.srcId
            val termTopicCounter = triplet.srcAttr
            val docTopicCounter = triplet.dstAttr
            val topics = triplet.attr
            termTopicCounter.synchronized {
              docTopicCounter.synchronized {
                if (lastTermId != termId || gen.nextDouble() < 1e-4) {
                  lastWordTable = w(totalTopicCounter, termTopicCounter, termId,
                    numTokens, numTerms, alpha, beta, alphaAS)
                  lastTermId = termId
                }
                if (tCache == null || gen.nextDouble() < 1e-7) {
                  tCache = this.t(totalTopicCounter, numTokens, numTerms,
                    numTopics, alpha, beta, alphaAS)
                }
                val t = tCache
                var i = 0
                while (i < topics.length) {
                  val currentTopic = topics(i)
                  this.d(totalTopicCounter, termTopicCounter, docTopicCounter, d,
                    currentTopic, numTokens, numTerms, numTopics, beta, alphaAS)
                  val newTopic = multinomialDistSampler(gen, docTopicCounter, d, lastWordTable, t)
                  if (currentTopic != newTopic) {
                    topics(i) = newTopic
                    docTopicCounter(currentTopic) -= 1
                    docTopicCounter(newTopic) += 1
                    //if (docTopicCounter(currentTopic) == 0) docTopicCounter.compact()

                    termTopicCounter(currentTopic) -= 1
                    termTopicCounter(newTopic) += 1
                    //if (termTopicCounter(currentTopic) == 0) termTopicCounter.compact()

                    totalTopicCounter(currentTopic) -= 1
                    totalTopicCounter(newTopic) += 1
                  }
                  i += 1
                }
              }
            }
            topics
        }
      }, TripletFields.All)
    GraphImpl(newGraph.vertices.mapValues(t => null), newGraph.edges)
  }

  private def w(
                 totalTopicCounter: BDV[Count],
                 termTopicCounter: VD,
                 termId: VertexId,
                 numTokens: Long,
                 numTerms: Int,
                 alpha: Float,
                 beta: Float,
                 alphaAS: Float): BSV[Float] = {
    val numTopics = totalTopicCounter.length
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1F + alphaAS * numTopics
    val betaSum = numTerms * beta
    val length = termTopicCounter.length
    val used = termTopicCounter.activeSize
    val index =
      termTopicCounter match {
        case idx:BSV[_] => idx.index.slice(0, used)
        case idx:BDV[_] => (0 until idx.length).toArray
      }
    val data = termTopicCounter.data
    val w = new Array[Float](used)

    var lastSum = 0F
    var i = 0

    while (i < used) {
      val topic = index(i)
      val count = data(i)
      val lastW = count * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      lastSum += lastW
      w(i) = lastSum
      i += 1
    }
    new BSV[Float](index, w, used, length)
  }

  private def t(
                 totalTopicCounter: BDV[Count],
                 numTokens: Long,
                 numTerms: Int,
                 numTopics: Int,
                 alpha: Float,
                 beta: Float,
                 alphaAS: Float): BDV[Float] = {
    val t = BDV.zeros[Float](numTopics)
    val alphaSum = alpha * numTopics
    val termSum = numTokens - 1F + alphaAS * numTopics
    val betaSum = numTerms * beta

    var lastSum = 0F
    for (topic <- 0 until numTopics) {
      val lastT = beta * alphaSum * (totalTopicCounter(topic) + alphaAS) /
        ((totalTopicCounter(topic) + betaSum) * termSum)
      lastSum += lastT
      t(topic) = lastSum
    }
    t
  }

  private def d(
                 totalTopicCounter: BDV[Count],
                 termTopicCounter: VD,
                 docTopicCounter: VD,
                 d: BDV[Float],
                 currentTopic: Int,
                 numTokens: Long,
                 numTerms: Int,
                 numTopics: Int,
                 beta: Float,
                 alphaAS: Float): Unit = {
    val used = docTopicCounter.activeSize
    val index =
      docTopicCounter match {
        case idx:BSV[_] => idx.index
        case idx:BDV[_] => (0 until idx.length).toArray
      }
    val data = docTopicCounter.data

    // val termSum = numTokens - 1D + alphaAS * numTopics
    val betaSum = numTerms * beta
    var i = 0
    var lastSum = 0F

    while (i < used) {
      val topic = index(i)
      val count: Float = data(i)
      val adjustment = if (currentTopic == topic) -1F else 0
      // val lastD = count * termSum * (termTopicCounter(topic) + beta) /
      //   ((totalTopicCounter(topic) + betaSum) * termSum)

      val lastD = (count + adjustment) * (termTopicCounter(topic) + adjustment + beta) /
        (totalTopicCounter(topic) + adjustment + betaSum)
      lastSum += lastD
      d(topic) = lastSum
      i += 1
    }
    d(numTopics - 1) = lastSum
  }

  /**
   * A multinomial distribution sampler, using roulette method to sample an Int back.
   */
  private def multinomialDistSampler(
                                      gen: Random,
                                      docTopicCounter: VD,
                                      d: BDV[Float],
                                      w: BSV[Float],
                                      t: BDV[Float]): Int = {
    val numTopics = d.length
    val lastSum = t(numTopics - 1) + w.data(w.used - 1) + d(numTopics - 1)
    val distSum = gen.nextFloat() * lastSum
    val fun = index(docTopicCounter, d, w, t) _
    val topic = binarySearchInterval[Float](fun, distSum, 0, numTopics, true)
    math.min(topic, numTopics - 1)
  }

  private def index(
                     docTopicCounter: VD,
                     d: BDV[Float],
                     w: BSV[Float],
                     t: BDV[Float])(i: Int) = {
    val lastDS = binarySearchDenseVector(i, docTopicCounter, d)
    val lastWS = binarySearchSparseVector(i, w)
    val lastTS = t(i)
    lastDS + lastWS + lastTS
  }

  private[topicModeling] def binarySearchInterval[K](
                                            index: Int => K,
                                            key: K,
                                            begin: Int,
                                            end: Int,
                                            greater: Boolean)(implicit ord: Ordering[K], ctag: ClassTag[K]): Int = {
    if (begin == end) {
      return if (greater) end else begin - 1
    }
    var b = begin
    var e = end - 1

    var mid: Int = (e + b) >> 1
    while (b <= e) {
      mid = (e + b) >> 1
      val v = index(mid)
      if (ord.lt(v, key)) {
        b = mid + 1
      }
      else if (ord.gt(v, key)) {
        e = mid - 1
      }
      else {
        return mid
      }
    }

    val v = index(mid)
    mid = if ((greater && ord.gteq(v, key)) || (!greater && ord.lteq(v, key))) {
      mid
    }
    else if (greater) {
      mid + 1
    }
    else {
      mid - 1
    }

    if (greater) {
      if (mid < end) assert(ord.gteq(index(mid), key))
      if (mid > 0) assert(ord.lteq(index(mid - 1), key))
    } else {
      if (mid > 0) assert(ord.lteq(index(mid), key))
      if (mid < end - 1) assert(ord.gteq(index(mid + 1), key))
    }
    mid
  }

  private[topicModeling] def binarySearchArray[K](
                                         index: Array[K],
                                         key: K,
                                         begin: Int,
                                         end: Int,
                                         greater: Boolean)(implicit ord: Ordering[K], ctag: ClassTag[K]): Int = {
    binarySearchInterval(i => index(i), key, begin, end, greater)
  }

  private[topicModeling] def binarySearchSparseVector(topic: Int, sv: BSV[Float]) = {
    val index = sv.index
    val used = sv.used
    val data = sv.data
    val pos = binarySearchArray(index, topic, 0, used, false)
    if (pos > -1) data(pos) else 0F
  }

  private[topicModeling] def binarySearchDenseVector[V](
                                               topic: Int,
                                               sv: StorageVector[V],
                                               dv: BDV[Float]): Float = {
    val index =
      sv match {
        case sv:BDV[_] => (0 until sv.length).toArray
        case sv:BSV[_] => sv.index
      }
    val used = sv.activeSize
    val pos = binarySearchArray(index, topic, 0, used, false)
    if (pos > -1) dv(index(pos)) else 0F
  }
}
