package org.apache.spark.mllib.topicModeling

import java.util.Random

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum, max}
import breeze.numerics.{abs, digamma, exp, log, lgamma}
import breeze.stats.distributions.{Gamma, RandBasis}
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.collection.AppendOnlyMap
import scala.collection.mutable.ArrayBuffer

/**
 * :: Experimental ::
 *
 * An Optimizer based on OnlineLDAOptimizer, that can also output the the document~topic
 * distribution
 *
 * An early version of the implementation was merged into MLlib (PR #4419), and several extensions (e.g., predict) are added here
 *
 */
object OnlineLDAOptimizer{
  def inference(
       k: Int,
       vocabSize: Int,
       lambda: BDM[Double],
       alpha: Double,
       gammaShape: Double,
       batch: RDD[(Long, Vector)]):
  (BDM[Double], RDD[BDM[Double]], RDD[(Long, BDV[Double])]) = {

    val Elogbeta = dirichletExpectation(lambda)
    val expElogbeta = exp(Elogbeta)

    val statsAndgammaArray: (RDD[(BDM[Double], Array[(Long, BDV[Double])])]) =
      batch.mapPartitions { docs =>
        val stat = BDM.zeros[Double](k, vocabSize)
        val gammaList = new ArrayBuffer[(Long, BDV[Double])]
        docs.foreach { doc =>
          val termCounts = doc._2
          val (ids: List[Int], cts: Array[Double]) = termCounts match {
            case v: DenseVector => ((0 until v.size).toList, v.values)
            case v: SparseVector => (v.indices.toList, v.values)
            case v => throw new IllegalArgumentException("Online LDA does not support vector type "
              + v.getClass)
          }

          // Initialize the variational distribution q(theta|gamma) for the mini-batch
          var gammad = new Gamma(gammaShape, 1.0 / gammaShape).samplesVector(k).t // 1 * K
          var Elogthetad = digamma(gammad) - digamma(sum(gammad))     // 1 * K
          var expElogthetad = exp(Elogthetad)                         // 1 * K
          val expElogbetad = expElogbeta(::, ids).toDenseMatrix       // K * ids

          var phinorm = expElogthetad * expElogbetad + 1e-100         // 1 * ids
          var meanchange = 1D
          val ctsVector = new BDV[Double](cts).t                      // 1 * ids

          // Iterate between gamma and phi until convergence
          while (meanchange > 1e-3) {
            val lastgamma = gammad
            //        1*K                  1 * ids               ids * k
            gammad = (expElogthetad :* ((ctsVector / phinorm) * expElogbetad.t)) + alpha
            Elogthetad = digamma(gammad) - digamma(sum(gammad))
            expElogthetad = exp(Elogthetad)
            phinorm = expElogthetad * expElogbetad + 1e-100
            meanchange = sum(abs(gammad - lastgamma)) / k
          }

          gammaList += Tuple2(doc._1, gammad.t)
          val m1 = expElogthetad.t
          val m2 = (ctsVector / phinorm).t.toDenseVector
          var i = 0
          while (i < ids.size) {
            stat(::, ids(i)) := stat(::, ids(i)) + m1 * m2(i)
            i += 1
          }
        }
        Iterator((stat, gammaList.toArray))
      }

    val stats = statsAndgammaArray.map(_._1)
    val gammaArray = statsAndgammaArray.flatMap(_._2)

    (expElogbeta, stats, gammaArray)
  }

  /**
   * For theta ~ Dir(alpha), computes E[log(theta)] given alpha. Currently the implementation
   * uses digamma which is accurate but expensive.
   */
  def dirichletExpectation(alpha: BDM[Double]): BDM[Double] = {
    val rowSum = sum(alpha(breeze.linalg.*, ::))
    val digAlpha = digamma(alpha)
    val digRowSum = digamma(rowSum)
    val result = digAlpha(::, breeze.linalg.*) - digRowSum
    result
  }
}

final class OnlineLDAOptimizer extends LDAOptimizer with Serializable{
  // LDA common parameters
  private var k: Int = 0
  private var corpusSize: Long = 0
  private var vocabSize: Int = 0

  /** alias for docConcentration */
  private var alpha: Double = 0

  /** (private[lda] for debugging)  Get docConcentration */
  private[topicModeling] def getAlpha: Double = alpha

  /** alias for topicConcentration */
  private var eta: Double = 0

  /** (private[lda] for debugging)  Get topicConcentration */
  private[topicModeling] def getEta: Double = eta

  private var randomGenerator: java.util.Random = null

  // Online LDA specific parameters
  // Learning rate is: (tau0 + t)^{-kappa}
  private var tau0: Double = 1024
  private var kappa: Double = 0.51
  private var miniBatchFraction: Double = 0.05

  // internal data structure
  private var docs: RDD[(Long, Vector)] = null

  /** Dirichlet parameter for the posterior over topics */
  private var lambda: BDM[Double] = null

  /** (private[lda] for debugging) Get parameter for topics */
  private[topicModeling] def getLambda: BDM[Double] = lambda

  /** Current iteration (count of invocations of [[next()]]) */
  private var iteration: Int = 0
  private var gammaShape: Double = 100

  /**
   * A (positive) learning parameter that downweights early iterations. Larger values make early
   * iterations count less.
   */
  def getTau0: Double = this.tau0

  /**
   * A (positive) learning parameter that downweights early iterations. Larger values make early
   * iterations count less.
   * Default: 1024, following the original Online LDA paper.
   */
  def setTau0(tau0: Double): this.type = {
    require(tau0 > 0, s"LDA tau0 must be positive, but was set to $tau0")
    this.tau0 = tau0
    this
  }

  /**
   * Learning rate: exponential decay rate
   */
  def getKappa: Double = this.kappa

  /**
   * Learning rate: exponential decay rate---should be between
   * (0.5, 1.0] to guarantee asymptotic convergence.
   * Default: 0.51, based on the original Online LDA paper.
   */
  def setKappa(kappa: Double): this.type = {
    require(kappa >= 0, s"Online LDA kappa must be nonnegative, but was set to $kappa")
    this.kappa = kappa
    this
  }

  /**
   * Mini-batch fraction, which sets the fraction of document sampled and used in each iteration
   */
  def getMiniBatchFraction: Double = this.miniBatchFraction

  /**
   * Mini-batch fraction in (0, 1], which sets the fraction of document sampled and used in
   * each iteration.
   *
   * Note that this should be adjusted in synch with
   * so the entire corpus is used.  Specifically, set both so that
   * maxIterations * miniBatchFraction >= 1.
   *
   * Default: 0.05, i.e., 5% of total documents.
   */
  def setMiniBatchFraction(miniBatchFraction: Double): this.type = {
    require(miniBatchFraction > 0.0 && miniBatchFraction <= 1.0,
      s"Online LDA miniBatchFraction must be in range (0,1], but was set to $miniBatchFraction")
    this.miniBatchFraction = miniBatchFraction
    this
  }

  /**
   * (private[lda])
   * Set the Dirichlet parameter for the posterior over topics.
   * This is only used for testing now. In the future, it can help support training stop/resume.
   */
  private[topicModeling] def setLambda(lambda: BDM[Double]): this.type = {
    this.lambda = lambda
    this
  }

  /**
   * (private[lda])
   * Used for random initialization of the variational parameters.
   * Larger value produces values closer to 1.0.
   * This is only used for testing currently.
   */
  private[topicModeling] def setGammaShape(shape: Double): this.type = {
    this.gammaShape = shape
    this
  }

  override def initialize(docs: RDD[(Long, Vector)], lda: LDA): this.type = {
    this.k = lda.getK
    this.corpusSize = docs.count()
    this.vocabSize = docs.first()._2.size
    this.alpha = if (lda.getDocConcentration == -1) 1.0 / k else lda.getDocConcentration
    this.eta = if (lda.getTopicConcentration == -1) 1.0 / k else lda.getTopicConcentration
    this.randomGenerator = new Random(lda.getSeed)

    this.docs = docs
    this.lambda = getGammaMatrix(k, vocabSize)

    this.iteration = 0
    this
  }

  def initialize(corpusSize: Long, vocabSize: Int, lda: LDA): this.type ={
    this.k = lda.getK
    this.corpusSize = corpusSize
    this.vocabSize = vocabSize
    this.alpha = if (lda.getDocConcentration == -1) 1.0 / k else lda.getDocConcentration
    this.eta = if (lda.getTopicConcentration == -1) 1.0 / k else lda.getTopicConcentration
    this.randomGenerator = new Random(lda.getSeed)

    // Initialize the variational distribution q(beta|lambda)
    this.lambda = getGammaMatrix(k, vocabSize)
    this.iteration = 0
    this
  }

  override def next(): this.type = {
    val batch = docs.sample(withReplacement = true, miniBatchFraction, randomGenerator.nextLong())
    submitMiniBatch(batch)
  }

  /**
   * Submit a subset (like 1%, decide by the miniBatchFraction) of the corpus to the Online LDA
   * model, and it will update the topic distribution adaptively for the terms appearing in the
   * subset.
   */
  private[topicModeling] def submitMiniBatch(batch: RDD[(Long, Vector)]): this.type = {
    if (batch.isEmpty()) return this
    iteration += 1

    val (expElogbeta, stats, _) =
      OnlineLDAOptimizer.inference(k, vocabSize, lambda, alpha, gammaShape, batch)
    val statsSum: BDM[Double] = stats.reduce(_ += _)
    val batchResult = statsSum :* expElogbeta

    // Note that this is an optimization to avoid batch.count
    update(batchResult, iteration, batch.count().toInt)
    this
  }

  override def getLDAModel(iterationTimes: Array[Double]): LDAModel = {
    new OnlineLDAModel(Matrices.fromBreeze(lambda).transpose, this.alpha, this.gammaShape)
  }

  /**
   * Update lambda based on the batch submitted. batchSize can be different for each iteration.
   */
  private[topicModeling] def update(stat: BDM[Double], iter: Int, batchSize: Int): Unit = {
    // weight of the mini-batch.
    val weight = math.pow(getTau0 + iter, -getKappa)

    // Update lambda based on documents.
    lambda = lambda * (1 - weight) +
      (stat * (corpusSize.toDouble / batchSize.toDouble) + eta) * weight
  }

  /**
   * Get a random matrix to initialize lambda
   */
  private def getGammaMatrix(row: Int, col: Int): BDM[Double] = {
    val randBasis = new RandBasis(new org.apache.commons.math3.random.MersenneTwister(
      randomGenerator.nextLong()))
    val gammaRandomGenerator = new Gamma(gammaShape, 1.0 / gammaShape)(randBasis)
    val temp = gammaRandomGenerator.sample(row * col).toArray
    new BDM[Double](col, row, temp).t
  }

  def perplexity(docs: RDD[(Long, Vector)]): Double = {
    val alphaVector = Vectors.dense(Array.fill(k)(alpha))
    val brzAlpha = alphaVector.toBreeze.toDenseVector

    val Elogbeta = OnlineLDAOptimizer.dirichletExpectation(lambda)

    val (_, _, gammaArray) = OnlineLDAOptimizer.inference(k, vocabSize, lambda, alpha, gammaShape, docs)

    var score = docs.join(gammaArray).map { case (id: Long, (termCounts: Vector, gammad: BDV[Double])) =>
      var docScore = 0.0D

      val Elogthetad: BDV[Double] = digamma(gammad) - digamma(sum(gammad))

      // E[log p(doc | theta, beta)]
      termCounts.foreachActive { case (idx, count) =>
        val x = Elogthetad + Elogbeta(::, idx)
        val a = max(x)
        docScore += count * (a + log(sum(exp(x :- a))))
      }
      // E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
      docScore += sum((brzAlpha - gammad) :* Elogthetad)
      docScore += sum(lgamma(gammad) - lgamma(brzAlpha))
      docScore += lgamma(sum(brzAlpha)) - lgamma(sum(gammad))

      docScore
    }.sum()

    // E[log p(beta | eta) - log q (beta | lambda)]; assumes eta is a scalar
    score += sum((eta - lambda) :* Elogbeta)
    score += sum(lgamma(lambda) - lgamma(eta))

    val sumEta = eta * vocabSize
    score += sum(lgamma(sumEta) - lgamma(sum(lambda(::, breeze.linalg.*))))

    math.exp(-1 * score / vocabSize)
  }
}

