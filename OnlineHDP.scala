package org.apache.spark.mllib.topicModeling

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV, _}
import breeze.numerics.{abs, digamma, exp, _}
import org.apache.spark.mllib.linalg.{SparseVector, Vector}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

class SuffStats(
                 val T: Int,
                 val Wt: Int,
                 val m_chunksize: Int) extends Serializable {
  var m_var_sticks_ss = BDV.zeros[Double](T)
  var m_var_beta_ss = BDM.zeros[Double](T, Wt)

  def set_zero: Unit = {
    m_var_sticks_ss = BDV.zeros(T)
    m_var_beta_ss = BDM.zeros(T, Wt)
  }
}

object OnlineHDPOptimizer extends Serializable {
  val rhot_bound = 0.0

  def log_normalize(v: BDV[Double]): (BDV[Double], Double) = {
    val log_max = 100.0
    val max_val = v.toArray.max
    val log_shift = log_max - log(v.size + 1.0) - max_val
    val tot: Double = sum(exp(v + log_shift))
    val log_norm = log(tot) - log_shift
    (v - log_norm, log_norm)
  }

  def log_normalize(m: BDM[Double]): (BDM[Double], BDV[Double]) = {
    val log_max = 100.0
    // get max for every row
    val max_val: BDV[Double] = m(*, ::).map(v => max(v))
    val log_shift: BDV[Double] = log_max - log(m.cols + 1.0) - max_val

    val m_shift: BDM[Double] = exp(m(::, *) + log_shift)
    val tot: BDV[Double] = sum(m_shift(*, ::))

    val log_norm: BDV[Double] = log(tot) - log_shift
    (m(::, *) - log_norm, log_norm)
  }

  def expect_log_sticks(m: BDM[Double]): BDV[Double] = {
    //    """
    //    For stick-breaking hdp, return the E[log(sticks)]
    //    """
    val column = sum(m(::, *))

    val dig_sum: BDV[Double] = digamma(column.toDenseVector)
    val ElogW: BDV[Double] = digamma(m(0, ::).inner) - dig_sum
    val Elog1_W: BDV[Double] = digamma(m(1, ::).inner) - dig_sum
    //
    val n = m.cols + 1
    val Elogsticks = BDV.zeros[Double](n)
    Elogsticks(0 until n - 1) := ElogW(0 until n - 1)
    val cs = accumulate(Elog1_W)

    Elogsticks(1 to n - 1) := Elogsticks(1 to n - 1) + cs
    Elogsticks
  }

  /**
   * For theta ~ Dir(alpha), computes E[log(theta)] given alpha. Currently the implementation
   * uses digamma which is accurate but expensive.
   */
  private def dirichletExpectation(alpha: BDM[Double]): BDM[Double] = {
    val rowSum = sum(alpha(breeze.linalg.*, ::))
    val digAlpha = digamma(alpha)
    val digRowSum = digamma(rowSum)
    val result = digAlpha(::, breeze.linalg.*) - digRowSum
    result
  }

}

/**
 * Implemented based on the paper "Online Variational Inference for the Hierarchical Dirichlet Process" (Chong Wang, John Paisley and David M. Blei)
 */

class OnlineHDPOptimizer(
                          val corpus: RDD[(Long, Vector)],
                          val chunkSize: Int = 256,
                          val m_kappa: Double = 1.0,
                          var m_tau: Double = 64.0,
                          val m_K: Int = 15,
                          val m_T: Int = 150,
                          val m_alpha: Double = 1,
                          val m_gamma: Double = 1,
                          val m_eta: Double = 0.01,
                          val m_scale: Double = 1.0,
                          val m_var_converge: Double = 0.0001,
                          val iterations: Int = 10
                          ) extends Serializable {


  val lda_alpha: Double = 1D
  val lda_beta: Double = 1D
  val m_W: Int = corpus.first()._2.size
  var m_D: Double = corpus.count().toDouble

  val m_var_sticks = BDM.zeros[Double](2, m_T - 1) // 2 * T - 1
  m_var_sticks(0, ::) := 1.0
  m_var_sticks(1, ::) := new BDV[Double]((m_T - 1 to 1 by -1).map(_.toDouble).toArray).t
  var m_varphi_ss: BDV[Double] = BDV.zeros[Double](m_T) // T

  // T * W
  val m_lambda: BDM[Double] = BDM.rand(m_T, m_W) * (m_D.toDouble) * 100.0 / (m_T * m_W).toDouble - m_eta

  // T * W
  val m_Elogbeta = OnlineHDPOptimizer.dirichletExpectation(m_lambda + m_eta)

  m_tau = m_tau + 1
  var m_updatect = 0
  var m_status_up_to_date = true

  val m_timestamp: BDV[Int] = BDV.zeros[Int](m_W)
  val m_r = collection.mutable.MutableList[Double](0)
  var m_lambda_sum = sum(m_lambda(*, ::)) // row sum

  val rhot_bound = 0.0


  def update(docs: RDD[(Long, Vector)]): Unit = {
    for (i <- 1 to iterations) {
      val chunk = docs
      update_chunk(chunk)
    }
  }


  def update_chunk(chunk: RDD[(Long, Vector)], update: Boolean = true): (Double, Int) = {
    // Find the unique words in this chunk...
    val unique_words = scala.collection.mutable.Map[Int, Int]()
    val raw_word_list = ArrayBuffer[Int]()
    chunk.collect().foreach(doc => {
      doc._2.foreachActive { case (id, count) =>
        if (count > 0 && !unique_words.contains(id)) {
          unique_words += (id -> unique_words.size)
          raw_word_list.append(id)
        }
      }
    })

    val word_list = raw_word_list.toList

    val Wt = word_list.length // length of words in these documents

    // ...and do the lazy updates on the necessary columns of lambda
    //    rw = np.array([self.m_r[t] for t in self.m_timestamp[word_list]])
    //    self.m_lambda[:, word_list] *= np.exp(self.m_r[-1] - rw)
    //    self.m_Elogbeta[:, word_list] = \
    //    sp.psi(self.m_eta + self.m_lambda[:, word_list]) - \
    //    sp.psi(self.m_W * self.m_eta + self.m_lambda_sum[:, np.newaxis])


    val rw: BDV[Double] = new BDV(word_list.map(id => m_timestamp(id)).map(t => m_r(t)).toArray)

    val exprw: BDV[Double] = exp(rw.map(d => m_r.last - d))

    val wordsMatrix = m_lambda(::, word_list).toDenseMatrix
    for (row <- 0 until wordsMatrix.rows) {
      wordsMatrix(row, ::) := (wordsMatrix(row, ::).t :* exprw).t
    }
    m_lambda(::, word_list) := wordsMatrix

    for (id <- word_list) {
      m_Elogbeta(::, id) := digamma(m_lambda(::, id) + m_eta) - digamma(m_lambda_sum + m_W * m_eta)
    }

    val ss = new SuffStats(m_T, Wt, chunk.count().toInt)

    val Elogsticks_1st: BDV[Double] = OnlineHDPOptimizer.expect_log_sticks(m_var_sticks) // global sticks

    // run variational inference on some new docs
    var score = 0.0
    var count = 0D
    chunk.collect().foreach(doc =>
      if (doc._2.size > 0) {
        val doc_word_ids = doc._2.asInstanceOf[SparseVector].indices
        val doc_word_counts = doc._2.asInstanceOf[SparseVector].values
        val dict = unique_words.toMap
        val wl = doc_word_ids.toList

        val doc_score = doc_e_step(doc, ss, Elogsticks_1st,
          word_list, dict, wl,
          new BDV[Double](doc_word_counts), m_var_converge)
        count += sum(doc_word_counts)
        score += doc_score
      }
    )
    if (update) {
      update_lambda(ss, word_list)
    }

    (score, count.toInt)
  }


  def update_lambda(sstats: SuffStats, word_list: List[Int]): Unit = {
    m_status_up_to_date = false
    // rhot will be between 0 and 1, and says how much to weight
    // the information we got from this mini-chunk.
    var rhot = m_scale * pow(m_tau + m_updatect, -m_kappa)
    if (rhot < rhot_bound)
      rhot = rhot_bound

    // Update appropriate columns of lambda based on documents.
    // T * Wt                    T * Wt                                      T * Wt
    m_lambda(::, word_list) := (m_lambda(::, word_list).toDenseMatrix * (1 - rhot)) + sstats.m_var_beta_ss * rhot * m_D / sstats.m_chunksize.toDouble
    m_lambda_sum = (1 - rhot) * m_lambda_sum + sum(sstats.m_var_beta_ss(*, ::)) * rhot * m_D / sstats.m_chunksize.toDouble

    m_updatect += 1
    m_timestamp(word_list) := m_updatect
    m_r += (m_r.last + log(1 - rhot))

    // T
    m_varphi_ss = (1.0 - rhot) * m_varphi_ss + rhot * sstats.m_var_sticks_ss * m_D.toDouble / sstats.m_chunksize.toDouble
    // update top level sticks
    // 2 * T - 1
    m_var_sticks(0, ::) := (m_varphi_ss(0 to m_T - 2) + 1.0).t
    val var_phi_sum = flipud(m_varphi_ss(1 to m_varphi_ss.length - 1)) // T - 1
    m_var_sticks(1, ::) := (flipud(accumulate(var_phi_sum)) + m_gamma).t

  }


  def doc_e_step(doc: (Long, Vector),
                 ss: SuffStats,
                 Elogsticks_1st: BDV[Double],
                 word_list: List[Int],
                 unique_words: Map[Int, Int],
                 doc_word_ids: List[Int],
                 doc_word_counts: BDV[Double],
                 var_converge: Double): Double = {

    val chunkids = doc_word_ids.map(id => unique_words(id))

    val Elogbeta_doc: BDM[Double] = m_Elogbeta(::, doc_word_ids).toDenseMatrix // T * Wt
    // very similar to the hdp equations, 2 * K - 1
    val v = BDM.zeros[Double](2, m_K - 1)
    v(0, ::) := 1.0
    v(1, ::) := m_alpha

    var Elogsticks_2nd = OnlineHDPOptimizer.expect_log_sticks(v)

    // back to the uniform
    var phi: BDM[Double] = BDM.ones[Double](doc_word_ids.size, m_K) * 1.0 / m_K.toDouble // Wt * K

    var likelihood = 0.0
    var old_likelihood = -1e200
    val converge = 1.0
    val eps = 1e-100

    var iter = 0
    val max_iter = 100

    var var_phi_out: BDM[Double] = BDM.ones[Double](1, 1)

    // not yet support second level optimization yet, to be done in the future
    while (iter < max_iter && (converge < 0.0 || converge > var_converge)) {

      // var_phi
      val (log_var_phi: BDM[Double], var_phi: BDM[Double]) =
        if (iter < 3) {
          val element = Elogbeta_doc.copy // T * Wt
          for (i <- 0 to element.rows - 1) {
            element(i, ::) := (element(i, ::).t :* doc_word_counts).t
          }
          var var_phi: BDM[Double] = phi.t * element.t // K * Wt   *  Wt * T  => K * T
          val (log_var_phi, log_norm) = OnlineHDPOptimizer.log_normalize(var_phi)
          var_phi = exp(log_var_phi)
          (log_var_phi, var_phi)
        }
        else {
          val element = Elogbeta_doc.copy
          for (i <- 0 to element.rows - 1) {
            element(i, ::) := (element(i, ::).t :* doc_word_counts).t
          }
          val product: BDM[Double] = phi.t * element.t
          for (i <- 0 until product.rows) {
            product(i, ::) := (product(i, ::).t + Elogsticks_1st).t
          }

          var var_phi: BDM[Double] = product
          val (log_var_phi, log_norm) = OnlineHDPOptimizer.log_normalize(var_phi)
          var_phi = exp(log_var_phi)
          (log_var_phi, var_phi)
        }

      val (log_phi, log_norm) =
      // phi
        if (iter < 3) {
          phi = (var_phi * Elogbeta_doc).t
          val (log_phi, log_norm) = OnlineHDPOptimizer.log_normalize(phi)
          phi = exp(log_phi)
          (log_phi, log_norm)
        }
        else {
          //     K * T       T * Wt
          val product: BDM[Double] = (var_phi * Elogbeta_doc).t
          for (i <- 0 until product.rows) {
            product(i, ::) := (product(i, ::).t + Elogsticks_2nd).t
          }
          phi = product
          val (log_phi, log_norm) = OnlineHDPOptimizer.log_normalize(phi)
          phi = exp(log_phi)
          (log_phi, log_norm)
        }


      // v
      val phi_all = phi.copy
      for (i <- 0 until phi_all.cols) {
        phi_all(::, i) := (phi_all(::, i)) :* doc_word_counts
      }

      v(0, ::) := sum(phi_all(::, m_K - 1)) + 1.0
      val selected = phi_all(::, 1 until m_K)
      val t_sum = sum(selected(::, *)).toDenseVector
      val phi_cum = flipud(t_sum)
      v(1, ::) := (flipud(accumulate(phi_cum)) + m_alpha).t
      Elogsticks_2nd = OnlineHDPOptimizer.expect_log_sticks(v)

      likelihood = 0.0
      // compute likelihood
      // var_phi part/ C in john's notation

      val diff = log_var_phi.copy
      for (i <- 0 to diff.rows - 1) {
        diff(i, ::) := (Elogsticks_1st :- diff(i, ::).t).t
      }

      likelihood += sum(diff :* var_phi)

      // v part/ v in john's notation, john's beta is alpha here
      val log_alpha = log(m_alpha)
      likelihood += (m_K - 1) * log_alpha
      val dig_sum = (digamma(sum(v(::, *)))).toDenseVector
      val vCopy = v.copy
      for (i <- 0 until v.cols) {
        vCopy(::, i) := BDV[Double](1.0, m_alpha) - vCopy(::, i)
      }

      val dv = digamma(v)
      for (i <- 0 until v.rows) {
        dv(i, ::) := dv(i, ::) - dig_sum.t
      }

      likelihood += sum(vCopy :* dv)
      likelihood -= sum(lgamma(sum(v(::, *)))) - sum(lgamma(v))

      // Z part
      val log_phiCopy = log_phi.copy
      for (i <- 0 until log_phiCopy.rows) {
        log_phiCopy(i, ::) := (Elogsticks_2nd - log_phiCopy(i, ::).t).t
      }
      likelihood += sum(log_phiCopy :* phi)

      // X part, the data part
      val Elogbeta_docCopy = Elogbeta_doc.copy
      for (i <- 0 until Elogbeta_docCopy.rows) {
        Elogbeta_docCopy(i, ::) := (Elogbeta_docCopy(i, ::).t :* doc_word_counts).t
      }

      likelihood += sum(phi.t :* (var_phi * Elogbeta_docCopy))

      val converge = (likelihood - old_likelihood) / abs(old_likelihood)
      old_likelihood = likelihood

      if (converge < -0.000001)
        println("likelihood is decreasing!")

      iter += 1
      var_phi_out = var_phi
    }

    // update the suff_stat ss
    // this time it only contains information from one doc
    val sumPhiOut = sum(var_phi_out(::, *))
    ss.m_var_sticks_ss += sumPhiOut.toDenseVector

    val phiCopy = phi.copy.t
    for (i <- 0 until phi.rows) {
      phiCopy(i, ::) := (phiCopy(i, ::).t :* doc_word_counts).t
    }

    val middleResult: BDM[Double] = var_phi_out.t * phiCopy
    for (i <- 0 until chunkids.size) {
      ss.m_var_beta_ss(::, chunkids(i)) := ss.m_var_beta_ss(::, chunkids(i)) + middleResult(::, i)
    }

    return likelihood
  }

}