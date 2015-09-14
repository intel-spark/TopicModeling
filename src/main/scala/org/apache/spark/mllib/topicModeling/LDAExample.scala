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

import java.text.BreakIterator

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import scala.collection.mutable
import scala.reflect.runtime.universe._

/**
 * Abstract class for parameter case classes.
 * This overrides the [[toString]] method to print all case class fields by name and value.
 * @tparam T  Concrete parameter class.
 */
abstract class AbstractParams[T: TypeTag] {

  private def tag: TypeTag[T] = typeTag[T]

  /**
   * Finds all case class fields in concrete class instance, and outputs them in JSON-style format:
   * {
   *   [field name]:\t[field value]\n
   *   [field name]:\t[field value]\n
   *   ...
   * }
   */
  override def toString: String = {
    val tpe = tag.tpe
    val allAccessors = tpe.declarations.collect {
      case m: MethodSymbol if m.isCaseAccessor => m
    }
    val mirror = runtimeMirror(getClass.getClassLoader)
    val instanceMirror = mirror.reflect(this)
    allAccessors.map { f =>
      val paramName = f.name.toString
      val fieldMirror = instanceMirror.reflectField(f)
      val paramValue = fieldMirror.get
      s"  $paramName:\t$paramValue"
    }.mkString("{\n", ",\n", "\n}")
  }
}

/**
 * An example Latent Dirichlet Allocation (LDA) app. Run with
 * {{{
 * ./bin/run-example mllib.LDAExample [options] <input>
 * }}}
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object LDAExample {

  private case class Params(
     input: Seq[String] = Seq.empty,
     k: Int = 20,
     maxIterations: Int = 10,
     maxInnerIterations: Int = 5,
     docConcentration: Double = 0.01,
     topicConcentration: Double = 0.01,
     vocabSize: Int = 10000,
     stopwordFile: String = "",
     checkpointDir: Option[String] = None,
     checkpointInterval: Int = 10,
     optimizer:String = "em",
     gibbsSampler:String = "alias",
     gibbsAlphaAS:Double = 0.1,
     gibbsPrintPerplexity:Boolean = false,
     gibbsEdgePartitioner:String = "none",
     partitions:Int = 2,
     logLevel:String = "info",
     psMasterAddr:String = null
     ) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("LDAExample") {
      head("LDAExample: an example LDA app for plain text data.")
      opt[Int]("k")
        .text(s"number of topics. default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[Int]("maxIterations")
        .text(s"number of iterations of learning. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
      opt[Int]("maxInnerIterations")
        .text(s"number of inner iterations of learning. default: ${defaultParams.maxInnerIterations}")
        .action((x, c) => c.copy(maxInnerIterations = x))
      opt[Double]("docConcentration")
        .text(s"amount of topic smoothing to use (> 1.0) (-1=auto)." +
        s"  default: ${defaultParams.docConcentration}")
        .action((x, c) => c.copy(docConcentration = x))
      opt[Double]("topicConcentration")
        .text(s"amount of term (word) smoothing to use (> 1.0) (-1=auto)." +
        s"  default: ${defaultParams.topicConcentration}")
        .action((x, c) => c.copy(topicConcentration = x))
      opt[Int]("vocabSize")
        .text(s"number of distinct word types to use, chosen by frequency. (-1=all)" +
        s"  default: ${defaultParams.vocabSize}")
        .action((x, c) => c.copy(vocabSize = x))
      opt[String]("stopwordFile")
        .text(s"filepath for a list of stopwords. Note: This must fit on a single machine." +
        s"  default: ${defaultParams.stopwordFile}")
        .action((x, c) => c.copy(stopwordFile = x))
      opt[String]("checkpointDir")
        .text(s"Directory for checkpointing intermediate results." +
        s"  Checkpointing helps with recovery and eliminates temporary shuffle files on disk." +
        s"  default: ${defaultParams.checkpointDir}")
        .action((x, c) => c.copy(checkpointDir = Some(x)))
      opt[Int]("checkpointInterval")
        .text(s"Iterations between each checkpoint.  Only used if checkpointDir is set." +
        s" default: ${defaultParams.checkpointInterval}")
        .action((x, c) => c.copy(checkpointInterval = x))
      opt[String]("optimizer")
        .text(s"available optimizer are online and gibbs, default: ${defaultParams.optimizer}")
        .action((x, c) => c.copy(optimizer = x))
      opt[String]("gibbs.sampler")
        .text(s"sampler for gibbs optimizer, available options are alias, sparse, light and fast, default: ${defaultParams.gibbsSampler}")
        .action((x, c) => c.copy(gibbsSampler = x))
      opt[Double]("gibbs.alphaAS")
        .text(s"alphaAS for gibbs optimizer, default: ${defaultParams.gibbsAlphaAS}")
        .action((x, c) => c.copy(gibbsAlphaAS = x))
      opt[Boolean]("gibbs.printPerplexity")
        .text(s"print perplexity for gibbs optimizer, default: ${defaultParams.gibbsPrintPerplexity}")
        .action((x, c) => c.copy(gibbsPrintPerplexity = x))
      opt[String]("gibbs.edgePartitioner")
        .text(s"edge partitioner for gibbs optimizer, available options are none, degree, and even, default: ${defaultParams.gibbsEdgePartitioner}}")
        .action((x, c) => c.copy(gibbsEdgePartitioner = x))
      opt[Int]("partitions")
        .text(s"Minimum edge partitions, default: ${defaultParams.partitions}")
        .action((x, c) => c.copy(partitions = x))
      opt[String]("logLevel")
        .text(s"Log level, default: ${defaultParams.logLevel}")
        .action((x, c) => c.copy(logLevel = x))
      opt[String]("psMasterAddr")
        .text(s"psMaster address, default: ${defaultParams.psMasterAddr}")
        .action((x, c) => c.copy(psMasterAddr = x))
      arg[String]("<input>...")
        .text("input paths (directories) to plain text corpora." +
        "  Each text file line should hold 1 document.")
        .unbounded()
        .required()
        .action((x, c) => c.copy(input = c.input :+ x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      parser.showUsageAsError
      sys.exit(1)
    }
  }

//    private def createOptimizer(params: Params, lineRdd: RDD[Int], columnRdd: RDD[Int]):LDAOptimizer = {
  private def createOptimizer(params: Params):LDAOptimizer = {
    params.optimizer match {
      case "online" => val optimizer = new OnlineLDAOptimizer
                        optimizer
      case "gibbs" =>
        val optimizer = new GibbsLDAOptimizer
        optimizer.setSampler(params.gibbsSampler)
        optimizer.printPerplexity = params.gibbsPrintPerplexity
        optimizer.edgePartitioner = params.gibbsEdgePartitioner
        optimizer.setAlphaAS(params.gibbsAlphaAS.toFloat)
        optimizer
      case _ =>
        throw new IllegalArgumentException(s"available optimizers are em, online and gibbs, but got ${params.optimizer}")
    }
  }

  /**
   * run LDA
   * @param params
   */
  private def run(params: Params) {
    val conf = new SparkConf().setAppName(s"LDAExample with $params")
    val sc = new SparkContext(conf)

    val logLevel = Level.toLevel(params.logLevel, Level.INFO)
    Logger.getRootLogger.setLevel(logLevel)
    println(s"Setting log level to $logLevel")

    // Load documents, and prepare them for LDA.
    val preprocessStart = System.nanoTime()
    val (corpus, actualCorpusSize, vocabArray, actualNumTokens) =
      preprocess(sc, params.input, params.vocabSize, params.partitions, params.stopwordFile)
    val actualVocabSize = vocabArray.size
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9

    println()
    println(s"Corpus summary:")
    println(s"\t Training set size: $actualCorpusSize documents")
    println(s"\t Vocabulary size: $actualVocabSize terms")
    println(s"\t Training set size: $actualNumTokens tokens")
    println(s"\t Preprocessing time: $preprocessElapsed sec")
    println()

    // Run LDA.
    val lda = new LDA()
    lda.setK(params.k)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .setCheckpointInterval(params.checkpointInterval)
      .setOptimizer(createOptimizer(params))

    if (params.checkpointDir.nonEmpty) {
      sc.setCheckpointDir(params.checkpointDir.get)
    }

    val startTime = System.nanoTime()
    val ldaModel = lda.run(corpus)
    val elapsed = (System.nanoTime() - startTime) / 1e9

    println(s"Finished training LDA model using ${lda.getOptimizer.getClass.getName}")

    // Print the topics, showing the top-weighted terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)

    val topics = topicIndices.map { case (terms, termWeights) =>
      terms.zip(termWeights).map { case (term, weight) => (vocabArray(term.toInt), weight) }
    }
    println(s"${params.k} topics:")
    topics.zipWithIndex.foreach { case (topic, i) =>
      println(s"TOPIC $i")
      topic.foreach { case (term, weight) =>
        println(s"$term\t$weight")
      }
      println()
    }
    sc.stop()
  }

  /**
   * Load documents, tokenize them, create vocabulary, and prepare documents as term count vectors.
   * @return (corpus, vocabulary as array, total token count in corpus)
   */
  private def preprocess(
      sc: SparkContext,
      paths: Seq[String],
      vocabSize: Int,
      partitions:Int,
      stopwordFile: String): (RDD[(Long, Vector)], Long, Array[String], Long) = {

    // Get dataset of document texts
    // One document per line in each text file. If the input consists of many small files,
    // this can result in a large number of small partitions, which can degrade performance.
    // In this case, consider using coalesce() to create fewer, larger partitions.
    val textRDD: RDD[String] = sc.textFile(paths.mkString(","), Math.max(1, partitions)).coalesce(partitions)


    // Split text into words
    val tokenizer = new SimpleTokenizer(sc, stopwordFile)
    val tokenized: RDD[(Long, IndexedSeq[String])] = textRDD.zipWithIndex().map { case (text, id) =>
      id -> tokenizer.getWords(text)
    }
    tokenized.cache()

    // Counts words: RDD[(word, wordCount)]
    val wordCounts: RDD[(String, Long)] = tokenized
      .flatMap { case (_, tokens) => tokens.map(_ -> 1L) }
      .reduceByKey(_ + _)
    wordCounts.cache()
    val fullVocabSize = wordCounts.count()
    // Select vocab
    //  (vocab: Map[word -> id], total tokens after selecting vocab)
    val (vocab: Map[String, Int], selectedTokenCount: Long) = {
      val tmpSortedWC: Array[(String, Long)] = if (vocabSize == -1 || fullVocabSize <= vocabSize) {
        // Use all terms
        wordCounts.collect().sortBy(-_._2)
      } else {
        // Sort terms to select vocab
        wordCounts.sortBy(_._2, ascending = false).take(vocabSize)
      }
      (tmpSortedWC.map(_._1).zipWithIndex.toMap, tmpSortedWC.map(_._2).sum)
    }

    val documents = tokenized.map { case (id, tokens) =>
      // Filter tokens by vocabulary, and create word count vector representation of document.
      val wc = new mutable.HashMap[Int, Int]()
      tokens.foreach { term =>
        if (vocab.contains(term)) {
          val termIndex = vocab(term)
          wc(termIndex) = wc.getOrElse(termIndex, 0) + 1
        }
      }
      val indices = wc.keys.toArray.sorted
      val values = indices.map(i => wc(i).toDouble)

      val sb = Vectors.sparse(vocab.size, indices, values)
      (id, sb)
    }.filter(_._2.asInstanceOf[SparseVector].values.length > 0).cache
    val corpusSize = documents.count
    tokenized.unpersist(false)

    val vocabArray = new Array[String](vocab.size)
    vocab.foreach { case (term, i) => vocabArray(i) = term }

    (documents, corpusSize, vocabArray, selectedTokenCount)
  }
}

/**
 * Simple Tokenizer.
 *
 * TODO: Formalize the interface, and make this a public class in mllib.feature
 */
private class SimpleTokenizer(sc: SparkContext, stopwordFile: String) extends Serializable {

  private val stopwords: Set[String] = if (stopwordFile.isEmpty) {
    Set.empty[String]
  } else {
    val stopwordText = sc.textFile(stopwordFile).collect()
    stopwordText.flatMap(_.stripMargin.split("\\s+")).toSet
  }

  // Matches sequences of Unicode letters
  private val allWordRegex = "^(\\p{L}*)$".r

  // Ignore words shorter than this length.
  private val minWordLength = 3

  def getWords(text: String): IndexedSeq[String] = {

    val words = new mutable.ArrayBuffer[String]()

    // Use Java BreakIterator to tokenize text into words.
    val wb = BreakIterator.getWordInstance
    wb.setText(text)

    // current,end index start,end of each word
    var current = wb.first()
    var end = wb.next()
    while (end != BreakIterator.DONE) {
      // Convert to lowercase
      val word: String = text.substring(current, end).toLowerCase
      // Remove short words and strings that aren't only letters
      word match {
        case allWordRegex(w) if w.length >= minWordLength && !stopwords.contains(w) =>
          words += w
        case _ =>
      }

      current = end
      try {
        end = wb.next()
      } catch {
        case e: Exception =>
          // Ignore remaining text in line.
          // This is a known bug in BreakIterator (for some Java versions),
          // which fails when it sees certain characters.
          end = BreakIterator.DONE
      }
    }
    words
  }

}
