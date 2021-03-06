package com.nitro.scalda.examples

import java.io.File

import com.nitro.scalda.models.OnlineLdaParams
import com.nitro.scalda.models.onlineLDA.distributed.DistributedOnlineLda
import org.apache.spark.{ SparkConf, SparkContext }
import breeze.stats.distributions.Gamma
import org.apache.spark.mllib.linalg.{Vector, Vectors}

object DistributedOnlineLdaExample extends App {

  val corpusLoc = new File(args(0))
  val vocabFile = new File(args(1))
  val mbSize = args(2).toInt
  val numTopics = args(3).toInt
  val numDocs = args(4).toInt

  log(
    s"""[DistributedOnlineLdaExample]
       |Text file corpus directory: $corpusLoc
       |Vocabulary file:            $vocabFile
       |Minibatch size:             $mbSize
       |Number of topics:           $numTopics
       |Corpus size:                $numDocs
       |----------------------------
     """.stripMargin
  )
  
  val test = Gamma(100.0, 1.0 / 100.0).sample(10).toArray
  test.foreach { println }
  
  implicit val sc = new SparkContext(
    new SparkConf()
      .setAppName("Distributed Online LDA Example")
      .setMaster("local[3]")
  )
  
  val lda = new DistributedOnlineLda(
    OnlineLdaParams(
      vocabulary = lines(vocabFile).toIndexedSeq,
      alpha = Vectors.dense(-1),
      eta = 1.0 / numTopics,
      decay = 1024,
      learningRate = 0.7,
      maxIter = 100,
      convergenceThreshold = 0.001,
      numTopics = numTopics,
      totalDocs = numDocs
    )
  )

  val model = lda.inference(new TextFileRddIterator(corpusLoc, mbSize))
  lda.saveModel(model, new File("distributedModel"))

  println("<-------------TOPICS LEARNED--------------->")
  lda.printTopics(model)

  sc.stop()
}