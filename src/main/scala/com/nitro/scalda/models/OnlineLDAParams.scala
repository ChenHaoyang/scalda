package com.nitro.scalda.models

import org.apache.spark.mllib.linalg.{Vector, Vectors}

case class OnlineLdaParams(
  vocabulary: IndexedSeq[String],
  alpha: Vector = Vectors.dense(0),
  eta: Double,
  decay: Double,
  learningRate: Double,
  maxIter: Int,
  convergenceThreshold: Double,
  numTopics: Int,
  totalDocs: Int,
  lemmatize: Boolean = false,
  perplexity: Boolean = false
)