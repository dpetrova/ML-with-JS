/* 
Use Multivariate Linear Regression with Tensorflow 
to find the relation between one dependend variable and many independent variables:
y = b + m1*x1 + m2*x2 + ... + mn*xn 
*/

const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LinearRegression {
  constructor(features, labels, options) {
    //create a tensor of features data, standardize it and append column of "1"
    this.features = this.processFeatures(features);

    //create a tensor of labels data
    this.labels = tf.tensor(labels);

    // override default learning_rate and max_number_of_iterations (times that run gradientDescent algorithm)
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    //initialize b and m as "0" ([b, m1, m2, ..., mn])
    this.weights = tf.zeros([this.features.shape[1], 1]);

    //record MSE to can adjust learning rate
    this.mseHistory = [];
  }

  /* get tensor of standardized features data */
  processFeatures(features) {
    //create tensor of data
    features = tf.tensor(features);

    //standartize
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    //create column of "1" with same shape as features and concat it to features along a column axis
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  /* standardize data with formula (val - average) / stDev */
  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);
    //standartize must be done with same mean and variance so cache them as instance variables
    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  /* calculate slope of MSE with respect to "m" and "b" = Featues * ((Features * Weights) - Labels) / n */
  gradientDescent(features, labels) {
    //^y = (Features * Weights)
    const labelsEstimates = features.matMul(this.weights); //matMul -> matrix multiplication

    //diffs = (Features * Weights) - Labels
    const differences = labelsEstimates.sub(labels);

    //slope = Featues * ((Features * Weights) - Labels) / n
    const slopes = features
      .transpose() // transpose Features so can make matrix multiplication
      .matMul(differences) // Features * diffs
      .div(features.shape[0]); // divide by number of observations n

    //update weights (tensor of "b" and "mn") as: multiply slopes by learning rate and substract result from weights
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  /* train model with training data */
  train() {
    /* train using all data */
    // for (let i = 0; i < this.options.iterations; i++) {
    //   this.gradientDescent(this.features, this.labels);
    //   this.recordMSE();
    //   this.updateLearningRate();
    // }

    /* train using batch (portion of data) / stochastic (one data) approach for speed improvement */
    //calculate number of times we will loop through dataset: numberOfBatches = totaRowsOfData / batchSize
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;

        //extract data with tf slice([startRowIndex, startColIndex], [sizeRow, sizeCol])
        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1] //-1 means all colimns
        );
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        this.gradientDescent(featureSlice, labelSlice);
      }

      this.recordMSE();
      this.updateLearningRate();
    }
  }

  /* calculate Coefficient of Determination that show how good a fit is (use test data to make predictions about observations with known labels) */
  test(testFeatures, testLabels) {
    //create a tensor of features, append column of "1" and standardize data
    testFeatures = this.processFeatures(testFeatures);
    //create a tensor of labels
    testLabels = tf.tensor(testLabels);

    //make predicitions as y^ = (Features * Weights)
    const predictions = testFeatures.matMul(this.weights);

    //sum of square of residuals = Sum(actual - predicted)^2
    const SSres = testLabels.sub(predictions).pow(2).sum().arraySync(); //call arraySync() at the end to turn tensor to array of values and get the single value

    //total sum of squares = Sum(actual - average)^2
    const SStot = testLabels.sub(testLabels.mean()).pow(2).sum().arraySync();

    //coefficient of determination R^2 = 1 - SSres/SStot
    return 1 - SSres / SStot;
  }

  /* calculate MSE = 1/n * sum(((Features * Weights) - Labels)^2) */
  recordMSE() {
    const mse = this.features
      .matMul(this.weights) // estimates = Features * Weights
      .sub(this.labels) // differenses = estimates - actuals -> (Features * Weights) - Labels
      .pow(2) // differenses^2
      .sum() // sum
      .div(this.features.shape[0]) //divide by number of observations
      .arraySync();

    //save it
    this.mseHistory.unshift(mse);
  }

  /* learning rate optimizer */
  updateLearningRate() {
    //no enough records to compare
    if (this.mseHistory.length < 2) {
      return;
    }

    //compare last two records in array (we unshift in it)
    if (this.mseHistory[0] > this.mseHistory[1]) {
      //if MSE went "up" (we did a bad update) -> divide LR by 2
      this.options.learningRate /= 2;
    } else {
      //if MSE went "down" (we going in right direction) -> increse LR by 5%
      this.options.learningRate *= 1.05;
    }

    //console.log(this.options.learningRate);
  }

  /* make a prediction about y = b + m1x1 + m2x2 + ... + mnxn */
  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights);
  }
}

module.exports = LinearRegression;
