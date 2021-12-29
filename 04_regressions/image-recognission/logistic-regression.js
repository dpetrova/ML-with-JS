/* 
MULTINOMINAL CLASSIFICATION
Logistic Regression to find the softmax equation (probability of being "1" label rather than "0" label)
y = e^(m*x + b) / Sum(e^(m*x + bk))
and predict discrete multinominal values
*/

const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class LogisticRegression {
  constructor(features, labels, options) {
    //create a tensor of features data, standardize it and append column of "1"
    this.features = this.processFeatures(features);

    //create a tensor of labels data
    this.labels = tf.tensor(labels);

    // override default learning_rate, max_number_of_iterations and decision_boundary
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000, decisionBoundary: 0.5 }, //decision_boundary is boundary of probability below/above which label is accepted 0/1
      options
    );

    //record Cross Entropy to can adjust learning rate
    this.crossEntropyHistory = [];

    //initialize b and m as "0" ([[b1, m11, m12, ..., m1k],...,[bn, mn1, mn2, ..., mnk]]),
    //where k -> number of observation, n -> number of multinominal label values (columns of tensor of encoded labels)
    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }

  /* calculate slope of Cross Entropy (metric of how bad we guessed) with respect to "m" and "b" = (FeaturesT * (softmax(Features * Weights) - Labels)) / n */
  gradientDescent(features, labels) {
    const labelsEstimates = features.matMul(this.weights).softmax(); // ^y = softmax(mx + b)
    const differences = labelsEstimates.sub(labels); // diffs = ^y - y

    //slope = (FeatuesT * diffs) / n
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    //update weights (tensor of "b" and "mn") as: multiply slopes by learning rate and substract result from weights
    return this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;

        //use tf.tidy() to help avoid memory leaks (minimize memory usage) as clean up all allocated intermediate tensors after execution provided function
        this.weights = tf.tidy(() => {
          const featureSlice = this.features.slice(
            [startIndex, 0],
            [batchSize, -1]
          );
          const labelSlice = this.labels.slice(
            [startIndex, 0],
            [batchSize, -1]
          );

          return this.gradientDescent(featureSlice, labelSlice);
        });
      }

      //debugger

      this.recordCrossEntropy();
      this.updateLearningRate();
    }
  }

  /* calculate how many times we got incorrect results:
    Column of                Column of         Places
    Max Value                Max Value         Not 
    (Predictions)            (Labels)          Equal                
        0                       0                0        Sum    
        0                       0      --->      0        ---->    1 
        1         notEqual      1                0               
        1                       1                0
        2                       0                1
        2                       2                0                    
  */
  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures); //column of max Value of predictions
    testLabels = tf.tensor(testLabels).argMax(1); //column of max Value of labels

    const incorrect = predictions.notEqual(testLabels).sum().arraySync(); //get sum of all not equal places

    //return the percentage that we got correct out of all of our guesses
    // correctness = (numberOfPredicitions - incorrectPredictions) / numberOfPredicitions
    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  /* make a prediction about y = e^(m*x + b) / Sum(e^(m*x + bk)) */
  predict(observations) {
    // return probability of beign the "1" encoded label rather than "0" label
    return (
      this.processFeatures(observations)
        //get Logistic Regression Model
        .matMul(this.weights)
        .softmax()
        //get the actual label value with argMax which return the column index of the largest value inside of each row
        .argMax(1)
    );
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

    //concat a column of "1" to features along a column axis
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  /* standardize data with formula (val - average) / stDev */
  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    /*
    Dealing with zero variances:
    When all features values in a column are "0", then variance will be "0" and because we cannot divide by "0" (results in Infinity), we've got NaN
    So we apply some trick to deal with zero variances:
    1.cast to bool variances in a tensor, so any non-zero variances become 1, and zero variances become 0
    2.cast back to number
    3.flip that values with logicalNot, so 0 become 1, and 1 become 0
    4.add these values to the original variances, so if we have non-zero original variance it remain the same as we will add 0 to it, 
    and if we have zero original variance it will become 1, as we will add 1 to it

       original 
       variances    cast("bool")            logicalNot             add to original value
    [0, 0.04, 0.16]  --------->   [0, 1, 1] --------->  [1, 0, 0]  -------------------->  [1, 0.04, 0.16]
    */
    const filler = variance.cast("bool").logicalNot().cast("float32");
    this.variance = variance.add(filler);

    return features.sub(mean).div(this.variance.pow(0.5));
  }

  recordCrossEntropy() {
    //use tf.tidy() to help avoid memory leaks (minimize memory usage) as clean up all allocated intermediate tensors after execution provided function
    const cost = tf.tidy(() => {
      const guesses = this.features.matMul(this.weights).sigmoid();

      const termOne = this.labels.transpose().matMul(
        guesses
          .add(1e-7) //add very small number to avoid taking logarithm of "0" (result in -Infinity)
          .log()
      );

      const termTwo = this.labels.mul(-1).add(1).transpose().matMul(
        guesses
          .mul(-1)
          .add(1)
          .add(1e-7) //add very small number to avoid taking logarithm of "0" (result in -Infinity)
          .log()
      );

      return termOne
        .add(termTwo)
        .div(this.features.shape[0])
        .mul(-1)
        .arraySync()[0][0];
    });

    this.crossEntropyHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.crossEntropyHistory.length < 2) {
      return;
    }

    if (this.crossEntropyHistory[0] > this.crossEntropyHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
