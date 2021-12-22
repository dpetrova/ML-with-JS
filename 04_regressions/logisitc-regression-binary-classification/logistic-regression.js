/* 
BINARY CLASSIFICATION
Logistic Regression to find the sigmoid equation (probability of being "1" label)
y = 1 / (1 + e^-(m*x + b))
and predict discrete binary values
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
      { learningRate: 0.1, iterations: 1000, decisionBoundary: 0.5 }, //decisionBoundary is boundary of probability below/above which label is accepted 0/1
      options
    );

    //record Cross Entropy to can adjust learning rate
    this.crossEntropyHistory = [];

    //initialize b and m as "0" ([b, m1, m2, ..., mn])
    this.weights = tf.zeros([this.features.shape[1], 1]);
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
    //standartize must be done with same mean and variance so cache them as instance variables
    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  /* calculate slope of Cross Entropy (metric of how bad we guessed) with respect to "m" and "b" = (FeaturesT * (sigmoid(Features * Weights) - Labels)) / n */
  gradientDescent(features, labels) {
    const labelsEstimates = features.matMul(this.weights).sigmoid(); // ^y = sigmoid(mx + b)
    const differences = labelsEstimates.sub(labels); // diffs = ^y - y

    //slope = (FeatuesT * diffs) / n
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    //update weights (tensor of "b" and "mn") as: multiply slopes by learning rate and substract result from weights
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  /* train model with training data */
  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;

        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        this.gradientDescent(featureSlice, labelSlice);
      }

      this.recordCrossEntropy();
      this.updateLearningRate();
    }
  }

  /* calculate how many times we got incorrect results 
  Labels    Features                          Predicted probabilities of beign "1"       Apply decision boundary               Actual labels      Differences          Abs differences        Sum of incorrect guesses           Percent of correct guesses
    1       80  200 1.09      Logistic                      0.99                                   1              substract          1                 0        abs          0          sum
    0       57  250 1.9  ---> Regression  --->              0.96                   --->            1              --------->         0         =       1      ------->       1         ----->          2                ----->         (4 -2)/4 = 0.5
    1       150 307 2.1       Model                         0.2                                    0                                 1                -1                     1
    0       180 425 2.2                                     0.01                                   0                                 0                 0                     0
  */
  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures); //predicted labels
    testLabels = tf.tensor(testLabels); //actual labels

    //calculate number of incorrects results during our predicitons
    const incorrect = predictions
      .sub(testLabels) // differences = predicted - actual
      .abs() // abs(differences)
      .sum() // reduce entire tensor to single result: number of incorrects = sum(differences)
      .arraySync(); // get the final sum of incorrects results

    //return the percentage that we got correct out of all of our guesses
    // correctness = (numberOfPredicitions - incorrectPredictions) / numberOfPredicitions
    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  /* make a prediction about y = 1 / (1 + e^-(m*x + b)) */
  predict(observations) {
    // return probability of beign the "1" encoded label
    return (
      this.processFeatures(observations)
        //get Logistic Regression Model
        .matMul(this.weights)
        .sigmoid()
        //apply decision boundary (val > DB -> return 1; val < DB return 0)
        .greater(this.options.decisionBoundary) // returns the truth value of (element > decisionBoundary)
        .cast("float32") //cast boolean truth value to number
    );
  }

  /* 
  calculate Cross Entropy (metric of how bad we guess) 
  CE = -(1/n)(actual^T * log(estimates) + (1-actual)^T * log(1-estimates))  
  */
  recordCrossEntropy() {
    //calculate guesses (estimates of y)
    const estimates = this.features.matMul(this.weights).sigmoid();

    //actual^T * log(estimates)
    const termOne = this.labels.transpose().matMul(estimates.log());

    //(1-actual)^T * log(1-estimates)
    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(estimates.mul(-1).add(1).log());

    //-(1/n)(termOne + termTwo)
    const crossEntropy = termOne
      .add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .arraySync()[0][0];

    //save it in array
    this.crossEntropyHistory.unshift(crossEntropy);
  }

  /* learning rate optimizer */
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
