/* Given the horsepower, weight and engine displacement of a vehicle, will it have HIGH, MEDIUM, or LOW fuel_effivciency */

require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("../load-csv");
const LogisticRegression = require("./logistic-regression");
const plot = require("node-remote-plot");
const _ = require("lodash");

/*
Classifying continious values:
fuel_effivciency ~ miles_per_gallon
LOW -> 0-15 MPG, MEDIUM -> 15-30 MPG, HIGH -> 30+ MPG

MPG       Label         Low  Medium  High
 9         low           1     0      0
 12        low           1     0      0
 16  -->   medium  -->   0     1      0
 27        medium        0     1      0
 32        high          0     0      1
 45        high          0     0      1
*/

const { features, labels, testFeatures, testLabels } = loadCSV("../cars.csv", {
  dataColumns: ["horsepower", "displacement", "weight"], //extract features columns that we will use
  labelColumns: ["mpg"], //extract labels that we will use
  shuffle: true, //shuffle csv data to avoid taking test/training data sets of specific area which will bias the result
  splitTest: 50, //split data into test set (50) and training set (the rest)
  converters: {
    //convert continious data into discrete classifying data
    mpg: (value) => {
      const mpg = parseFloat(value);
      if (mpg < 15) return [1, 0, 0];
      else if (mpg < 30) return [0, 1, 0];
      else return [0, 0, 1];
    },
  },
});

//_.flatMap remove one level of nesting of arrays
const regression = new LogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
});

//regression.weights.print();

//train
regression.train();

//test
const accuracy = regression.test(testFeatures, _.flatMap(testLabels));
console.log(accuracy);

//plot calculated CrossEntropy against the iterations
plot({
  x: regression.crossEntropyHistory.reverse(),
  xLabel: "Iteration #",
  yLabel: "Cross Entropy",
});
