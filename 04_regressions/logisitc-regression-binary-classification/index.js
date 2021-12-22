/* Given the horsepower, weight and engine displacement of a vehicle, will it PASS or NOT PASS a smog emmissions check? */

require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("../load-csv");
const LogisticRegression = require("./logistic-regression");
const plot = require("node-remote-plot");

const { features, labels, testFeatures, testLabels } = loadCSV("../cars.csv", {
  dataColumns: ["horsepower", "displacement", "weight"], //extract features columns that we will use
  labelColumns: ["passedemissions"], //extract labels that we will use
  shuffle: true, //shuffle csv data to avoid taking test/training data sets of specific area which will bias the result
  splitTest: 50, //split data into test set (50) and training set (the rest)
  converters: {
    //encode discrete values into 0 or 1
    passedemissions: (value) => {
      return value === "TRUE" ? 1 : 0;
    },
  },
});

//initialize regression class
const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.6, //probability boundary below/above which label is accepted to match encoded value of 0/1
});

//train
regression.train();

//test
const correctness = regression.test(testFeatures, testLabels);
console.log(
  "Percentage that we got correct out of all our guesses is: ",
  correctness
);

//plot calculated CrossEntropy against the iterations
plot({
  x: regression.crossEntropyHistory.reverse(),
  xLabel: "Iteration #",
  yLabel: "Cross Entropy",
});

//make a prediction of y(passedemissions) depends on horsepower=130, displacement=307 and weight=1.75
console.log(
  "prediction of passedemissions (true=1/false=0) for horsepower=130, displacement=307 and weight=1.75 is: ",
  regression.predict([[130, 307, 1.75]]).arraySync()[0][0]
);
