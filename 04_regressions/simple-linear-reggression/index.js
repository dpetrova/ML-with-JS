/* Use Linear Regression to find the relation between one independent variable (HORSEPOWER) and one dependent variable (MILES_PER_GALLON)
MPG = m * HP + b
*/

const loadCSV = require("../load-csv");
const LinearRegression = require("./linear-regression");

//load CSV file and use destructuring to extract features, labels, testFeatures, testLabels
let { features, labels, testFeatures, testLabels } = loadCSV("../cars.csv", {
  shuffle: true, //shuffle csv data to avoid taking test/training data sets of specific area which will bias the result
  splitTest: 50, //split data into test set (50) and training set (the rest)
  dataColumns: ["horsepower"], //extract features that we will use
  labelColumns: ["mpg"], //extract labels that we will use
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.01,
  iterations: 1000,
});

regression.train();

console.log(`Updated m: ${regression.m}, updated b: ${regression.b}`);
