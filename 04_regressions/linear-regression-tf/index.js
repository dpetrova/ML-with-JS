require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("../load-csv");
const LinearRegression = require("./linear-regression");
const plot = require("node-remote-plot");

//load CSV file and use destructuring to extract features, labels, testFeatures, testLabels
let { features, labels, testFeatures, testLabels } = loadCSV("../cars.csv", {
  shuffle: true, //shuffle csv data to avoid taking test/training data sets of specific area which will bias the result
  splitTest: 50, //split data into test set (50) and training set (the rest)
  dataColumns: ["horsepower", "weight", "displacement"], //extract features columns that we will use
  labelColumns: ["mpg"], //extract labels that we will use
});

//initialize regression class
const regression = new LinearRegression(features, labels, {
  //you must play with different learningRates/iterations to receive the best fit (Coefficient of Determination R^2 close to 1 as much as possible)!!!!!
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10, //use batches to improve speed and performance (batch -> portion of data; stochastic -> one data)
});

//train
regression.train();

//calculate coefficient of determination
const r2 = regression.test(testFeatures, testLabels);
console.log("Coefficient of Determination R^2 is: ", r2);

//plot calculated MSE against the iterations
plot({
  x: regression.mseHistory.reverse(),
  xLabel: "Iteration #",
  yLabel: "Mean Squared Error (MSE)",
});

//make a prediction of y(MPG) depends on horsepower=120, weight=2 and displacement=380
regression.predict([[120, 2, 380]]).print();
