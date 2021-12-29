/* Given the pixel intensity values in an image, identify whether the character is a handwritten 0,1,2,3,4,5,6,7,8,9 */

require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./logistic-regression");
const plot = require("node-remote-plot");
const _ = require("lodash");
const mnist = require("mnist-data"); //utilities for working with MNIST database that contain tons of different images of handwritten digits

/*
FEATURES:
Every image is collection of pixels (grid of grayscale pixel values)

Grayscale values: 0 -> 50 -> 100 -> 150 -> 200 -> 255
                 white         gray              black

Need to represent each picture instead of a grid of pixel values rather as a single array of pixel values:
0 0 ... 0  0   0  ... 0
0 0 ... 0  0   0  ... 0
0 0 ... 89 115 80 ... 0  ---> 0 0 0 0 ... 89 115 80 ... 0 0 0
0 0 ... 90 190 99 ... 0
.......................
0 0 ... 0  0   0  ... 0

*/

/*
LABELS:
Need to create a column for every possible label value (digits 0 - 9) and fill it with 0 or 1 depends on the label match column value
  Label            Encoded labels for
  value            0 1 2 3 4 5 6 7 8 9
    7       --->   0 0 0 0 0 0 0 1 0 0
    4       --->   0 0 0 0 1 0 0 0 0 0
    0       --->   1 0 0 0 0 0 0 0 0 0

*/

function loadTrainingData() {
  // const singleImage = mnist.training(0, 1);
  // console.log(singleImage)

  //get a slice of training data set as every image is a grid of pixel values (array of arrays)
  const trainingMnistData = mnist.training(0, 60000);
  //const trainingMnistData = mnist.training(0, 5000);

  //flat array of arrays into a single array
  const features = trainingMnistData.images.values.map((image) =>
    _.flatMap(image)
  ); //_.flatMap remove one level of array nesting

  //for every label value create a set of arrays for every possible digits (0-9) and encode matching index with 1 and 0 the others
  const encodedLabels = trainingMnistData.labels.values.map((label) => {
    const row = new Array(10).fill(0); // array of 0's for every possible digits (0-9)
    row[label] = 1; //set matching index to 1
    return row;
  });

  return { features, labels: encodedLabels };
}

function loadTestingData() {
  //get a slice of test data set
  const testMnistData = mnist.testing(0, 10000);
  //const testMnistData = mnist.testing(0, 100);

  const testFeatures = testMnistData.images.values.map((image) =>
    _.flatMap(image)
  );

  const encodedTestLabels = testMnistData.labels.values.map((label) => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
  });

  return { testFeatures, encodedTestLabels };
}

//get destructured training features and labels
const { features, labels } = loadTrainingData();

//create regression instance
const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 40,
  batchSize: 500,
});

//train
regression.train();

//debugger

//get testing features and labels
const { testFeatures, encodedTestLabels } = loadTestingData();

//test accuracy of model
const accuracy = regression.test(testFeatures, encodedTestLabels);
console.log("Accuracy is", accuracy);

//plot calculated CrossEntropy against the iterations
plot({
  x: regression.crossEntropyHistory.reverse(),
  xLabel: "Iteration #",
  yLabel: "Cross Entropy",
});
