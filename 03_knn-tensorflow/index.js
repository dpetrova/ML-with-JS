require("@tensorflow/tfjs-node"); //instruct tf to use CPU to do calculations (tf can either use CPU or GPU to run calculations)
//require("@tensorflow/tfjs-node-gpu");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

//load CSV file and use destructuring to extract features, labels, testFeatures, testLabels
let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true, //shuffle csv data to avoid taking test/training data sets of specific area which will bias the result
    splitTest: 10, //split data into test set (10) and training set
    dataColumns: ["lat", "long", "sqft_lot", "sqft_living"], //extract features columns that we will include in knn (most important features that impact the house price)
    labelColumns: ["price"], //extract labels columns that we will use
  }
);

//console.log(testFeatures);
//console.log(testLabels);

function knn(features, labels, predictionPoint, k) {
  //use tf moments() method to calculate average and variance for each feature
  const { mean, variance } = tf.moments(features, 0);

  //standardize prediction point as: (val - average) / stDev
  const standardizedPrediction = predictionPoint
    .sub(mean) // substact average from value
    .div(variance.pow(0.5)); // divide difference by standard deviation (stdev is square root of variance)

  return (
    features
      // features data standartization
      .sub(mean)
      .div(variance.pow(0.5))
      // calc distance between feature point and prediction point
      .sub(standardizedPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      // concat labels
      .concat(labels, 1)
      // sort from lowest point to greatest
      .unstack()
      .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1))
      // take the top k records
      .slice(0, k)
      // average the label value of those top k records
      .reduce((acc, tensor) => acc + tensor.arraySync()[1], 0) / k
  );
}

//convert arrays of data to tensors
features = tf.tensor(features);
labels = tf.tensor(labels);

//loop through the testSet and for each testPoint calculate prediction and error
testFeatures.forEach((testPoint, i) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10);
  const err = (testLabels[i][0] - result) / testLabels[i][0];
  console.log(
    `Actual: ${testLabels[i][0]}$ Prediction: ${result}$ Error: ${Math.floor(
      err * 100
    )}%`
  );
});
