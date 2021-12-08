/*
features -> latitude and longitude of houses locations
labels -> houses prices
distance between two points = ((latA - latB)** 2 + (longA - longB)** 2) ** 0.5
*/

const tf = require("@tensorflow/tfjs-node");

const features = tf.tensor([
  [-121, 47],
  [-121.2, 46.5],
  [-122, 46.4],
  [-120.9, 46.7],
]);

const labels = tf.tensor([[200], [250], [215], [240]]);

const predictionPoint = tf.tensor([-121, 47]);

/*
calculate distance between prediction point and each row data in features
*/
features.sub(predictionPoint).pow(2);
