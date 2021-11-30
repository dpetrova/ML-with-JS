/*
jumpData -> jump results from 4 players and each player has 3 tries
playerData -> player number and its height
*/

const tf = require("@tensorflow/tfjs-node");

const jumpData = tf.tensor([
  [70, 50, 56],
  [80, 70, 65],
  [50, 67, 75],
  [70, 70, 79],
]);

const playerData = tf.tensor([
  [1, 160],
  [2, 165],
  [3, 182],
  [4, 179],
]);

const testData = jumpData.sum(1, true).concat(playerData, 1);
testData.print();
