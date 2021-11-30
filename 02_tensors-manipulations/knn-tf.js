/*
KNN algorithm using Tensorflow of predict house price depends on its location (latitude, longitude)
1. Find distance between features (lat, long) and prediction point
2. Sort from lowest point to greatest
3. Take the top k records
4. Average the label value (price) of those top k records
*/

const tf = require("@tensorflow/tfjs-node");
const k = 2;

//latitude and longitude of houses locations
const features = tf.tensor([
  [-121, 47],
  [-121.2, 46.5],
  [-122, 46.4],
  [-120.9, 46.7],
]);

//houses prices
const labels = tf.tensor([[200], [250], [215], [240]]);

//prediction point
const predictionPoint = tf.tensor([-121, 47]);

/*
distance between two points = ((latA - latB)** 2 + (longA - longB)** 2) ** 0.5
KNN tensor manipulations:
Tensor                            Tensor                          Tensor                          Tensor                  Tensor                      Tensor                    Tensor
[                                 [                               [                               [                       [                           [                         [                             [                                      [                                       
  [-121, 47],                       [0, 0],                         [0, 0],                         0,                       0,                         [0],                      [0, 200],                     Tensor[0, 200],                        Tensor[0, 200],                        [  
  [-121.2, 46.5],   - [-121, 47]    [-0.2, -0.5],     square        [0.04, 0.25],   sum by rows     0.29,    square root     0.54,  expand dimensions   [0.54],  concat prices    [0.54, 250],    unstack       Tensor[0.54, 250],  sort by distance   Tensor[0.32, 240],  get top k records    Tensor[0, 200],    average prices
  [-122, 46.4],     ------------>   [-1, -0.6],    ------------>    [1, 0.36],     ------------>    1.36,   ------------>    1.17,    ------------>     [1.17],  ------------>    [1.17, 215],  ------------>   Tensor[1.17, 215],    ------------>    Tensor[0.54, 250],    ------------>      Tensor[0.32, 240]   ------------>    220  
  [-120.9, 46.7]                    [0.1 , -0.3]                    [0.01, 0.09]                    0.1                      0.32                       [0.32]                    [0.32, 240]                   Tensor[0.32, 240]                      Tensor[1.17, 215]                      ]
]                                 ]                               ]                               ]                       ]                           ]                         ]                             ]                                      ]
*/
const predictionPrice =
  features
    .sub(predictionPoint)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(labels, 1)
    .unstack()
    .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1))
    .slice(0, k)
    .reduce((acc, tensor) => acc + tensor.arraySync()[1], 0) / k;

console.log(
  `The prediction price for house at [${predictionPoint.arraySync()[0]}, ${
    predictionPoint.arraySync()[1]
  }] is ${predictionPrice}$`
);
