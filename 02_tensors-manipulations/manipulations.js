const tf = require("@tensorflow/tfjs-node");

//one-dimensional
const tensorA = tf.tensor([10, 8, 9]);

//two-dimensional
const tensorB = tf.tensor([
  [4, 3, 2],
  [5, 7, 1],
]);

const tensorC = tf.tensor([
  [5, 4, 1],
  [2, 1, 3],
]);

const tensorD = tf.tensor([
  [9, 9],
  [5, 11],
]);

//three-dimensional
const tensorE = tf.tensor([
  [
    [2, 11, 6],
    [1, 1, 8],
  ],
]);

/* SHAPE AND DIMENSIONS */
console.log("shape of tensorA: ", tensorA.shape); // [ 3 ]
console.log("shape of tensorB: ", tensorB.shape); // [ 2, 3 ]
console.log("shape of tensorC: ", tensorC.shape); // [ 2, 3 ]
console.log("shape of tensorD: ", tensorD.shape); // [ 2, 2 ]
console.log("shape of tensorE: ", tensorE.shape); // [ 1, 2, 3 ]

/* LOGGING TENSOR DATA */
console.log("tensorA");
tensorA.print();
console.log("tensorB");
tensorB.print();
console.log("tensorC");
tensorC.print();
console.log("tensorD");
tensorD.print();
console.log("tensorE");
tensorE.print();

/* OPERATION WITH TENSORS */
//addition
const add = tensorA.add(tensorB);
console.log("tensorA + tensorB");
add.print();

//substraction
const sub = tensorB.sub(tensorC);
console.log("tensorB - tensorC");
sub.print();
// const sub2 = tensorC.sub(tensorD);
// sub2.print(); //Incompatible shapes: [2,3] vs. [2,2]

//multiplication
const mul = tensorB.mul(tensorC);
console.log("tensorB * tensorC");
mul.print();

//division
const div = tensorB.div(tensorC);
console.log("tensorB / tensorC");
div.print();

/* EXTRACT VALUES FROM A TENSOR */
//returns the multi dimensional array of values.
//tensorB.array().then((array) => console.log(array));
console.log("tensorB as multi dimensional array of values");
console.log(tensorB.arraySync());

// returns the flattened data that backs the tensor.
//tensorB.data().then((data) => console.log(data));
console.log("tensorB as flattened data that backs the tensor");
console.log(tensorB.dataSync());

//extract data ([startRow, startCol], [sizeRow, sizeCol])
console.log(
  "extract values as start from row=0,col=1 and get 2 rows and 1 column"
);
const slice = tensorB.slice([0, 1], [2, 1]);
slice.print();

console.log(
  "extract values as start from row=0,col=1 and get all rows and all columns"
);
const slice2 = tensorB.slice([0, 1], [-1, -1]);
slice2.print();

/* TENSOR CONCARENATION */
console.log("concat tensorB and tensorC by row");
const concatByRow = tensorB.concat(tensorC, 0);
concatByRow.print();

console.log("concat tensorB and tensorC by column");
const concatByCol = tensorB.concat(tensorC, 1);
concatByCol.print();

/* SUM VALUE ALONG AXIS */
console.log("tensorB: sum of each column");
const sumByRow = tensorB.sum(0);
sumByRow.print();

console.log("tensorB: sum of each column and keep dimensions");
const sumByRowkeepDim = tensorB.sum(0, true);
sumByRowkeepDim.print();

console.log("tensorB: sum of each row");
const sumByCol = tensorB.sum(1);
sumByCol.print();

console.log("tensorB: sum of each row and keep dimensions");
const sumByColKeepDim = tensorB.sum(1, true);
sumByColKeepDim.print();

console.log("tensorB: another way to sum of each row and keep dimensions");
const sumByColKeepDim2 = tensorB.sum(1).expandDims(1);
sumByColKeepDim2.print();
