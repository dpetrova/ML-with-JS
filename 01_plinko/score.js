/* k-NEAREST NEIGHBOUR ALGORITHM (KNN) with n independent varaibles */

//outputs is array of arrays, e.g [[552, 0.52, 16, 8], [390, 0.51, 16, 7],...]
let outputs = [];

// Ran every time a balls drops into a bucket
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  //outputs.push([dropPosition, bounciness, size, bucketLabel]);
  outputs.push([...arguments]);
}

/* split shuffled data into "training" and "test" sets */
function splitDataset(data, testSetCount) {
  //shuffle initial dataset to randomize records data
  const shuffled = _.shuffle(data);

  const testSet = _.slice(shuffled, 0, testSetCount); // from 0 to testCount
  const trainingSet = _.slice(shuffled, testSetCount); //from testCount to the end

  return [testSet, trainingSet];
}

/*
Data normalization: 
- All variables must be normalized before run knn, because they are of very different magnitute of values (position: 0-600, bounciness: 0-1, ballsize: 1-30)
- With no normalization some variables will have great significance (e.g drop position: (10-300)^2=84100), while others will have pretty no significance (e.g. bounciness: (0.5-0.55)^2=0.0025)
- We will use min-max-formula: (val - min) / (max - min)
*/
function minMax(data) {
  //count number of variables in array that need to normalize
  //our data is array of arrays with pointdata
  //e.g we have 4 items in datapoint, and need to normalize first 3 of them (position, bounciness and ballsize), last is bucket
  const featuresCount = data[0].length - 1;

  //create clone of outputs array, so not to mutate the original data
  const clonedData = _.cloneDeep(data);

  for (let featureIndex = 0; featureIndex < featuresCount; featureIndex++) {
    //extract each individual feature values as array of numbers
    const column = clonedData.map((point) => point[featureIndex]);

    //get min and max values of each feature
    const min = _.min(column);
    const max = _.max(column);

    //modify the cloned data with normalized values
    for (let point = 0; point < clonedData.length; point++) {
      clonedData[point][featureIndex] =
        (clonedData[point][featureIndex] - min) / (max - min);
    }
  }

  return clonedData;
}

/* distance in 1-dimension (line) */
// function distance(pointA, pointB) {
//   return Math.abs(pointA - pointB);
// }

/* distance in n-dimensional plane (n independent variables): pythagorean theorem D = ((x1 -x2)^2 + (y1-y2)^2 + (z1-z2)^2 + ...)^0.5 */
function distance(pointA, pointB) {
  //pointA = [x1,y1,z1,...], pointB = [x2,y2,z2,...]
  return (
    _.chain(pointA)
      .zip(pointB) // from [x1, y1, z1,...] and [x2, y2, z2,...] -> [[x1, x2], [y1, y2], [z1,z2],...]
      .map(([a, b]) => (a - b) ** 2) //using destructuring to map arrays in array and for each pair of elements substract them and square the result
      .sum() //sum all elements in array
      .value() ** 0.5 //take the square root
  );
}

// Implement k-nearest neighbors algorithm (KNN) with one independent variable */
// 1. For each observation get distance as substract drop point from prediction point and take abs value
// 2. Sort the results from least to greatest distance
// 3. Look at the top k top records: What is the most common buckets?
// function knn(dataset, predictionPoint, k) {
//   /*
//   example of chain array manipulations with k=3 and predictionPoint=300:
//      bunch of data      get distance from      sort by       get first k   object with bucket   turn object into   sort by bucket     get last     get first   parse to
//                         prediction and bucket  distance        elements       appearances        array of arrays     appearance        element      element    integer
//   [
//     [10, 0.5, 16, 1],         [290, 1]         [50, 4]         [50, 4]         {                 [                  [
//     [200, 0.5, 16, 4],  --->  [100, 4]  --->   [100, 4]  --->  [100, 4]  --->    "4": 2,  --->     ["4", 2],  --->    ["1", 1],  --->  ["4", 2]  ---> "4"  --->  4
//     [350, 0.5, 16, 4],        [50, 4]          [290, 1]        [290, 1]          "1": 1            ["1", 1]           ["4", 2]
//     [600, 0.5, 16, 5]         [300, 5]         [300, 5]                        }                 ]                  ]
//   ]
// */
//   return _.chain(dataset)
//     .map((row) => [distance(row[0], predictionPoint), row[3]]) //return array with first element distance from predictionPoint, and second - number of bucket
//     .sortBy((row) => row[0]) //sort by distance
//     .slice(0, k) //get first k records
//     .countBy((row) => row[1]) //count buckets ({number-of-bucket: number-of-times-it-is-occured})
//     .toPairs() //turn object into array of arrays [[number-of-bucket, number-of-times-it-is-occured]]
//     .sortBy((row) => row[1]) //sort by bucket occurences
//     .last() //get the last element of array (with most bucket ocurrences)
//     .first() //get first element which is number of bucket
//     .parseInt() //convert string key as number
//     .value(); //stop the chain and eventually return the value of all chained computations
// }

/* Implement k-nearest neighbors algorithm (KNN) with n independent variables */
//1. For each observation get distance from prediction to actual point
//2. Sort the results from least to greatest distance
//3. Look at the top k top records: What is the most common buckets? */
function knn(dataset, predictionPoint, k) {
  return _.chain(dataset)
    .map((point) => {
      return [
        distance(_.initial(point), _.initial(predictionPoint)), //distance from actual to prediction point (_.initial(point) will get all elements in array except last, i.e variables)
        _.last(point), //_.last(point) will get last element in array, i.e bucket
      ];
    })
    .sortBy((point) => point[0]) //sort by distance
    .slice(0, k) //get first k records
    .countBy((point) => point[1]) //count buckets ({number-of-bucket: number-of-times-it-is-occured})
    .toPairs() //turn object into array of arrays [[number-of-bucket, number-of-times-it-is-occured]]
    .sortBy((point) => point[1]) //sort by bucket occurences
    .last() //get the last element of array (with most bucket ocurrences)
    .first() //get first element which is number of bucket
    .parseInt() //convert string key as number
    .value(); //stop the chain and eventually return the value of all chained computations
}

function runAnalysisByDifferentK() {
  //split data into "training" and "test" sets
  const testSetSize = 50;
  const normalizedData = minMax(outputs);
  const [testSet, trainingSet] = splitDataset(normalizedData, testSetSize);

  /*
  Adjust k:
  1. Record a bunch of data point
  2. Split that data into a "training" set and a "test" set
  3. Foreach "test" record run KNN using the "training" data
  4. Does the result of KNN equal the "test" record bucket?
  */
  _.range(1, 20).forEach((k) => {
    //run KNN with "training" set for each record in "test" set
    // let numberCorrect = 0;
    // for (const record of testSet) {
    //   const bucket = knn(trainingSet, record[0], k);
    //   if (bucket === record[3]) numberCorrect++;
    // }
    // console.log(`For k of ${k} accuracy is : ${numberCorrect / testSetSize}`);

    const accuracy = _.chain(testSet)
      .filter(
        (testPoint) => knn(trainingSet, testPoint, k) === _.last(testPoint)
      ) //filter items with predicted bucket === actual bucket
      .size() //get size of array of corrects
      .divide(testSetSize) //calculate sizeOfCorrectsArray/testSetSize
      .value(); //terminate chain

    console.log("For k of ", k, "accuracy is: ", accuracy);
  });
}

function runAnalysisByIndividualFeature() {
  const k = 10;
  const testSetSize = 50;
  const featuresCount = outputs[0].length - 1; //get number of independent variables

  _.range(0, featuresCount).forEach((feature) => {
    //get only desired feature and bucket result
    const data = _.map(outputs, (point) => [point[feature], _.last(point)]);
    //normalize data
    const normalizedData = minMax(data);
    //split data into "training" and "test" sets
    const [testSet, trainingSet] = splitDataset(normalizedData, testSetSize);
    const accuracy = _.chain(testSet)
      .filter(
        (testPoint) => knn(trainingSet, testPoint, k) === _.last(testPoint)
      ) //filter items with predicted bucket === actual bucket
      .size() //get size of array of corrects
      .divide(testSetSize) //calculate sizeOfCorrectsArray/testSetSize
      .value(); //terminate chain

    const features = ["Drop position", "Bounciness", "Ball size"];
    console.log(`For feature "${features[feature]}" accuracy is: ${accuracy}`);
  });
}
