const fs = require("fs");
const _ = require("lodash");
const shuffleSeed = require("shuffle-seed"); //library to shuffle array of records

//extract colums that we will use
function extractColumns(data, columnNames) {
  //get header row
  const headers = _.first(data);
  //get indexes that want to extract
  const indexes = _.map(columnNames, (column) => headers.indexOf(column));
  //exctract desired column data
  const extracted = _.map(data, (row) => _.pullAt(row, indexes));
  return extracted;
}

module.exports = function loadCSV(
  filename,
  {
    dataColumns = [], //features
    labelColumns = [], //labels
    converters = {}, //convert data?
    shuffle = false, //shuffle data?
    splitTest = false, //split data into test/training sets?
  }
) {
  //read csv data from a file
  let data = fs.readFileSync(filename, { encoding: "utf-8" });
  //split raw data into rows, and then each row into separate data
  data = _.map(data.split("\n"), (row) => row.split(","));
  //remove trailing empty columns if any
  data = _.dropRightWhile(data, (val) => _.isEqual(val, [""]));
  //get headers
  const headers = _.first(data);

  //convert or parse data except the header
  data = _.map(data, (row, index) => {
    //skip header
    if (index === 0) {
      return row;
    }

    //process data
    return _.map(row, (element, index) => {
      //convert data if converter is provided (object with key=name_of_the_column, value=custom_function)
      if (converters[headers[index]]) {
        const converted = converters[headers[index]](element);
        return _.isNaN(converted) ? element : converted;
      }

      //try to parse to number
      const result = parseFloat(element.replace('"', ""));
      return _.isNaN(result) ? element : result;
    });
  });

  //extract colums that we will use
  let labels = extractColumns(data, labelColumns);
  data = extractColumns(data, dataColumns);
  //remove first row, which is headers (column titles)
  data.shift();
  labels.shift();

  //shuffle data
  if (shuffle) {
    data = shuffleSeed.shuffle(data, "phrase"); //use a string as second argument to assure shuffle to be always in the same shuffling order when use identical seed phrase
    labels = shuffleSeed.shuffle(labels, "phrase");
  }

  //split data into test and training sets
  if (splitTest) {
    const trainSize = _.isNumber(splitTest)
      ? splitTest //if splitTest is number
      : Math.floor(data.length / 2); //if splitTest is boolean split data into half

    return {
      features: data.slice(trainSize),
      labels: labels.slice(trainSize),
      testFeatures: data.slice(0, trainSize),
      testLabels: labels.slice(0, trainSize),
    };
  } else {
    return { features: data, labels };
  }
};
