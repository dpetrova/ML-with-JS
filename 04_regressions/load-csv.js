const fs = require("fs");
const _ = require("lodash");
const shuffleSeed = require("shuffle-seed");

//extract colums that we will use
function extractColumns(data, columnNames) {
  const headers = _.first(data);
  const indexes = _.map(columnNames, (column) => headers.indexOf(column));
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
  //read csv data
  let data = fs.readFileSync(filename, { encoding: "utf-8" });
  data = _.map(data.split("\n"), (d) => d.split(","));
  data = _.dropRightWhile(data, (val) => _.isEqual(val, [""]));
  //get headers
  const headers = _.first(data);

  //convert and parse data except the header
  data = _.map(data, (row, index) => {
    if (index === 0) {
      return row;
    }
    return _.map(row, (element, index) => {
      if (converters[headers[index]]) {
        const converted = converters[headers[index]](element);
        return _.isNaN(converted) ? element : converted;
      }

      const result = parseFloat(element.replace('"', ""));
      return _.isNaN(result) ? element : result;
    });
  });

  //extract colums that we will use
  let labels = extractColumns(data, labelColumns);
  data = extractColumns(data, dataColumns);
  //remove headers data
  data.shift();
  labels.shift();

  //shuffle data
  if (shuffle) {
    data = shuffleSeed.shuffle(data, "phrase");
    labels = shuffleSeed.shuffle(labels, "phrase");
  }

  //split data into test and training sets
  if (splitTest) {
    const trainSize = _.isNumber(splitTest)
      ? splitTest
      : Math.floor(data.length / 2);

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
