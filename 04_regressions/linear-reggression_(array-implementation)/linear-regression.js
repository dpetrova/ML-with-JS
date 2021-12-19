const _ = require("lodash");
const math = require("mathjs");

class LinearRegression {
  constructor(features, labels, options) {
    //estimate min, max, mean and standard deviation
    this.min = math.min(features);
    this.max = math.max(features);
    this.mean = math.mean(features);
    this.stDev = math.std(features);

    //initialize features and labels as normalize/standardize features data
    this.features = this.standardize(features); //x
    this.labels = labels; //y

    // override default learning_rate and max_number_of_iterations (times that run gradientDescent algorithm)
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    //initialize m and b
    this.m = 0;
    this.b = 0;
  }

  standardize(data) {
    return data.map((x) => [(x[0] - this.mean) / this.stDev]);
  }

  normalize(data) {
    return data.map((x) => [(x[0] - this.min) / (this.max - this.min)]);
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }

  gradientDescent() {
    //calculate estimates of dependent variable y = mx + b
    const labelsEstimates = this.features.map(
      (row) => this.m * row[0] + this.b // x (features) are array of arrays
    );

    //calculate the slope d(MSE)/db = 2/n * Sum(estimate - actual)
    const bSlope =
      _.sum(
        labelsEstimates.map(
          (estimate, i) => estimate - this.labels[i][0] // y (labels) are array of arrays
        )
      ) *
      (2 / this.labels.length);

    //calculate the slope d(MSE)/dm = 2/n * Sum(-x *(actual -estimate))
    const mSlope =
      _.sum(
        labelsEstimates.map(
          (estimate, i) =>
            -1 * this.features[i][0] * (this.labels[i][0] - estimate)
        )
      ) *
      (2 / this.labels.length);

    //update "m" and "b" as multiply both slopes by learning rate and substract result from both 'b' and 'm'
    this.b = this.b - bSlope * this.options.learningRate;
    this.m = this.m - mSlope * this.options.learningRate;
  }
}

module.exports = LinearRegression;
