# 01 Plinko

k-nearest neighbors algorithm (KNN) for plinco game:

- basic concepts
- array of arrays data structures
- training and test sets
- data normalization
- feature selection

# 02 Tensor manipulations

Basic tensor manipulations:

- dimensions and shape
- logging tensor data
- add/sub/mul/div
- extract and slice
- concatenation
- sum value along axis

Process jump data of players

Predict house price depends on its location (latitude, longitude) using KNN algorithm with Tensorflow

# 03 KNN Tensorflow

KNN algorithm using Tensorflow of predict houses prices depends on their location (latitude, longitude), square foot living, square foot lot

1. Find distance between features data and prediction point
2. Sort from lowest distance to greatest
3. Take the top k records
4. Average the label value (price) of those top k records

# 04 Regression

### Linear Regression using Gradient descent algorithm with TensorFlow

y = b + x1m1 + x2m2 + ... + xnmn

1. Pick starting values of coefficients "b" and "m"
2. Calculate the slope of MSE with respect to "b" and "m's"
3. Multiply the slope by learning rate
4. Update coefficients "b" and "m's"

### Logistic Regression for Binary Classification

Use Marginal Probability Distribution -> consider one possible output case in isolation

sigmoid equation:
y = 1 / (1 + e^-(m.x + b))

1. Encode label values as either "0" or "1"
2. Pick starting values of coefficients "b" and "m's"
3. Calculate the slope of Cross Entropy with respect to "b" and "m's"
4. Multiply the slope by learning rate
5. Update coefficients "b" and "m's"

### Logistic Regression for Multinominal Classification

Use Conditional Probability Distribution -> consider all possible output cases together

softmax equation:
y = e^(m*x + b) / Sum(e^(m*x + bk))

1. Create a column for every possible label value and encode label value with "1" if label value match column value, otherwise encode with "0"
2. Pick starting values of coefficients "b's" and "m's"
3. Calculate the slope of Cross Entropy with respect to "b" and "m's"
4. Multiply the slope by learning rate
5. Update coefficients "b's" and "m's"
