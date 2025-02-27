# Digit Recognition Neural Network

Machine learning recognition of handwritten digits 0 to 9 via a three-layer neural network of 784x30x10 units. The network utilized the MNIST dataset of 28x28 pixel handwritten digit data which was split into training, cross validation, and test subsets of 50,000, 10,000, and 10,000 examples each.

The test data was fed into input neurons, and the network was trained using an advanced optimization algorithm via forward and backpropagation. Regularization was also used, and the optimal lambda was chosen using the lowest cross validation error on a range of lambda values.

## Performance and Results
Test Set Accuracy: **95.6%**

Test Set error: **0.312623**

The trained theta with the above results is saved [here](trainedTheta/3layer30units.mat)

Optimal lambda: 0.030

![alt text](https://github.com/rachelang/digitRecognition/blob/master/graph/lambdaVsCost.PNG "lambdaVsCost")

## Reflection
I originally attempted to deploy a multi-hidden-layer neural network, but my test runs suggested that it either had too much bias or variance, and a one hidden layer network was perfect for the problem. Based on the lambda vs. cost plot and optimal lambda resulting from a 30 unit hidden layer, the accuracy can most likely be increased as the optimal lambda is rather low, suggesting that the network is not very biased and could do better with more units in its hidden layer.

The code resulting from my experiments with multi-hidden-layer neural networks can be found under my [neuralNetworkTemplate](https://github.com/rachelang/neuralNetworkTemplate) repo.
