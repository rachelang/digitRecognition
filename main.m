% setup
clear ; close all; clc
addpath('./lib'); addpath('./data');

% neural network specifications
num_hidden_layers = 1;  % will be varied to get best result
input_layer_size = 784;
hidden_layer_size = 15;
num_labels = 10;

nn_specs = [num_hidden_layers, input_layer_size, hidden_layer_size, num_labels];

% load training and testing data
X_all = [loadMNISTImages('train-images.idx3-ubyte')'; loadMNISTImages('t10k-images.idx3-ubyte')'];
y_all = [loadMNISTLabels('train-labels.idx1-ubyte'); loadMNISTLabels('t10k-labels.idx1-ubyte')];

% partition data
X_train = X_all(1:50000, :);
y_train = y_all(1:50000, :);
X_cv = X_all(50001:60000, :);
y_cv = y_all(50001:60000, :);
X_test = X_all(60001:70000, :);
y_test = y_all(60001:70000, :);

% constant(s)
m_train = size(X_train, 1);

% visualize data
rand_indices = randperm(m_train);
sel = X_train(rand_indices(1:100), :);
displayData(sel);


% Neural Network Training
% -----------------------

% randomly initialize weights
initial_params = randInitWeights(num_hidden_layers, input_layer_size, hidden_layer_size, num_labels);

% training network using fmincg rather than the native fminunc, as this 
% optimization algorithm uses much less memory and makes it possible 
% to be run on older machines (like mine)
max_hidden_layers = 20;

[Theta, costResults, bestNumLayers] = varyNumHiddenLayers(max_hidden_layers, ...
                                              nn_specs, ...
                                              initial_params, ...
                                              X_train, y_train, ...
                                              X_cv, y_cv);
