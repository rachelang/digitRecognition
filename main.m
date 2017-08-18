% setup
clear ; close all; clc
addpath('./lib'); addpath('./data');

% neural network specifications
num_hidden_layers = 3;
input_layer_size = 784;
hidden_layer_size = 25;
num_labels = 10;

% load training and testing data
X_train = loadMNISTImages('train-images.idx3-ubyte')';
y_train = loadMNISTLabels('train-labels.idx1-ubyte');
X_test = loadMNISTImages('t10k-images.idx3-ubyte')';
y_test = loadMNISTLabels('t10k-labels.idx1-ubyte');

% constants
m_train = size(X_train, 1);

rand_indices = randperm(m_train);
%displayData(X_train(1:100,:));

X_train = X_train(1:20000, :);
y_train = y_train(1:20000);
X_test = X_test(1:500, :);
y_test = y_test(1:500);

m_train = size(X_train, 1);
% display data
% rand_indices = randperm(m_train);
% sel = X_train(rand_indices(1:100), :);
%displayData(X_train(rand_indices(1:100),:));
%displayData(X_train(1:100,:));
% displayData(sel);


% Neural Network Training
% -----------------------
fprintf('\nTraining Neural Network... please hold\n');

% set lambda
lambda = 2;

% shorthand for cost function
costFunction = @(p) costFunction(p, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_train, y_train, lambda);
% randomly initialize weights
initial_params = randInitWeights(num_hidden_layers, input_layer_size, hidden_layer_size, num_labels);

% training network using fmincg rather than the native fminunc, as this 
% optimization algorithm uses much less memory and makes it possible 
% to be run on older machines
options = optimset('GradObj', 'on', 'MaxIter', 10);   
[Theta, cost] = fmincg(costFunction, initial_params, options); 


Theta = reshapeParams(Theta, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels);
                              
fprintf("\nDone\n");
y_pred = predict(Theta, X_test);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(y_pred == y_test)) * 100);
