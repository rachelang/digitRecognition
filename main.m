% setup
clear ; close all; clc
addpath('./lib'); addpath('./data');

% neural network specifications
num_hidden_layers = 1;  % will be varied to get best result
input_layer_size = 784;
hidden_layer_size = 30;
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

% training network using fmincg rather than the native fminunc, as this 
% optimization algorithm uses much less memory and makes it possible 
% to be run on older machines (like mine)
max_iters = 100;

% vary lambda to find best result
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10];  
[ThetaRolled, lambda_train_cost, lambda_cv_cost, bestLambda] = varyLambda(lambda_vec, ...
                                              max_iters, nn_specs, ...
                                              X_train, y_train, ...
                                              X_cv, y_cv);

Theta = reshapeParams(ThetaRolled, bestNumLayers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels);                                          
y_pred = predict(Theta, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(y_pred == y_test)) * 100);
[test_cost, ~] = costFunction(ThetaRolled, bestNumLayers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_test, y_test, bestLambda);
fprintf('\nTest Set error: %f\n', test_cost);
                                              
% save trained theta
save theta.mat Theta;

% plot errors as function of lambda                                             
figure;
plot(lambda_vec, lambda_train_cost, lambda_vec, lambda_cv_cost);
title('Error as function of lambda')
legend('Train', 'Cross Validation')
xlabel('Lambda')
ylabel('Error')
                                              
