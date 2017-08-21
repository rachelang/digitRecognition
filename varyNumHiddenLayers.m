function [bestParams, train_cost, cv_cost, bestNumLayers] = varyNumHiddenLayers(max_hidden_layers, ...
                                              max_iters, nn_specs, ...
                                              X_train, y_train, ...
                                              X_cv, y_cv)
                            
num_hidden_layers = nn_specs(1);
input_layer_size = nn_specs(2);
hidden_layer_size = nn_specs(3);
num_labels = nn_specs(4);
lambda = 0; % lambda will be varied in varyLambda

train_cost = size(1, max_hidden_layers);
cv_cost = size(1, max_hidden_layers);

for i = 1:max_hidden_layers
  num_hidden_layers = i;
  fprintf('\ntraining nn with %d hidden layer(s)\n', num_hidden_layers);

  % shorthand for cost function
  costFunctionS = @(p) costFunction(p, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_train, y_train, lambda);
                               
  % randomly initialize weights
  initial_params = randInitWeights(num_hidden_layers, input_layer_size, hidden_layer_size, num_labels);

  options = optimset('MaxIter', max_iters);   
  [params, ~] = fmincg(costFunctionS, initial_params, options); 
                       
  [train_cost(i), ~] = costFunction(params, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_train, y_train, lambda);
                                   
  [cv_cost(i), ~] = costFunction(params, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_cv, y_cv, lambda);
                               
  if cv_cost(i) == min(cv_cost)
      bestParams = params;
  end
end

[~, bestNumLayers] = min(cv_cost);
fprintf('Optimal number of hidden layers is %d\n', bestNumLayers);

end