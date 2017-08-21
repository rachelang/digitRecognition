function [bestParams, train_cost, cv_cost, bestLambda] = varyLambda(lambda_vec, ...
                                              max_iters, nn_specs, ...
                                              X_train, y_train, ...
                                              X_cv, y_cv)
                            
num_hidden_layers = nn_specs(1);
input_layer_size = nn_specs(2);
hidden_layer_size = nn_specs(3);
num_labels = nn_specs(4);
lambda = lambda_vec(1);

train_cost = size(1, length(lambda_vec));
cv_cost = size(1, length(lambda_vec));

% randomly initialize weights
initial_params = randInitWeights(num_hidden_layers, input_layer_size, hidden_layer_size, num_labels);    

for i = 1:length(lambda_vec)
  lambda = lambda_vec(i);
  fprintf('\ntraining regularized nn with %.3f lambda\n', lambda);

  % shorthand for cost function
  costFunctionS = @(p) costFunction(p, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_train, y_train, lambda);

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

[~, bestLambda_index] = min(cv_cost);
bestLambda = lambda_vec(bestLambda_index);
fprintf('Optimal lambda is %.3f\n', bestLambda);

end