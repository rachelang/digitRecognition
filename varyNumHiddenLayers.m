function [Theta, costResults, bestNumLayers] = varyNumHiddenLayers(max_hidden_layers, ...
                                              nn_specs, ...
                                              initial_params, ...
                                              X_train, y_train, ...
                                              X_cv, y_cv)
                            
num_hidden_layers = nn_specs(1);
input_layer_size = nn_specs(2);
hidden_layer_size = nn_specs(3);
num_labels = nn_specs(4);
lambda = 0; % lambda will be varied in varyLambda

costResults = size(1, max_hidden_layers);
% shorthand for cost function
costFunction = @(p) costFunction(p, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_train, y_train, lambda);
                                   
for num_hidden_layers = 1:max_hidden_layers
  fprintf('\training nn with %d hidden layer(s)\n', num_hidden_layers);

  options = optimset('GradObj', 'on ', 'MaxIter', 10);   
  [Theta, cost] = fmincg(costFunction, initial_params, options); 
  costResults(i) = cost;

  Theta = reshapeParams(Theta, num_hidden_layers, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels);
                              
  y_pred = predict(Theta, X_cv);
  accuracy = mean(double(y_pred == y_cv));
  if i == 1
    highestAccuracy = accuracy;
    bestNumLayers = 1;
  else
    if accuracy > highestAccuracy
      highestAccuracy = accuracy;
      bestNumLayers = i;
    end
  end
  
  fprintf('\n%d hidden layers cross validation set accuracy: %f\n', num_hidden_layers, accuracy * 100);
end