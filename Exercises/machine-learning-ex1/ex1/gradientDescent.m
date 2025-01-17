function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    
    temp_sum1 = 0;
    temp_sum2 = 0;

    for i = 1 : m
        temp_sum1 = temp_sum1 + (X(i, :) * theta - y(i));
        temp_sum2 = temp_sum2 + (X(i, :) * theta - y(i)) * X(i, 2);
    end
    
    
    theta(1) = theta(1) - alpha / m * temp_sum1;
    theta(2) = theta(2) - alpha / m * temp_sum2;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
