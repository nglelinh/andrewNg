function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

short_theta = theta(2:size(theta),1);

hx = X * theta;

J = ((hx - y)' * (hx - y))/2/m + lambda * (short_theta' * short_theta)/2/m;

short_theta = [0 short_theta'];

grad = ((hx - y)' * X)/m + lambda * short_theta / m; %row;

% =========================================================================

grad = grad(:);

end
