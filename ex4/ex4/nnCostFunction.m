function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % 25 x 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % 10x26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1 ------------------------------------------------

X = [ones(m, 1) X]; % (5000, 401)

a1 = X;

% Theta1 (25, 401)

z2 = Theta1 * X'; % (25, 5000)

a2 = sigmoid(z2); % (25, 5000)

ma2 = size(a2, 2);

a2 = [ones(1, ma2); a2]; % (26, 5000)

% Theta2 (10 26)

z3 = Theta2 * a2;

h = sigmoid(z3);

for i = 1:m
	for k = 1:num_labels
		if y(i) == k
			J -= log(h(k, i));
		else
			J -= log(1 - h(k, i));
		end
	end
end

J /= m;

J += lambda * (sum((Theta1(:,2:input_layer_size+1).^2)(:)) 
	+ sum((Theta2(:,2:hidden_layer_size+1).^2)(:)))/2/m;

% Part 2-------------------------------------------------------------

delta_3 = zeros(num_labels, m); %10 x 5000
delta_2 = zeros(num_labels, m);

for t = 1:m
	v = 1:num_labels;
	y_binary = v == y(t);
	delta_3(:,t) =  h(:,t) - y_binary';
end

delta_2 = ((Theta2(:,2:hidden_layer_size+1))' * delta_3) .* sigmoidGradient(z2);

Theta1(:,1) = zeros(hidden_layer_size,1);
Theta2(:,1) = zeros(num_labels,1);

Theta1_grad += delta_2*(a1)/m + lambda*Theta1/m;
Theta2_grad += delta_3*(a2')/m + lambda*Theta2/m;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
