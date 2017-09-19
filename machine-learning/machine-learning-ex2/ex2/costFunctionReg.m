function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



h = sigmoid(X * theta);

inner = -1 * y .* log(h) - (1 - y) .* log(1 - h);
J_1 = (1/m) * inner(1);
reg_J = (1/m) * sum(inner(2:size(inner))) + lambda/(2 * m) * sum(theta(2:size(theta)).^2);

inner_grad = h - y; 
grad_1 = (1/m) * X(:, 1)' * inner_grad;
reg_grad = (1/m) * X(:, 2:size(X,2))' * inner_grad + (lambda/m)* theta(2:size(theta));

J = J_1 + reg_J;
grad = [grad_1; reg_grad];

% =============================================================

end
