function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h= zeros(size(y),1);
h=  sigmoid(X * theta);
beta= h-y;
mult=zeros(size(grad),1);
mult= (1/m)*(X'*beta);
grad(1) = mult(1);
theta2= zeros(size(theta-1),1);
theta2= (lambda/m)*theta(2:size(theta));
grad(2:size(grad))= mult(2:size(mult))+theta2;
 
 %COST FUNCTION

suma= zeros(size(y),1);
unos= ones(size(y),1);
suma= ((-y).*log(h))-((unos-y).*log(1-h));
suma= sum(suma);
theta3=zeros(size(theta)-1,1);
theta3= theta(2:size(theta));
suma2= sum(theta3.^2);
suma2= (lambda/(2*m))*suma2;

J= (suma/m)+suma2;









% =============================================================

grad = grad(:);

end
