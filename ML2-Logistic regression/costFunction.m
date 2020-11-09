function [J, grad] = costFunction(theta, X, y)

m=size(X,1);

J = 0;
h=sigmoid(X*theta);
J=-(1/m) *sum (y.*log(h) + (1 -y).* log(1 -h));
grad=zeros(size(theta,1),1);

for i=1:size(grad),
grad(i)=(1/m) *sum((h-y)'*X(:,i));
end



end
