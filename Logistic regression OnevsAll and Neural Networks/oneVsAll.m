function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);


X = [ones(m, 1) X];

options = optimset('GradObj', 'on', 'MaxIter', 50);
for i=1:num_labels
	initial_theta = zeros(size(X, 2), 1);
	[theta] = fmincg(@(t)(lrCostFunction(t, X,y==i, lambda)), initial_theta, options);
	all_theta(i,:) = theta';
end


end
