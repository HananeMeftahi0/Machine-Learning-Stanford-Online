function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1');
a2 = [ones(m, 1) a2];
htheta = sigmoid(a2 * Theta2');

[temp, p] = max(htheta, [], 2);
end
