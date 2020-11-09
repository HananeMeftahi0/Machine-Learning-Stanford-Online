function p = predict(theta, X)

p = sigmoid( X * theta);
p = p >= 0.5;
end
