function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m , 1)  X];
a1 = X;
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m , 1)  a2];
a3 = sigmoid(a2*Theta2');

ry = eye(num_labels)(y,:);

cost = ry.*log(a3) + (1 - ry).*log(1 - a3);
J = -sum(sum(cost,2)) / m;

reg = sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(: , 2:end).^2));

J = J + lambda/(2*m)*reg;


delta3 = a3 - ry;
delta2 = (delta3*Theta2)(:,2:end) .* sigmoidGradient(z2);

Delta1 = delta2'*a1;
Delta2 = delta3'*a2;

Theta1_grad = Delta1 / m + lambda*[zeros(hidden_layer_size , 1) Theta1(:,2:end)] / m;
Theta2_grad = Delta2 / m + lambda*[zeros(num_labels , 1) Theta2(:,2:end)] / m;






grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
