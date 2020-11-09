function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    h=X*theta;
    t1=theta(1)-alpha*(1/m)*sum(h-y);
    t2=theta(2)-alpha*(1/m)*sum((h-y).*X(:,2));
    theta(1)=t1;
    theta(2)=t2;
    
    J_history(iter) = computeCost(X, y, theta);

end

end
