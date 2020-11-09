
clear ; close all; clc


data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);


fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plotData(X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;


[m, n] = size(X);

X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);

[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
pause;
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============= Optimizing using fminunc  =============

options = optimset('GradObj', 'on', 'MaxIter', 400);
initial_theta = zeros(n + 1, 1);

[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

pause;

% Plot Boundary
plotDecisionBoundary(theta, X, y);

hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')

legend('Admitted', 'Not admitted')
hold off;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============== Predict and Accuracies ==============


prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission  probability of %f\n'], prob);

% Compute accuracy on our training set
p = predict(theta, X);


fprintf('Train Accuracy: %f\n', mean(double(p == y) * 100));
fprintf('\n');


