%ML4

%Regularized Linear Regression
%Procedure 4.1
%load the data
x = load('ml4Linx.dat');
y = load('ml4Liny.dat');
%plot 
plot(x, y, 'o');
xlabel('x');
ylabel('y');
hold on;
[m,n] = size(x);
%5th order polynomial
x = [ones(m,1), x, x.^2, x.^3, x.^4, x.^5];
%Procedure 4.2
%normal equation theta
theta = zeros(size(x(1,:)))';
%set regularization parameter(lambda) [0, 1, 10];
lambda = 10;
L = lambda .* eye(6);
L(1) = 0;
theta = (x' * x + L)\(x' * y);
%theta @ lambda = 0: [0.472528772874319; 0.681352894856652; -1.380128418612173; -5.977687467469015; 2.441732684792994; 4.737114334830815]
%theta @ lambda = 1: [0.397595299175466; -0.420666371376896; 0.129592111980193; -0.397473899391432; 0.175255526708740; -0.339387717362337]
%theta @ lambda = 10: [0.520470738359628; -0.182507058295231; 0.060642582038725; -0.148177206219198; 0.074330064766671; -0.127957368751850]

%Procedure 4.3
theta_norm = norm(theta);
%set x axis 
x_vals = (-1:0.05:1)';
%set regularization line
feats = [ones(size(x_vals)), x_vals, x_vals .^2,...
    x_vals.^3, x_vals.^4, x_vals.^5];
plot(x_vals, feats * theta, '--');
xlabel('x-vals');
ylabel('features');
legend('Training data', 'Lambda = 10');

%From looking at the previous graphs, what conclusions can you make about how the regularization parameter lambda affects your model?
%lambda = 0 is overfitting whereas lambda = 10 is underfitting. lambda = 1 is just right
%according to the manual

