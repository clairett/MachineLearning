D = load('SVR_dataset.txt');
X = D(:,1);
Y = D(:,2);

[N, ~] = size(X);
h = 0.5;

% RBF Kernel
K = exp(-(repmat(X, 1, N)-repmat(X',N, 1)).^2/(2*h*h));


% train SVR 
C = 4;
cvx_begin
    variables a(N) a_star(N)
    maximize( -1/2.*quad_form((a-a_star),K)-0.1*sum(a+a_star)+sum(Y.*(a-a_star)) )
    subject to
        0 <= a <= C;
        0 <= a_star <= C;
cvx_end

x = 0:0.0001:1;
[~,M] = size(x);
K1 =  exp(-(repmat(x', 1, N) - repmat(X', M, 1)).^2/(2*h*h));
f = K1*(a - a_star);


% criterion to see if a - a_star is 0
diff = 0.00001; 
indexs = a - a_star > diff;

figure;
hold on;

plot(x, f);
scatter(X(indexs), Y(indexs), 'g*');
scatter(X, Y, 'rs');
legend({'Prediction Curve with x in [0,1]', 'Support Vectors', 'Training Points'}, 'Location', 'north');


        



    