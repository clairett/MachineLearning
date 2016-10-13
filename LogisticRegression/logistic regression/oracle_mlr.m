% Oracle mlr returns the function and gradient evaluated at w. 
% W: (d+1) x c
% X: (d+1) x n
% y: 1 x n
function [f, g] = oracle_mlr(W, X, y)

% g = zeros(size(W));
Lambda = ones(size(W)) * 10;
Lambda(1,:) = 0; % do not penalize the bias term

% Y(i,j) = 1 if y(j) == i, otherwise 0
Y = full(sparse(y, 1:length(y), 1)); % c x n

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TODO: Complete this function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~, N] = size(X);
[~, C] = size(W);
 
h = W'*X;

% l2 regularizer
r = trace((Lambda/2.* W)'* W);

% objective function
f1 = sum(sum(Y.*h),2);
f2 = 0;
for i = 1 : N
    f2 = f2 + log(sum(exp(W'* X(:,i))));
end
f = f1 - f2 - r;

% compute probability u
zc = exp(h);
temp = sum(zc);
z = repmat(temp, C, 1);
u = zc./z;

% gradient
g = X*(Y-u)' - Lambda.*W;








    
