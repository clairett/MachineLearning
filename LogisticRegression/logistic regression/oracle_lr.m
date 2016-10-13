% Returns the function and gradient evaluated at w. 
% w: (d+1) x 1
% X: (d+1) x n
% y: 1 x n
function [f, g] = oracle_lr(w, X, y)

% f = 0;
% g = zeros(size(w));
lambda = ones(size(w)) * 10; % used in Q2.2.7
lambda(1) = 0; % do not penalize the bias term

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TODO: Complete the function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

z = w' * X; % dimension: 1 x n
h = sigmoid(-z);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     objective and gradient    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% f = sum(y.* z - log(1+exp(z)));
% g = X*(y - (1-h))';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% l2 regularized objective and gradient  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f = sum(y.* z - log(1+exp(z))) - (lambda/2 .* w)'* w;
g = X*(y - (1-h))' - lambda .* w;








        
    


