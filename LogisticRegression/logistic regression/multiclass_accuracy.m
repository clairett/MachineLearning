% Returns the accuracy of multiclass classifier W
% W: (d+1) x c
% X: (d+1) x n
% y: 1 x n
function acc = multiclass_accuracy(W, X, y)

% pred = zeros(size(y));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TODO: Complete this function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~, C] = size(W);

p1 = exp(W'*X);
p2 = sum(p1);
p = repmat(p2, C, 1);
u = p1./p;

[~, pred] = max(u);
    
nacc = sum(y == pred);
acc = nacc / length(y);
