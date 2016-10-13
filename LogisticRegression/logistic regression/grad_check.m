function err = grad_check(oracle, t, varargin)

h = 1e-6;
d = length(t);
c = length(t(1,:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TODO: Complete the function
% Hint: Use [f,g] = oracle(t, varargin{:}) to call oracle with the rest of the
% parameters
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~, gt] = oracle(t, varargin{:});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       logistic regression     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% err = 0.0;
% for i = 1:d
%     e = zeros(d, 1);
%     e(i) = 1;
%     [f1, ~] = oracle(t+h*e,varargin{:});
%     [f2, ~] = oracle(t-h*e,varargin{:});
%     gt2 = (f1 - f2)/(2*h);
%     err = err + abs(gt2-gt(i));
% end
% err = err/d;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% multiclass logistic regression  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

err = zeros(1,c);
for j = 1:c
    for i = 1:d
        e = zeros(d,c);
        e(i,j) = 1;
        [f1, ~] = oracle(t+h*e,varargin{:});
        [f2, ~] = oracle(t-h*e,varargin{:});
        gt2 = (f1 - f2)/(2*h);
        err(j) = err(j) + abs(gt2-gt(i,j));
    end  
end
err = sum(err/d,2)/c;



    