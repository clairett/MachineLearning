%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      %
%  Decision Boundary Draw For Q1.4     % 
%                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu = [0,0];
SIGMA1 = [1 0; 0 4];
x1 = -5:.1:5; x2 = -5:.1:5;
[x1,x2] = meshgrid(x1,x2);
p1 = mvnpdf([x1(:) x2(:)],mu,SIGMA1);
p1 = reshape(p1, size(x1));

SIGMA2 = [4 0; 0 1];
p2 = mvnpdf([x1(:) x2(:)],mu,SIGMA2);
p2 = reshape(p2, size(x1));

% decision boundary
d = '3/8*(x1^2-x2^2)';

figure;
hold on;
[~, h1] = contour(x1,x2,p1,'LineColor','r');
h1_ = plot(NaN, 'r');
[~, h2] = contour(x1,x2,p2,'LineColor','b');
h2_ = plot(NaN, 'b');

% draw decision boundary
h3 = ezplot(d);

% define the position for labels
xt0 = [-.5 -.5]; 
yt0 = [-4.5 4.3];
class0 = 'y=0';
text(xt0, yt0, class0, 'FontSize', 16, 'FontWeight', 'bold');

xt1 = [-4.8 4.5];
yt1 = [.5 .5];
class1 = 'y=1';
text(xt1, yt1, class1, 'FontSize', 16, 'FontWeight', 'bold');

legend([h1_ h2_ h3],'p(x|y=0)', 'p(x|y=1)', 'decision boundary', 'Location', 'north');
title('Contours of P(x|y=0) and P(x|y=1) and decision boundary');