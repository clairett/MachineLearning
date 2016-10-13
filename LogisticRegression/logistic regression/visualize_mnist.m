addpath helper

images = load_mnist_images('data/train-images-idx3-ubyte');
labels = load_mnist_labels('data/train-labels-idx1-ubyte');
% display_network(images(:,1:100)); % print first 100 images


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TODO: Explore the data
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[imgSize, imgNum] = size(images);

% get the size of each image
fprintf('Image Size: %d\n', imgSize);


% get the range of labels
l = unique(labels);
lRange = max(l) - min(l);
fprintf('Range of labels: %d\n', lRange);


% get the range of pixel values
p = max(max(images)) - min(min(images));
fprintf('Range of pixel values: %d\n', p);


% compute the max and min l2-norm
l2Norm = zeros(1, imgNum);
for i = 1:imgNum
    l2Norm(i) = norm(images(:,i));
end
maxNorm = max(l2Norm);
minNorm = min(l2Norm);
fprintf('Max l2-norm: %f\n', maxNorm);
fprintf('Min l2-norm: %f\n', minNorm);


% investigate the density of data
t = imgSize * imgNum;
nonZeros = sum(sum(images~=0));
prob = nonZeros/t * 100;
fprintf('Percentage of nonzero items: %f%%\n', prob);


% investigate label distribution
plotmatrix(labels);

