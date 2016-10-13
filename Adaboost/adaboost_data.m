clear;
figure; hold on;
xpos = [1 2 5 5];
ypos = [2 3 1 2];
xneg = [3 3 3 4 5];
yneg = [1 2 4 4 4];
plot(xpos, ypos, 'b^', 'linewidth', 3, 'markersize', 17);
plot(xneg, yneg, 'mo', 'linewidth', 3, 'markersize', 17);
text(1-0.4, 2, 'x_1', 'fontsize', 20);
text(2-0.4, 3, 'x_2', 'fontsize', 20);
text(3-0.4, 4, 'x_3', 'fontsize', 20);
text(3-0.4, 2, 'x_4', 'fontsize', 20);
text(3-0.4, 1, 'x_5', 'fontsize', 20);
text(4-0.4, 4, 'x_6', 'fontsize', 20);
text(5-0.4, 4, 'x_7', 'fontsize', 20);
text(5-0.4, 2, 'x_8', 'fontsize', 20);
text(5-0.4, 1, 'x_9', 'fontsize', 20);
set(gca,'FontSize',20);
axis([0 6 0 5]);
set(gca, 'YTick', 0:1:5);

pos = [xpos' ypos'];
neg = [xneg' yneg'];

%% Define training data
datafeatures = [pos; neg];
dataclass(1:4) = 1;
dataclass(5:9) = -1;

%% use AdaBoost to make a classifier
[classestimate, model]=adaboost('train', datafeatures, dataclass, 3);
pos = datafeatures(classestimate==1, :);
neg = datafeatures(classestimate==-1, :);

plot(pos(:,1), pos(:,2));
plot(neg(:,1), neg(:,2));

error=zeros(1,length(model)); 
for i=1:length(model)
    error(i)=model(i).error; 
end



