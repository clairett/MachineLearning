data = load('kmeans_data.csv');

[n, ~] = size(data);

M = 50;  % number of iterations
k = 5;   % number of clusters
bobj = Inf;
bface = [];

fig = figure;

%% Run 15 times
for t = 1:15
    
    %initialization 
    labels = zeros(n, 1);
    muID = randi(n, k, 1);
    mu = data(muID, :);
    iter = 0;    % record the number of iterations
    obj = [];    % record objectives
    
    while (true)
        % calculate distance and update label
        for i = 1:n
            dis = distance(data(i,:), mu, k);
            [~,l] = min(dis);
            labels(i) = l;
        end
        
        % update means
        for j = 1:k
            mu(j,:) = mean( data(labels == j,:), 1);
        end
        iter = iter + 1;
        
        obj(iter) = 0;
        for m = 1:n
            obj(iter) = obj(iter) + min(distance(data(m,:), mu, k));
        end
        
        % calculate objective and check convergence
        if iter>2 && obj(iter-1)==obj(iter)
            fprintf('Run %d: converges at %dth iteration.\n', t, iter);
            break;
        end
        
        % break if reach the required iteration
        if(iter == M)
            break;
        end
        
    end
   
    % plot objectives
    hold on ;
    plot(2*obj);
    
    % get the best objective and best mean face
    if bobj > obj(end)
        bobj = obj(end);
        bface = mu;
    end

end

legend('Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5', 'Run 6', 'Run 7', 'Run 8', 'Run 9', 'Run 10', 'Run 11', 'Run 12', 'Run 13', 'Run 14', 'Run 15');
saveas(fig,'kmeans.png');
close all;

%% draw best mean face 
[num, ~] = size(bface);
for i2 = 1:num
    img = vec2mat(bface(i2,:), 19, 19);
    fig = figure;
    face = imshow(img', [0.0 255.00], 'InitialMagnification', 'fit');
    saveas(face, strcat('face-', num2str(i2), '.png'));
end
close all;


