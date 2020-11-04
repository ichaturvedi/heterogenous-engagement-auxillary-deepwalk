function [acc, fmea] = network_regression(embeddings, networkfile, trainfile, testfile, num_neighbours)

wordvec = load(embeddings);
[~,idx] = sort(wordvec(:,1)); % sort just the first column
sortedmat = wordvec(idx,:);   % sort the whole matrix using the sort indices
wordvec = sortedmat(:,2:end);

% cluster wordvec
[idx,C,sumd,D]=kmeans(wordvec,2);
[minv,idx2] = min(D);

networkc = load(networkfile);
network1 = full(networkc.network);
network1 = network1 - diag(diag(network1));

%video threshold is 60

X = load(trainfile);
outlier = X(:,2);
%outlier(outlier==2)=1;

videox = X(:,3:end);
videoy = outlier;

% test regression error

X2 = load(testfile);
outlier2 = X2(:,2);
videox2 = X2(:,3:end);
videoy2 = outlier2;

videoall = [videox; videox2];
outlier = [videoy; videoy2];

cv = cov(videoall');
network2 = zeros(size(videoall,1),size(videoall,1));
network2(1:size(videox,1),1:size(videox,1))=network1;

% create edges for test network
for i = size(videox,1)+1:size(cv,1)
        cnt =  0;
        value = cv(i,:);
        [~,idx]=sort(value,'descend');
        
        % we dont have this for test
        labela = outlier(i,1);
  
        for k = 1:size(cv,1)
          if idx(k) < size(videox,1)   % parent has to be from train
            %if ( labela == outlier(idx(k),1) || labela == 1) && cnt < 5 
            if cnt < num_neighbours
              network2(i,idx(k))=1;
              cnt = cnt + 1;
            end    
          end 
        end
end

% compte regression coefficent in video data
% predict embedding using coefficients

err = 0;
for i=1:size(videoall,1)
    if sum(network2(i,:)) > 0
       
    y = videoall(i,:);
    x = videoall(find(network2(i,:)),:);
    b = x/y;
    
    %predict embedding
    x = wordvec(find(network2(i,:)),:);   
    y2 = b'*x;
    
    % check which centroid
    err1 = mse(wordvec(idx2(1,1),:),y2);
    err2 = mse(wordvec(idx2(1,2),:),y2);
    
    if err1 < err2 
        label2(i) = outlier(idx2(1,1));
    else
        label2(i) = outlier(idx2(1,2));
    end
    
    end
end

% compute accuracy
label2b = label2(size(videox,1)+1:end);
outlierb = outlier(size(videox,1)+1:end);

%cm = confusionmat(label2(:,size(X,1):end)',outlier(size(X,1):end,:));
cm = confusionmat(label2',outlier);
nclass = 2;
for x=1:nclass

tp = cm(x,x);
tn = cm(1,1);
for y3=2:nclass
tn = tn+cm(y3,y3);
end
tn = tn-cm(x,x);

fp = sum(cm(:, x))-cm(x, x);
fn = sum(cm(x, :), 2)-cm(x, x);
pre(x)=tp/(tp+fp+0.01);
rec(x)=tp/(tp+fn+0.01);
fmea1(x) = 2*pre(x)*rec(x)/(pre(x)+rec(x)+0.01);
fmea2(x) = (tp+tn)/(tp+fp+tn+fn);

end

fmea = mean(fmea1);
acc = mean(fmea2);

end