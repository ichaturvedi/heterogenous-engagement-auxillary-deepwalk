% train regression error

wordvec = load('word2vec_count.txt');
[~,idx] = sort(wordvec(:,1)); % sort just the first column
sortedmat = wordvec(idx,:);   % sort the whole matrix using the sort indices
wordvec1 = sortedmat(:,2:end);

wordvec = load('word2vec_video.txt');
[~,idx] = sort(wordvec(:,1)); % sort just the first column
sortedmat = wordvec(idx,:);   % sort the whole matrix using the sort indices
wordvec2 = sortedmat(:,2:end);

wordvec = [wordvec1(:,1:20);wordvec2];

% cluster wordvec
[idx,C,sumd,D]=kmeans(wordvec,2);
[minv,idx2] = min(D);

% load DeepWalk/data/blogcatalog.mat;
% network1 = full(network);
% network1 = network1 - diag(diag(network1));

% make network from embeddings
cv1 = cov(wordvec');
network1 = zeros(size(wordvec,1),size(wordvec,1));

% create edges for train network
for i = 1:size(wordvec,1)
        cnt =  0;
        value = cv1(i,:);
        [~,idx]=sort(value,'descend');
  
        for k = 1:size(cv1,1)
          if idx(k) < size(wordvec,1)   % parent has to be from train
            %if ( labela == outlier(idx(k),1) || labela == 1) && cnt < 5 
            if cnt < 30
              network1(i,idx(k))=1;
              cnt = cnt + 1;
            end    
          end 
        end
end

%video threshold is 60

X = load('video_data2.txt');
outlier = zeros(size(X,1),1);
% students with total score less than 150
for j=1:size(X,1)
   
    if X(j,2) < median(X(:,2))
        outlier(j) = 0;
    else
        outlier(j) = 1;               
    end
end

videox = X(:,3:end);
videoy = outlier;

% test regression error

X2 = load('video_data.txt');
outlier2 = zeros(size(X2,1),1);
% students with total score less than 150
for j=1:size(X2,1)
   
    if X2(j,2) < median(X2(:,2))
        outlier2(j) = 0;
    else
        outlier2(j) = 1;               
    end
end

videox2 = X2(:,3:end);
videoy2 = outlier2;

videoall = [videox; videox2];
outlier = [videoy; videoy2];

cv = cov(videoall');

%network2 = zeros(size(videoall,1),size(videoall,1));
%network2(1:size(videox,1),1:size(videox,1))=network1;

% create edges for test network
for i = size(videox,1)+1:size(cv,1)
        cnt =  0;
        
        value = cv(i,1:size(videox));
        %[~,idx]=sort(value,'descend');
        
        [M, ind] = max(value); % closest true click sequence
        
        if sum(network1(ind,:)) > 0
            
             y = wordvec(ind,:);
             x = wordvec(find(network1(ind,:)),:);
             b = x/y;
    
             %predict embedding
             x = wordvec(find(network1(ind,:)),:);   
             y2 = b'*x;
    
             % check which centroid
             err1 = mse(wordvec(idx2(1,1),:),y2);
             err2 = mse(wordvec(idx2(1,2),:),y2);
    
             if err1 < err2 
                  label2(i) = 0;
             else
                  label2(i) = 1;
             end 
        end
        
       
end

% compute accuracy

cm = confusionmat(label2(size(videox,1)+1:end)',outlier2);
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

fmea1
fmea2
%}

dlmwrite('test_cm.txt',cm);
dlmwrite('test_acc.txt',fmea2);
dlmwrite('test_fme.txt',fmea1);