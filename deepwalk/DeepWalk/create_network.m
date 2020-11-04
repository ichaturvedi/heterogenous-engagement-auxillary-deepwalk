function create_network(inputfile, num_neighbours)

X = load(inputfile);
outlier = X(:,2);
outlier2 = outlier;
clickx = X(:,3:end);
cv = cov(clickx');
network1 = zeros(size(clickx,1),size(clickx,1));

for i = 1:size(cv,1)
        cnt =  0;
        value = cv(i,:);
        [~,idx]=sort(value,'descend');
        
        labela = outlier(i,1);
  
        for k = 1:size(cv,1)
          if ( labela == outlier(idx(k),1) || labela == 1) && cnt < num_neighbours
             network1(i,idx(k))=1;
             cnt = cnt + 1;
          end    
        end
end

network = sparse(network1);

group1 = [0 0];
for i=1:size(outlier2,1)
   
    if outlier(i,1) == 0
        group1 = [group1; [0 1]];
    else
        group1 = [group1; [1 0]];
    end
    
end
group1 = group1(2:end,:);
group = sparse(group1);

save network.mat group network
end

