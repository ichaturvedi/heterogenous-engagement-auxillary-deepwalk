close all
tic
data = load("video_data.txt");

X = data(:,3:end);
y = data(:,2);
z = y;
 
for i=1:size(X,1)   
    if y(i) < median(data(:,2))           
          z(i)=0;      
    else
      z(i)=1;
    end
end

indi = randi(size(X,2),1,5);
indj = randi(size(X,2),1,5);

for cn = 1:1
   
%for i=1:5 %size(X,2)
   %for j=i+1:5 %size(X,2)
          i = 1 % indi(cn);
          j = 2 % indj(cn);
          %{
          a = X(:,i);
          b = X(:,j);
          co = cov([a,b]);
          co2 = co(1,2);        
          figure
          hold on  
          scatter(a(z==1),b(z==1),'b')
          scatter(a(z==0),b(z==0),'r')
          title(sprintf('%d %d %f',i,j,co2'));
          %}

%%{
datax = X(:,[i j]);
datay = ones(size(datax,1),1);



SVMModel = fitcsvm(datax,datay,'KernelScale','auto','Standardize',true,...
    'OutlierFraction',0.05);

svInd = SVMModel.IsSupportVector;
h = 0.02; % Mesh grid step size

[X1,X2] = meshgrid(min(datax(:,1)):h:max(datax(:,1)),...
    min(datax(:,2)):h:max(datax(:,2)));
[~,score] = predict(SVMModel,[X1(:),X2(:)]);
scoreGrid = reshape(score,size(X1,1),size(X2,2));

figure
%plot(datax(:,1),datax(:,2),'k.')
a = X(:,i);
b = X(:,j);

plot(a(z==1),b(z==1),'k.');
hold on
%plot(datax(svInd,1),datax(svInd,2),'ro','MarkerSize',10)
plot(a(z==0),b(z==0),'ro','MarkerSize',10);
contour(X1,X2,scoreGrid,4)
colorbar;
%title('{\bf Elearning Outlier Detection via One-Class SVM}')
title(sprintf('%d %d',i,j));
xlabel('Video 1 (sec)')
ylabel('Video 2 (sec)')
legend('High-Risk','Low-Risk')
hold off

CVSVMModel = crossval(SVMModel);
[~,scorePred] = kfoldPredict(CVSVMModel);
outlierRate = mean(scorePred<0);
outliers = scorePred > 5;

filename = sprintf('outlierC%d_%d.txt',i,j);
dlmwrite(filename,outliers);

% compute f-measure
[~,score1] = predict(SVMModel,[datax(:,1),datax(:,2)]);
dlmwrite('score4_8c.txt',score1);

level = -1:3:24;
for k=1:size(level,2)

idx = double(score1 > level(k));
idx2 = z;

cm = confusionmat(idx,idx2);
nclass = 2;
for x=1:nclass

tp = cm(x,x);
tn = cm(1,1);
for y=2:nclass
tn = tn+cm(y,y);
end
tn = tn-cm(x,x);

fp = sum(cm(:, x))-cm(x, x);
fn = sum(cm(x, :), 2)-cm(x, x);
pre(x)=tp/(tp+fp+0.01);
rec(x)=tp/(tp+fn+0.01);
fmea2(x) = 2*pre(x)*rec(x)/(pre(x)+rec(x)+0.01);
fmea(x) = (tp+tn)/(tp+fp+tn+fn);

end

afmea{k} = fmea;
afmea2{k} = fmea2;
cmall{k} = cm;
end

a = cell2mat(afmea2);
b = reshape(a,2,9);
figure
hold on
plot(b');
xticks(level)

% remove the outliers
data2 = data(~outliers,:);
data3 = data(outliers==1,:);
dlmwrite('video_data2.txt',data2);
dlmwrite('video_test.txt',data3);
%%}
  %end
  %end
end
toc

dlmwrite('outliers.txt',outliers)
