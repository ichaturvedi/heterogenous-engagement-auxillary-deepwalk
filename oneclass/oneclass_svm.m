function oneclass_fraud(inputfile, outputfile, num_users, th)

data = load(inputfile);
indi = randi(size(data,1)+1,num_users,1);

X = data(indi,3:end);
Xall = data(:, 3:end);
z = data(indi,2);
yall = data(:,2);

indi = randi(size(X,2),1,5);
indj = randi(size(X,2),1,5);

cn = 1;
i = indi(cn);
j = indj(cn);
datax = X(:,[i j]);
datay = ones(size(datax,1),1);

SVMModel = fitcsvm(datax,datay,'KernelScale','auto','Standardize',true,...
    'OutlierFraction',0.05);

saveLearnerForCoder(SVMModel,'SVMFraud');
%SVMModel = loadLearnerForCoder('SVMFraud');

%svInd = SVMModel.IsSupportVector;
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
xlabel('Index 1 (sec)')
ylabel('Index 2 (sec)')
legend('High-Risk','Low-Risk')
hold off

CVSVMModel = crossval(SVMModel);
[~,scorePred] = kfoldPredict(CVSVMModel);

% compute f-measure
[~,score1] = predict(SVMModel,[Xall(:,i),Xall(:,j)]);
outliersall = score1 < th;

level = -1:3:24;
for k=1:size(level,2)

idx = double(score1 > level(k));
idx2 = yall;

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
data2 = data;
data2(outliersall==1,:)=[];
dlmwrite(outputfile,data2);
dlmwrite('outliers.txt',outliersall)
pause(5)

end
