
fewcl = 1;
data=[]; 
labels=[];

load digit0; data = [data; D]; labels = [labels; repmat([1 0 0], size(D,1), 1)];
load test0; data = [data; D]; labels = [labels; repmat([1 0 0], size(D,1), 1)];
load digit1; data = [data; D]; labels = [labels; repmat([0 1 0 ], size(D,1), 1)];
load test1; data = [data; D]; labels = [labels; repmat([0 1 0], size(D,1), 1)];   
load digit2; data = [data; D]; labels = [labels; repmat([0 0 1], size(D,1), 1)];
load test2; data = [data; D]; labels = [labels; repmat([0 0 1], size(D,1), 1)];   


data = data/255;
% traindata = [traindata;repmat(data(fewind(f(1:fewdatnum)),:), 600,1)+0.1*randn(6000,size(data,2))];
% trainlabel = [trainlabel;repmat(labels(fewind(f(1:fewdatnum)),:),600,1)];
% 
% mif = randperm(size(testdata,1));
% mia = randperm(size(testdata1,1));
% mutualset = [testdata(mif(1:mutualnum),:);testdata1(mia(1:mutualnum*9),:)];
% mutuallab = [testlabel(mif(1:mutualnum),:);testlabel1(mia(1:mutualnum*9),:)];

fewind = find(labels(:,fewcl)==1);
lastind = find(labels(:,fewcl)~=1);

% fewind = find(labels(:,cl(1))==1);
% lastind = [find(labels(:,cl(2))==1);find(labels(:,cl(3))==1);find(labels(:,cl(4))==1);find(labels(:,cl(5))==1);find(labels(:,cl(6))==1)];

% fewind = find(labels(:,cl(1))==1);
% lastind = [find(labels(:,cl(2))==1);find(labels(:,cl(3))==1)];

f = randperm(size(fewind,1));
a = randperm(size(lastind,1));

datn = fix(size(lastind,1)*6/7);

% traindata = [data(lastind(a(1:datn)),:);data(fewind(f(1:fewdatnum)),:)]; 
% trainlabel = [labels(lastind(a(1:datn)),:);labels(fewind(f(1:fewdatnum)),:)];
fewdata = data(fewind(f(1:fewdatnum)),:);
fewlabel = labels(fewind(f(1:fewdatnum)),:);
traindata = [data(lastind(a(1:datn)),:);fewdata];
trainlabel = [labels(lastind(a(1:datn)),:);fewlabel];

testdata = data(fewind(f(fewdatnum+1:end)),:);
testlabel = labels(fewind(f(fewdatnum+1:end)),:);
testdata1 = data(lastind(a(datn+1:end)),:);
testlabel1 = labels(lastind(a(datn+1:end)),:);
% 
% traindata = [traindata;repmat(data(fewind(f(1:fewdatnum)),:), 600,1)+0.1*randn(6000,size(data,2))];
% trainlabel = [trainlabel;repmat(labels(fewind(f(1:fewdatnum)),:),600,1)];

mif = randperm(size(testdata,1));
mia = randperm(size(testdata1,1));
mutualset = [testdata(mif(1:mutualnum),:);testdata1(mia(1:mutualnum*9),:)];
mutuallab = [testlabel(mif(1:mutualnum),:);testlabel1(mia(1:mutualnum*9),:)];
newmutset = testdata(mif(1:mutualnum),:);
newmutlab = testlabel(mif(1:mutualnum),:);


mut_test_x = repmat(fewdata,100,1);
mut_test_z = fewdata;
mut_test_w = data(lastind(a(1:1000)),:);
mut_test_l = labels(lastind(a(1:1000)),:);