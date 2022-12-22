% There is some parameters needed to be modifed w.r.t dataset you're using.
% those parameters are designated with some text next to it.

clear;
close all;

%% Data preprocessing and indexing for cross validation
%input을 data에 output을 labels에 저장 -> 데이터 한 개는 한 행으로 표현
differ = [];
% converter;
 mnistdata;
% 
 [numdata, numdims] = size(data); % total number of data(here, 70000 for mnist data)
 numclass = size(labels,2); % number of class
% 
%

numfold = 10; % you can change the number of folds for k-fold cross validation.
kfoldind = crossvalind('Kfold', numdata, numfold);
% 
%% Main code
for cv = 1:1 % change 1 to 'numfold' in actual execution
    %% make test dataset and training dataset
    testind = find(kfoldind == cv);
    trainind = find(kfoldind ~= cv);
    
    testdata = data(testind,:);
    testlabel = labels(testind,:);
    
    traindata = data(trainind,:);
    trainlabel = labels(trainind,:);
end

testdata1 = testdata(1,:);
testlabel1 = testlabel(1,:);
alltest = testdata1;
alllabel = testlabel1;
feedback = 1;

% miss_6class;
numdims = size(traindata,2);
numclass = size(trainlabel,2);
%% make minibatches from training data

numcases = 100; % can be modified w.r.t data number.
%% Layerwise Pre-training
numhid = 50;
maxepoch = 100; % num of epoch
makebatches;
rbm;

% Last layer
numpen = 50;
maxepoch = 100; % num of epoch
makebatches;
rbm_last;
disp(['Pretraining with ', num2str(numhid), ' hid units and ', num2str(numpen), ' pen units completed' ])

%% Deep learning
maxepoch = 100;
makebatches;
dbm;

% Set dropout rate
dr_vis = 0.2;
dr = 0.5;
p_vis = 1-dr_vis;
p = 1-dr;
%% Fine-tuning with backpropagation
maxepoch= 100;
makebatches; 
backprop;
