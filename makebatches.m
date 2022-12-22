numdata=size(traindata,1);
numbatches = floor(numdata/numcases); %batchsize에 안맞는 나머지 데이터 버림

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(numdata);

batchdata = zeros(numcases, numdims, numbatches);
batchlabel = zeros(numcases, numclass, numbatches);

for batch=1:numbatches
    batchdata(:,:,batch)=traindata(randomorder((batch-1)*numcases+1:batch*numcases),:);
    batchlabel(:,:,batch)=trainlabel(randomorder((batch-1)*numcases+1:batch*numcases),:);
end


%%% Reset random seeds
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 
