% Backpropagation like other NN
%% Preprocessing data

testnum = size(testdata,1);
[~,temp_h2_test] = mf_class(testdata,vishid,hidbiases,hidpen,penbiases);
% 
testnum1 = size(testdata1,1);
[~,temp_h2_test1] = mf_class(testdata1,vishid,hidbiases,hidpen,penbiases);

[~,fewh2] = mf_class(fewdata,vishid,hidbiases,hidpen,penbiases);

temp_h2_train = zeros(numcases,numpen,numbatches);

for batch = 1:numbatches
   data = batchdata(:,:,batch);
   [~, temp_h2] =  mf_class(data,vishid,hidbiases,hidpen,penbiases);
   temp_h2_train(:,:,batch) = temp_h2;
end

%% Initialize weights

w1_penhid = hidpen';
w1_vishid = vishid;
w2 = hidpen;
h1_biases = hidbiases; h2_biases = penbiases; 

w_class = 0.1*randn(numpen,numclass); 
topbiases = 0.1*randn(1,numclass);

numdims_tot = numdims;
numhid_tot = numhid;
numpen_tot = numpen;

disp('start backpropagation');
for epoch = 1:maxepoch
  
  %Calculate test error
  bias_hid= repmat(h1_biases,testnum,1);
  bias_pen = repmat(h2_biases,testnum,1);
  bias_top = repmat(topbiases,testnum,1);

  hidprob_test = 1./(1+exp(- testdata*w1_vishid - temp_h2_test*w1_penhid - bias_hid));
  penprob_test = 1./(1+exp(- hidprob_test*w2 - bias_pen));
  totin = penprob_test*w_class + bias_top;
  totin = totin - max(max(totin));
  target_test = exp(totin);
  target_test = target_test./repmat(sum(target_test,2),1,numclass);

  [~, J] = max(target_test,[],2);
  [~, J1] = max(testlabel,[],2);
  mis_test = sum(J~=J1);
  test_cr = - sum(sum(testlabel(:,:).*log(target_test)));
   
    % calculate test 1 error
    bias_hid= repmat(h1_biases,testnum1,1);
    bias_pen = repmat(h2_biases,testnum1,1);
    bias_top = repmat(topbiases,testnum1,1);

    hidprob_test1 = 1./(1+exp(- testdata1*w1_vishid - temp_h2_test1*w1_penhid - bias_hid));
    penprob_test1 = 1./(1+exp(- hidprob_test1*w2 - bias_pen));
    totin = penprob_test1*w_class + bias_top;
    totin = totin - max(max(totin));
    target_test1 = exp(totin);
    target_test1 = target_test1./repmat(sum(target_test1,2),1,numclass);

    [~, J] = max(target_test1,[],2);
    [~, J1] = max(testlabel1,[],2);
    mis_test1 = sum(J~=J1);
    test_cr1 = - sum(sum(testlabel1(:,:).*log(target_test1))); 
    
 
  %Calculate train error
  bias_hid= repmat(h1_biases,numcases,1);
  bias_pen = repmat(h2_biases,numcases,1);
  bias_top = repmat(topbiases,numcases,1);
   
  mis_train = 0;
  train_cr = 0;
  
  for batch = 1:numbatches
    data = batchdata(:,:,batch);
    temp_h2 = temp_h2_train(:,:,batch); 
    target = batchlabel(:,:,batch);

    hidprob_train = 1./(1+exp(- data*w1_vishid - temp_h2*w1_penhid - bias_hid));
    penprob_train = 1./(1+exp(- hidprob_train*w2 - bias_pen));
    totin = penprob_train*w_class + bias_top;
    totin = totin - max(max(totin));
    target_train = exp(totin);
    target_train = target_train./repmat(sum(target_train,2),1,numclass);
    
    [~, J] = max(target_train,[],2);
    [~, J1] = max(target,[],2);
    
    mis_train = mis_train + sum(J~=J1);
    train_cr = train_cr - sum(sum(target.*log(target_train)));
  end
 

   disp(['epoch ',num2str(epoch), ' test misclassification : ', num2str(mis_test),' out of ',num2str(testnum), ' cross entropy ', num2str(test_cr)]);
  disp(['epoch ',num2str(epoch), ' test1 misclassification : ', num2str(mis_test1),' out of ',num2str(testnum1), ' cross entropy ', num2str(test_cr1)]);
   disp(['epoch ',num2str(epoch), ' train misclassification : ', num2str(mis_train),' out of ',num2str(numcases*numbatches), ' cross entropy ', num2str(train_cr)]);
   
%   errp(epoch,:)=[mis_test;mis_test1;mis_train];
   %% Conjugate Gradient Optimization
   % Variables for dropout
    w1_penhid_tot = w1_penhid;
    w1_vishid_tot = w1_vishid;
    w2_tot = w2;
    h1_biases_tot = h1_biases;
    h2_biases_tot = h2_biases;
    w_class_tot = w_class;
 
   rr = randperm(numbatches);
   div_value = 100;
   value = floor(numbatches/div_value);
   
  for batch = 1:value
        visind = rand(1,numdims_tot) > dr_vis;
        while(sum(visind) < 1)
            visind = rand(1,numdims_tot) > dr_vis;
        end
        visind = find(visind == true);
        numdims = length(visind);
        
        visind_2 = rand(1,numpen_tot) > dr_vis;
        while(sum(visind_2) < 1)
            visind_2 = rand(1,numpen_tot) > dr_vis;
        end
        visind_2 = find(visind_2 == true);
        numdims_2 = length(visind_2);
        
        hidind = rand(1,numhid_tot) > dr;
        while(sum(hidind) < 1)
            hidind = rand(1,numhid_tot) > dr;
        end
        hidind = find(hidind == true);
        numhid = length(hidind);
        
        penind = rand(1,numpen_tot) > dr;
        while(sum(penind) < 1)
            penind = rand(1,numpen_tot) > dr;
        end
        penind = find(penind == true);
        numpen = length(penind);
        
        w1_penhid = w1_penhid_tot(visind_2,hidind)/p_vis;
        w1_vishid = w1_vishid_tot(visind,hidind)/p_vis;
        w2 = w2_tot(hidind,penind)/p;
        h1_biases = h1_biases_tot(:,hidind);
        h2_biases = h2_biases_tot(:,penind);
        w_class = w_class_tot(penind,:)/p;

        
      data = zeros(numcases*div_value,numdims);
      temp_h2 = zeros(numcases*div_value,numdims_2);
      targets = zeros(numcases*div_value,numclass);
      tt1=(batch-1)*div_value+1:batch*div_value;
      
      for tt = 1:div_value
            data((tt-1)*numcases+1:tt*numcases,:) = batchdata(:,visind,rr(tt1(tt)));
            temp_h2( (tt-1)*numcases+1:tt*numcases,:) = temp_h2_train(:,visind_2,rr(tt1(tt)));
            targets( (tt-1)*numcases+1:tt*numcases,:) = batchlabel(:,:,rr(tt1(tt)));
      end
        
      % CG with 3 linesearch

      VV = [w1_vishid(:)' w1_penhid(:)' w2(:)' w_class(:)' h1_biases(:)' h2_biases(:)' topbiases(:)']';
      Dim = [numdims; numdims_2; numhid; numpen; numclass];

      max_iter=3; 
      if epoch<6
        [X, fX, num_iter,ecg_XX] = minimize(VV,'CG_MNIST_INIT',max_iter,Dim,data,targets,temp_h2);
      else
        [X, fX, num_iter,ecg_XX] = minimize(VV,'CG_MNIST',max_iter,Dim,data,targets,temp_h2);
      end 
      
        w1_vishid_tot(visind,hidind) = reshape(X(1:numdims*numhid)*p_vis,numdims,numhid);
        xxx = numdims*numhid;
        w1_penhid_tot(visind_2,hidind) = reshape(X(xxx+1:xxx+numdims_2*numhid)*p_vis,numdims_2,numhid);
        xxx = xxx+numdims_2*numhid;
        w2_tot(hidind,penind) = reshape(X(xxx+1:xxx+numhid*numpen)*p,numhid,numpen);
        xxx = xxx+numhid*numpen;
        w_class_tot(penind,:) = reshape(X(xxx+1:xxx+numpen*numclass)*p,numpen,numclass);
        xxx = xxx+numpen*numclass;
        h1_biases_tot(:,hidind) = reshape(X(xxx+1:xxx+numhid),1,numhid);
        xxx = xxx+numhid;
        h2_biases_tot(:,penind) = reshape(X(xxx+1:xxx+numpen),1,numpen);
        xxx = xxx+numpen;
        topbiases = reshape(X(xxx+1:xxx+numclass),1,numclass);
        xxx = xxx+numclass;
  end
  numdims = numdims_tot;
    numhid = numhid_tot;
    numpen = numpen_tot;
    w1_penhid = w1_penhid_tot;
    w1_vishid = w1_vishid_tot;
    w2 = w2_tot;
    h1_biases = h1_biases_tot;
    h2_biases = h2_biases_tot;
    w_class = w_class_tot;
    
    % mutual information of each layer
    


end

save backprop_para w1_penhid w1_vishid w2 h1_biases h2_biases w_class topbiases

acc(1,feedback) = (testnum-mis_test)/testnum;
acc(2,feedback) = (testnum1-mis_test1)/testnum1;
