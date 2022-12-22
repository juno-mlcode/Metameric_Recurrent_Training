% Deep learning

%% Initialize learning rate, momentum

 epsilonw      = 0.001;   % Learning rate for weights 
 weightcost   = 0.0002;
 
 initialmomentum  = 0.5;
 finalmomentum    = 0.9;

%sparsity penalty
 sparsetarget =  0.2;
 sparsetarget2 = 0.1;
 sparsecost =  0.001;
 sparsedamping = 0.9;

hidmeans = sparsetarget*ones(1,numhid);
penmeans = sparsetarget2*ones(1,numpen);

vishidinc  = zeros(numdims,numhid);
hidpeninc  = zeros(numhid,numpen);
labpeninc =  zeros(numclass,numpen); 

visbiasinc = zeros(1,numdims);
hidbiasinc = zeros(1,numhid);
penbiasinc = zeros(1,numpen);
labbiasinc = zeros(1,numclass);

% sum hidbiases from crbm and crbm_last
hidbiases = hb + hidbiases;

%% Start learning
disp(['start deep learning']);
for epoch = 1:maxepoch 

  errsum=0;
  rr = randperm(numbatches);  

  for batch_rr = rr %1:numbatches
    
    %Start Positive Phase
    data = batchdata(:,:,batch_rr);
    targets = batchlabel(:,:,batch_rr); 
    data = data > rand(numcases,numdims);  

   [poshidprobs, pospenprobs] =  mf(data,targets,vishid,hidbiases,hidpen,penbiases,labpen); %MF approximation
    
    bias_vis = repmat(visbiases,numcases,1);
    bias_hid = repmat(hidbiases,numcases,1);
    bias_pen = repmat(penbiases,numcases,1);
    bias_lab = repmat(labbiases,numcases,1);
  
    posprods    = data' * poshidprobs;
    posprodspen = poshidprobs'*pospenprobs;
    posprodslabpen = targets'*pospenprobs;
    
    posvisact   = sum(data);
    poshidact   = sum(poshidprobs);
    pospenact   = sum(pospenprobs);
    poslabact   = sum(targets); 

    
    %Start Negative Phase
    neghidprobs = poshidprobs;
    negpenprobs = pospenprobs;
    
    for iter=1:5 %CD iteration 5
        neghidstates = neghidprobs > rand(numcases,numhid); 
        negpenstates = negpenprobs > rand(numcases,numpen);
        
        negdataprobs = 1./(1 + exp(-neghidstates*vishid' - bias_vis));
        negdata = negdataprobs > rand(numcases,numdims);
        
        totin = negpenstates*labpen' + bias_lab;
        totin = totin - max(max(totin));
        neglabprobs = exp(totin);
        neglabprobs = neglabprobs./repmat(sum(neglabprobs,2),1,numclass); 
        xx = cumsum(neglabprobs,2);
        xx1 = rand(numcases,1);
        neglabstates = neglabstates*0;
        for jj=1:numcases
            index = find(xx1(jj) <= xx(jj,:),1);
            neglabstates(jj,index) = 1;
        end
       
        neghidprobs = 1./(1+exp(- negdata*vishid - negpenstates*hidpen' - bias_hid));
        negpenprobs = 1./(1+exp(- neglabstates*labpen - neghidprobs*hidpen - bias_pen));
        
    end 

    negprods  = negdata'*neghidprobs;
    negprodspen = neghidprobs'*negpenprobs;
    negprodslabpen = neglabstates'*negpenprobs;
    
    negvisact = sum(negdata); 
    neghidact = sum(neghidprobs);
    negpenact = sum(negpenprobs);
    neglabact = sum(neglabstates); 
    

    %Calculate error
    err = sum(sum((targets-neglabstates).^2));
    errsum = errsum + err/2;
    
    %Update weights
    if epoch>5
     momentum=finalmomentum;
    else
     momentum=initialmomentum;
    end

   visbiasinc = momentum*visbiasinc + (epsilonw/numcases)*(posvisact-negvisact);
   labbiasinc = momentum*labbiasinc + (epsilonw/numcases)*(poslabact-neglabact);

   hidmeans = sparsedamping*hidmeans + (1-sparsedamping)*poshidact/numcases;
   sparsegrads = sparsecost*(repmat(hidmeans,numcases,1)-sparsetarget);

   penmeans = sparsedamping*penmeans + (1-sparsedamping)*pospenact/numcases;
   sparsegrads2 = sparsecost*(repmat(penmeans,numcases,1)-sparsetarget2);
    
   vishidinc = momentum*vishidinc + epsilonw*( (posprods-negprods)/numcases - weightcost*vishid - data'*sparsegrads/numcases);
   hidpeninc = momentum*hidpeninc + epsilonw*( (posprodspen-negprodspen)/numcases - weightcost*hidpen - ...
       poshidprobs'*sparsegrads2/numcases - (pospenprobs'*sparsegrads)'/numcases );
   labpeninc = momentum*labpeninc + epsilonw*( (posprodslabpen-negprodslabpen)/numcases - weightcost*labpen); 
   
   hidbiasinc = momentum*hidbiasinc + epsilonw*(poshidact-neghidact)/numcases - epsilonw*sum(sparsegrads)/numcases;
   penbiasinc = momentum*penbiasinc + epsilonw*(pospenact-negpenact)/numcases - epsilonw*sum(sparsegrads2)/numcases;

   vishid = vishid + vishidinc;
   hidpen = hidpen + hidpeninc;
   labpen = labpen + labpeninc;
   
   visbiases = visbiases + visbiasinc;
   hidbiases = hidbiases + hidbiasinc;
   penbiases = penbiases + penbiasinc;
   labbiases = labbiases + labbiasinc;
   
  end
  
  % Display error
  
  if rem(epoch,5)==0
      disp(['Number of misclassified training examples: ', num2str(errsum), ' out of ', num2str(numcases*numbatches)]);
  end
  
end

if feedback< 3
    [mih, mip] = mf(mutualset,mutuallab,vishid,hidbiases,hidpen,penbiases,labpen);
    mil = mutuallab;
    mutual_information;
    I_all_X(1,feedback) = ITX;
    I_all_Y(1,feedback) = ITY;
    
    mih = mip;
    mutual_information;
    I_all_X(2,feedback) = ITX;
    I_all_Y(2,feedback) = ITY;
end

