%Pre-training last layer, layer hidden layer name 'pen'
%save previous hidden biases in hb
hb = hidbiases;

%% Initialize learning rate, momentum

epsilonw      = 0.05;   % Learning rate for weights 
weightcost  =  0.001;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;

%% Initializing symmetric weights and biases.
hidpen = 0.01*randn(numhid, numpen);
labpen = 0.01*randn(numclass,numpen);

hidbiases = zeros(1,numhid); % re-initialized for hid-pen layer pretraining
penbiases = zeros(1,numpen);
labbiases = zeros(1,numclass);

posprods = zeros(numhid,numpen);
negprods = zeros(numhid,numpen);


hidpeninc = zeros(numhid,numpen);
labpeninc = zeros(numclass,numpen);
hidbiasinc = zeros(1,numhid);
penbiasinc = zeros(1,numpen);
labbiasinc = zeros(1,numclass);


%% Start learning
disp('start last layer pre-training');

for epoch = 1:maxepoch
    
    if rem(epoch,10)==10
        disp(['last pre-training epoch :', num2str(epoch)]);
    end
  
    %decreasing learning rate
     CD = ceil(epoch/20);
     epsilonw = epsilonw/CD;
 
     errsum=0;
    for batch = 1:numbatches

        %Start Positive Phase
        data = batchdata(:,:,batch);
        data = data > rand(numcases,numdims);
        
        poshidprobs = 1./(1+exp(-data*(2*vishid) - repmat(2*hb,numcases,1)));
        poshidstates = poshidprobs > rand(numcases,numhid);
        
        targets = batchlabel(:,:,batch); 

        bias_pen = repmat(penbiases,numcases,1);
        bias_hid = repmat(2*hidbiases,numcases,1);
        bias_lab = repmat(labbiases,numcases,1);
        
        pospenprobs = 1./(1+exp(-poshidstates*hidpen - targets*labpen - bias_pen));
        posprods    = poshidstates' * pospenprobs;
        posprodslabpen = targets' * pospenprobs;

        pospenact = sum(pospenprobs);
        poshidact = sum(poshidstates);
        poslabact = sum(targets);

        %Start Negative Phase
        negpenprobs = pospenprobs;
        
        for cditer=1:CD
            negpenstates = negpenprobs > rand(numcases,numpen);
            totin = negpenstates*labpen' + bias_lab;
            totin_norm = totin - max(max(totin)); % Preventing overflow
            neglabprobs = exp(totin_norm); % Softmax regression
            neglabprobs = neglabprobs./repmat(sum(neglabprobs,2),1,numclass);
            
            xx = cumsum(neglabprobs,2);
            xx1 = rand(numcases,1);
            neglabstates = neglabprobs*0;
            
            for jj=1:numcases
                index = find(xx1(jj) <= xx(jj,:),1);
                neglabstates(jj,index) = 1;
            end
            
            neghidprobs = 1./(1+exp(- negpenstates*(2*hidpen') - bias_hid));
            neghidstates = neghidprobs > rand(numcases,numhid);
            
            negpenprobs = 1./(1+exp(- neghidstates*hidpen - neglabstates*labpen - bias_pen));
         
        end
        
        negprods = neghidstates'*negpenprobs;
        negprodslabpen = neglabstates'*negpenprobs;
    
        negpenact = sum(negpenprobs);
        neghidact = sum(neghidstates);
        neglabact = sum(neglabstates);
        
        
        %Calculate Error
        err = sum(sum((targets-neglabstates).^2));
        errsum = errsum + err/2;

        if epoch>5
           momentum=finalmomentum;
        else
           momentum=initialmomentum;
        end

        %Update weights
        
        hidpeninc = momentum*hidpeninc + epsilonw*( (posprods-negprods)/numcases - weightcost*hidpen);
        labpeninc = momentum*labpeninc + epsilonw*( (posprodslabpen-negprodslabpen)/numcases - weightcost*labpen); 

        hidbiasinc = momentum*hidbiasinc + (epsilonw/numcases)*(poshidact-neghidact);
        penbiasinc = momentum*penbiasinc + (epsilonw/numcases)*(pospenact-negpenact);
        labbiasinc = momentum*labbiasinc + (epsilonw/numcases)*(poslabact-neglabact);
        
        hidpen = hidpen + hidpeninc;
        labpen = labpen + labpeninc;

        hidbiases = hidbiases + hidbiasinc;
        penbiases = penbiases + penbiasinc;
        labbiases = labbiases + labbiasinc;
    end
    
  % Display error
  
  if rem(epoch,5)==0
      disp(['Number of misclassified training examples: ', num2str(errsum), ' out of ', num2str(numcases*numbatches)]);
  end
  
end

save fullmnistpo labpen labbiases hidpen penbiases hidbiases epoch
