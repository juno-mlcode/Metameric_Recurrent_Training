%Layerwise Pretraining - not contain output layer

load fullmnistvh.mat

%% Initialize learning rate, momentum 
epsilonw      = 0.05;   % Learning rate for weights 
CD=1;  % number of steps in constrastive divergence
weightcost  = 0.001;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;

[numcases,numdims,numbatches]=size(batchdata);

disp('start pre-training');

%% Start learning
for epoch = 1:maxepoch
    
    if rem(epoch,10)==0
        disp(['pre-training epoch:',num2str(epoch)]);
    end
    
    for batch = 1:numbatches
        
        visbias = repmat(visbiases,numcases,1);
        hidbias = repmat(2*hidbiases,numcases,1); 
        
        %Start Positive Phase
        
        data = batchdata(:,:,batch);
        data = data > rand(numcases,numdims);
        poshidprobs = 1./(1 + exp(-data*(2*vishid) - hidbias));    
        posprods    = data' * poshidprobs;
        poshidact   = sum(poshidprobs);
        posvisact   = sum(data);
        
        %Start Negative Phase
        
        poshidstates = poshidprobs > rand(numcases,numhid);
        negdata = 1./(1 + exp(-poshidstates*vishid' - visbias));
        negdata = negdata > rand(numcases,numdims); 
        neghidprobs = 1./(1 + exp(-negdata*(2*vishid) - hidbias));

        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata); 

        %End Negative Phase
        
       if epoch>5
            momentum=finalmomentum;
       else
            momentum=initialmomentum;
       end

       %Update weights

       vishidinc  = momentum*vishidinc + epsilonw*((posprods-negprods)/numcases - weightcost*vishid);
       visbiasinc = momentum*visbiasinc + (epsilonw/numcases)*(posvisact-negvisact);
       hidbiasinc = momentum*hidbiasinc + (epsilonw/numcases)*(poshidact-neghidact);

       vishid    = vishid + vishidinc;
       visbiases = visbiases + visbiasinc;
       hidbiases = hidbiases + hidbiasinc;

    end
end

save fullmnistvh vishid visbiases hidbiases epoch 

