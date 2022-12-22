function [f, df] = ECG1(VV,Dim,XX,target,temp_h2)

numdims = Dim(1); 
numdims_2=Dim(2);
numhid = Dim(3);
numpen = Dim(4); 
numclass = Dim(5);
N = size(XX,1);

X=VV;
% Do decomversion.
 w1_vishid = reshape(X(1:numdims*numhid),numdims,numhid);
 xxx = numdims*numhid;
w1_penhid = reshape(X(xxx+1:xxx+numdims_2*numhid),numdims_2,numhid);
 xxx = xxx+numdims_2*numhid;
 hidpen = reshape(X(xxx+1:xxx+numhid*numpen),numhid,numpen);
 xxx = xxx+numhid*numpen;
 w_class = reshape(X(xxx+1:xxx+numpen*numclass),numpen,numclass);
 xxx = xxx+numpen*numclass;
 hidbiases = reshape(X(xxx+1:xxx+numhid),1,numhid);
 xxx = xxx+numhid;
 penbiases = reshape(X(xxx+1:xxx+numpen),1,numpen);
 xxx = xxx+numpen;
 topbiases = reshape(X(xxx+1:xxx+numclass),1,numclass);
 xxx = xxx+numclass;

  bias_hid= repmat(hidbiases,N,1);
  bias_pen = repmat(penbiases,N,1);
  bias_top = repmat(topbiases,N,1);

  w1probs = 1./(1 + exp(-XX*w1_vishid -temp_h2*w1_penhid - bias_hid  ));
  w2probs = 1./(1 + exp(-w1probs*hidpen - bias_pen));
  targetout = exp(w2probs*w_class + bias_top );
  targetout = targetout./repmat(sum(targetout,2),1,numclass);

  f = -sum(sum( target(:,1:end).*log(targetout)));

 IO = (targetout-target(:,1:end));
 Ix_class=IO; 
 dw_class =  w2probs'*Ix_class;
 dtopbiases = sum(Ix_class);

 Ix2 = (Ix_class*w_class').*w2probs.*(1-w2probs);
 dw2_hidpen =  w1probs'*Ix2;
 dw2_biases = sum(Ix2); 

 Ix1 = (Ix2*hidpen').*w1probs.*(1-w1probs); 
 dw1_penhid =  temp_h2'*Ix1;

 dw1_vishid = XX'*Ix1;
 dw1_biases = sum(Ix1);

 df = [dw1_vishid(:)' dw1_penhid(:)' dw2_hidpen(:)' dw_class(:)' dw1_biases(:)' dw2_biases(:)' dtopbiases(:)']'; 

