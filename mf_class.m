function  [hidprobs, penprobs] = mf_class(data,vishid,hidbiases,hidpen,penbiases)
  
[numhid, numpen] = size(hidpen);
numcases = size(data,1);
bias_hid = repmat(hidbiases,numcases,1);
bias_pen = repmat(penbiases,numcases,1);

hidprobs = 1./(1 + exp(-data*(2*vishid) - 2*bias_hid));
penprobs = 1./(1 + exp(-hidprobs*hidpen - bias_pen));

hidapprox = hidprobs;
penapprox = penprobs;

 for ii= 1:50 
  hidprobs = 1./(1+exp(- data*vishid - penapprox*hidpen' - bias_hid));
  penprobs = 1./(1+exp(- hidprobs*hidpen - bias_pen));

  diff_h1 = sum(sum(abs(hidapprox - hidprobs)))/(numcases*numhid);
  diff_h2 = sum(sum(abs(penapprox - penprobs)))/(numcases*numpen);
 
  if diff_h1 < 0.0000001 && diff_h2 < 0.0000001
        break;
  end
  hidapprox = hidprobs;
  penapprox = penprobs;

 end
