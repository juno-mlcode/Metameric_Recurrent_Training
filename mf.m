
function  [hidprobs, penprobs] = mf(data,targets,vishid,hidbiases,hidpen,penbiases,labpen)

[numhid, numpen] = size(hidpen);
 numcases = size(data,1);
 bias_hid = repmat(hidbiases,numcases,1);
 bias_pen = repmat(penbiases,numcases,1);
  
 hidprobs = 1./(1 + exp(-data*(2*vishid) - 2*bias_hid));
 penprobs = 1./(1 + exp(-hidprobs*hidpen - targets*labpen - bias_pen));

 hidapprox = hidprobs;
 penapprox = penprobs;

 for ii= 1:20 % Number of the mean-field updates.  
   hidprobs = 1./(1+exp(- data*vishid - penapprox*hidpen' - bias_hid));
   penprobs = 1./(1+exp(- hidprobs*hidpen - targets*labpen - bias_pen));

   diff_h1 = sum(sum(abs(hidapprox - hidprobs)))/(numcases*numhid);
   diff_h2 = sum(sum(abs(penapprox - penprobs)))/(numcases*numpen);
   
   if diff_h1 < 0.0000001 && diff_h2 < 0.0000001
        break;
   end
   
   hidapprox = hidprobs;
   penapprox = penprobs;
 end




