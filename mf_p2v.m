function  [hidprobs, visprobs] = mf_p2v(penstates,hidpen,hidbiases,vishid,visbiases)

[numdims, numhid]=size(vishid);
numcases = size(penstates,1);
bias_hid = repmat(hidbiases,numcases,1);
bias_vis = repmat(visbiases,numcases,1);

hidprobs = 1./(1 + exp(-penstates*(2*hidpen') - bias_hid));
visprobs = 1./(1 + exp(-hidprobs*vishid' - bias_vis));

hidapprox = hidprobs;
visapprox = visprobs;

 for ii= 1:50 
  hidprobs = 1./(1 + exp(-penstates*hidpen' - visapprox*vishid - bias_hid));
  visprobs = 1./(1 + exp(-hidprobs*vishid' - bias_vis));

  diff_h1 = sum(sum(abs(hidapprox - hidprobs)))/(numcases*numhid);
  diff_h2 = sum(sum(abs(visapprox - visprobs)))/(numcases*numdims);
 
  if diff_h1 < 0.0000001 && diff_h2 < 0.0000001
        break;
  end
  
  hidapprox = hidprobs;
  visapprox = visprobs;
 end
