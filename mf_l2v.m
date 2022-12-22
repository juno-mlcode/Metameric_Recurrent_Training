function  [temp_h2, temp_h1, temp_vis] = mf_l2v(target,labpen,penbiases,hidpen,hidbiases,vishid,visbiases)

  numcases = size(target,1);
  [numhid, numpen] = size(hidpen);
  numdims = size(vishid,1);
  bias_hid= repmat(hidbiases,numcases,1);
  bias_pen = repmat(penbiases,numcases,1);
  bias_vis=repmat(visbiases,numcases,1);
  
 temp_h2 = 1./(1 + exp(-target*(2*labpen) - bias_pen));
 temp_h1 = 1./(1 + exp(-temp_h2*(2*hidpen') -bias_hid));
 temp_vis = 1./(1+ exp(-temp_h1* vishid' - bias_vis));
 
for ii= 1:50 
   temp_h1_new = 1./(1 + exp(-temp_vis*vishid-temp_h2*hidpen'-bias_hid));
   temp_h2_new = 1./(1 + exp(-target*labpen-temp_h1_new*hidpen-bias_pen));
   temp_vis_new= 1./(1+ exp(-temp_h1_new* vishid' - bias_vis));
   
   diff_h1 = sum(sum(abs(temp_h1_new - temp_h1),2))/(numcases*numhid);
   diff_h2 = sum(sum(abs(temp_h2_new - temp_h2),2))/(numcases*numpen);
   diff_vis = sum(sum(abs(temp_vis_new - temp_vis),2))/(numcases*numdims);
   
   if (diff_h1 < 0.0000001 & diff_h2 < 0.0000001 & diff_vis < 0.0000001)
       break;
   end
   
   temp_h1 = temp_h1_new;
   temp_h2 = temp_h2_new;
   temp_vis= temp_vis_new;
 end

   temp_h1 = temp_h1_new;
   temp_h2 = temp_h2_new;
   temp_vis = temp_vis_new;
