genum = gen_num; %초기 데이터 개수의 배수로 설정
datanum = size(fewdata,1);

bias_vis = repmat(visbiases,datanum,1);
bias_pen = repmat(penbiases,datanum,1);

%hidden state
[poshidprobs, pospenprobs] = mf(fewdata,fewlabel,vishid,hidbiases,hidpen,penbiases,labpen);
neghidstates = poshidprobs > rand(datanum,numhid);
negpenstates = pospenprobs > rand(datanum,numpen);

newdata=zeros(genum,numdims);

for i=1:9
    ind = find(trainlabel(:,i+1)==1);
    scatdat = traindata(ind(1:100),:);
    scatlab = trainlabel(ind(1:100),:);
    [poshidprobs, pospenprobs] = mf(scatdat,scatlab,vishid,hidbiases,hidpen,penbiases,labpen);
    scatter3(poshidprobs(:,1),poshidprobs(:,2),poshidprobs(:,3),'o');
    %scatter3(poshidprobs(:,1),poshidprobs(:,2),poshidprobs(:,3),'o','MarkerFaceColor', [(i-1)/9, (9-i)/9, 1]);
end

for i=1:genum/datanum
    
    %% lab to pen
    %[negpenprobs, ~, ~]=mf_l2v(neglabstates,labpen,penbiases,hidpen,hidbiases,vishid,visbiases);
    %negpenstates = negpenprobs > rand(datanum, numpen);
    
%     if feedback > 3
%     %% pen to hid
%     [neghidprobs, negdataprobs] = mf_p2v(negpenstates,hidpen,hidbiases,vishid,visbiases);
%     %neghidstates = neghidprobs > rand(datanum,numhid);
%     end
    
    %% hid to vis
   
    negdataprobs = 1./(1+exp(- neghidstates*vishid' - bias_vis));
    negdata = negdataprobs > rand(datanum,numdims);
    newdata((i-1)*datanum+1:i*datanum,:) = negdata;
end

%% find more different data
newtarget=repmat(fewlabel,genum/datanum,1);
traindata = [newdata;traindata];
trainlabel = [newtarget;trainlabel];

if feedback < 2
    [mih, mip] = mf(fewdata,fewlabel,vishid, hidbiases, hidpen, penbiases,labpen);
    mil = fewlabel;
    mutual_information;
    I_tr(1,1) = ITX;
    
    mih = mip;
    mutual_information;
    I_tr(2,1) = ITX;
    
    [mih mip] = mf([fewdata;newdata],[fewlabel;newtarget],vishid,hidbiases,hidpen,penbiases,labpen);
    mil = [fewlabel;newtarget];
    mutual_information;
    I_ge(1,1) = ITX;
    
    mih = mip;
    mutual_information;
    I_ge(2,1) = ITX;
end

[mih, ~] = mf(fewdata,fewlabel,vishid, hidbiases, hidpen, penbiases,labpen);

dat = mut_test_x;
entropy;
en_x = HX;
[poshidprobs, pospenprobs] = mf(fewdata,fewlabel,vishid,hidbiases,hidpen,penbiases,labpen);
neghidstates = poshidprobs > rand(datanum,numhid);

mut_z = zeros(1000,numdims);
for i=1:1000/datanum
    %% hid to vis
   
    negdataprobs = 1./(1+exp(- neghidstates*vishid' - bias_vis));
    negdata = negdataprobs > rand(datanum,numdims);
    mut_z((i-1)*datanum+1:i*datanum,:) = negdata;
end

[mih, pospenprobs] = mf(newmutset,newmutlab,vishid,hidbiases,hidpen,penbiases,labpen);
mil = newmutlab;
mutual_information;
I_n(feedback,1) = ITX;



dat = mut_z;
entropy;
en_z = HX;

bias_vis = repmat(visbiases,1000,1);
bias_pen = repmat(penbiases,1000,1);

[poshidprobs, pospenprobs] = mf(mut_test_w,mut_test_l,vishid,hidbiases,hidpen,penbiases,labpen);
neghidstates = poshidprobs > rand(1000,numhid);
negdataprobs = 1./(1+exp(- neghidstates*vishid' - bias_vis));
negdata = negdataprobs > rand(1000,numdims);
mut_w = negdata;

dat = mut_w;
entropy;
en_w = HX;

dat = [mut_test_x,mut_z];
entropy;
H_xz = HX;

dat = [mut_test_x,mut_w];
entropy;
H_xw = HX;

I_xz = en_x+en_z - H_xz;
I_xw = en_x+en_w - H_xw;

Uxn = 1-I_xz/H_xz;
Ux = 1-I_xw/H_xw;
%en_x, en_z, en_w, I_xz, I_xw


%[poshidprobs, pospenprobs] = mf(fewdata,fewlabel,vishid,hidbiases,hidpen,penbiases,labpen);
%befpca = poshidprobs;
%for i=1:2
%    ind = find(trainlabel(:,i+1)==1);
%    scatdat = traindata(ind(1:100),:);
%    scatlab = trainlabel(ind(1:100),:);
%    [poshidprobs, pospenprobs] = mf(scatdat,scatlab,vishid,hidbiases,hidpen,penbiases,labpen);
%befpca = [befpca;poshidprobs];
%end

%coeff = pca(befpca);
%aftpca = befpca*coeff(:, 1:3);

%scatter3(aftpca(1:10,1),aftpca(1:10,2),aftpca(1:10,3),'o','MarkerFaceColor', [1, 1, 1]);hold on;
%for i=1:9
%    scatter3(aftpca(11+(i-1)*100:10+i*100,1),aftpca(11+(i-1)*100:10+i*100,2),aftpca(11+(i-1)*100:10+i*100,3),'o','MarkerFaceColor', [(i-1)/9, (9-i)/9, 1]);
%end
