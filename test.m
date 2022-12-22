%% Data preprocessing
mutualnum = 1; %mutualsetnum for each class
fewdatnum = 10; %set fewdatnum < particular class data number
numfid = 3; %number of recurrent training
gen_num =1500; % generation sample in 1 training

mnistdata;

numdims = size(traindata,2);
numclass = size(trainlabel,2);

numhid = 100;numpen = 100;

nummf = 30; 
preepoch = 50; mlpmaxepoch = 100; dbmmaxepoch = 100;

mnistdata;

numdims = size(traindata,2);
numclass = size(trainlabel,2);

acc = zeros(2,numfid);
I_all_X = zeros(2,2);
I_all_Y = zeros(2,2);
I_tr = zeros(2,1);
I_ge = zeros(2,1);
%% make minibatches from training data
feedback =1;
numcases = 100; % can be modified w.r.t data number.
%% Layerwise Pre-training

%First layer
maxepoch = preepoch; % num of epoch
makebatches;
rbm;

% Last layer
maxepoch = preepoch; % num of epoch
makebatches;
rbm_last;
disp(['Pretraining with ', num2str(numhid), ' hid units and ', num2str(numpen), ' pen units completed' ])

%% Deep learning
maxepoch = dbmmaxepoch;
makebatches;
dbm;

% Set dropout rate
dr_vis = 0;
dr = 0;
p_vis = 1-dr_vis;
p = 1-dr;
%% Fine-tuning with backpropagation
maxepoch= mlpmaxepoch;
makebatches; 
backprop;
% 
% generation.0g
gene_all;
%     traindata = [traindata;repmat(fewdata, gen_num/fewdatnum,1)];
%     trainlabel = [trainlabel;repmat(fewlabel,gen_num/fewdatnum,1)];

for feedback=2:numfid
        %First layer
    maxepoch = preepoch; % num of epoch
    makebatches;
    rbm_test;

    % Last layer
    maxepoch = preepoch; % num of epoch
    makebatches;
    rbm_last_test;
    disp(['Pretraining with ', num2str(numhid), ' hid units and ', num2str(numpen), ' pen units completed' ])

    %% Deep learning
    maxepoch = dbmmaxepoch;
    makebatches;
    dbm;

    % Set dropout rate
    dr_vis = 0;
    dr = 0;
    p_vis = 1-dr_vis;
    p = 1-dr;
    %% Fine-tuning with backpropagation
    maxepoch= mlpmaxepoch;
    makebatches; 
    backprop_test;
    % 
    % generation.0g
    gene_all;
    %     traindata = [traindata;repmat(fewdata, gen_num/fewdatnum,1)];
    %     trainlabel = [trainlabel;repmat(fewlabel,gen_num/fewdatnum,1)];
   
end






