10.5281/zenodo.7500710
# Metameric_Recurrent_Training

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.2019-0-01371, Development of brain-inspired AI with human-like intelligence).

#How to use

1. Install Matlab >= 2020 version
2. Open the main.m file
3. Modify following values

numfid : number of iterations of recurrent training

gen_num : number of generated data in one iteration

numhid, numpen : number of node in Deep Boltzmann Machines

preepoch, mlpmaxepoch, dbmmaxepoch : max epoch of each training stage
*****
#brief introduce of main files

main.m - training and test suggested model

rbm.m - declaration of restricted boltzmann machine

dbm.m - declaration of deep boltzmann machine

backprop.m - backpropagation of model

mnistdata.m - data preprocessing
