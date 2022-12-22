% mutual information of hidden state mih, label mil
dats = size(mih,1);
numcla = size(mil,2);

P = size(mih,1);
sig = sqrt(10);
ITX = 0;
for matsi = 1:dats
    mihi = (mih - repmat(mih(matsi,:),dats,1)).^2;
    mihi = sum(mihi,2);
    mihi = exp(mihi/(-2*sig^2));
    mihi = sum(mihi,1);
    IX = log(mihi/P);
    ITX = ITX + IX;
end

ITX = -ITX/P;

clas = find(mil' == 1);
clas = clas -1;
r = rem(clas,numcla);
n = fix(clas/numcla)+1;


ALLITY = 0;

for i=1:numcla
    ITY =0;
    mihc = mih(r==(i-1),:);
    Pl = size(mihc,1);
    
    cladat = size(mihc,1);

    if Pl==0
        ITY =ITY;
    else
        for matsi = 1:cladat
            mihi = (mihc - repmat(mihc(matsi,:),cladat,1)).^2;
            mihi = sum(mihi,2);
            mihi = exp(mihi/(-2*sig^2));
            mihi = sum(mihi,1);
            
        
            IY = log(mihi/Pl);
            ITY = ITY + IY;
        end
        ITY = -ITY/Pl;
    end
    
    
    ALLITY = ALLITY - Pl*ITY/P;
end

ITY = ALLITY + ITX;

    
        
