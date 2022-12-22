% entropy of dat
dats = size(dat,1);

P = size(dat,1);
sig = sqrt(10);
HX = 0;
for matsi = 1:dats
    mihi = (dat - repmat(dat(matsi,:),dats,1)).^2;
    mihi = sum(mihi,2);
    mihi = exp(mihi/(-2*sig^2));
    mihi = sum(mihi,1);
    IX = log(mihi/P);
    HX = HX + IX;
end

HX = -HX/P;
