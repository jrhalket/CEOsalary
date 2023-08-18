Initparams.n_sim=500;
Initparams.dist = 'Cauchy';
Initparams.trim_percent=0; 

igrid = 1:Compparams.nparams;
b_true = b_init;
b_true(1) = -1;
b_true(2) = -1;
b_true(3) = -1;
b_true(4) = -1;
b_true(6) = 0.5;
b_true(7) = 1;
b_true(17) = -1;
b_true(18) = 1;
b_true(8) = -9;
b_true(16) = 10;
bgrid = -1:.02:1;
llgridTrim = zeros(Compparams.nparams,length(bgrid));
for i = 1:Compparams.nparams
    ical = igrid;
    ical(i)=[];
    b_cal = b_true(ical);
    for j = 1:length(bgrid)
        b_est = b_true(i)+bgrid(j);
        llgridTrim(i,j) = loglikepr(b_est,b_cal,ical,InitData,h, Initparams);
    end
end

