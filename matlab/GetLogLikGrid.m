igrid = 1:Compparams.nparams;
b_true = b_init;
bgrid = -1:.02:1;
llgrid = zeros(Compparams.nparams,length(bgrid));
for i = 1:Compparams.nparams
    ical = igrid;
    ical(i)=[];
    b_cal = b_true(ical);
    for j = 1:length(bgrid)
        b_est = b_true(i)+bgrid(j);
        llgrid(i,j) = loglikepr(b_est,b_cal,ical,InitData,h, Initparams);
    end
end

