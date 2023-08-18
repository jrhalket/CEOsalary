Initparams.n_sim=50;
Initparams.dist = 'Cauchy';
Initparams.trim_percent=0; 

igrid = 1:Compparams.nparams;
b_true = rand(Compparams.nparams,1);
b_true(17)=-1;
b_true(10)=-1;
b_true(11)=0.05;
b_true(12:15)=0;
global ParamScale 
ParamScale = mean(InitData.wages_obs);
%b_true(4) = log(ParamScale);
b_true(8) = -log(ParamScale);
params_true = StrucParams(b_true);
[SimRaw.up_data_obs, SimRaw.down_data_obs, SimRaw.wages_obs, SimRaw.measures_obs]=SimData(params_true,i,Initparams) ;

h(1) = silvermanRoTBand(SimRaw.down_data_obs(:,1));
h(2) = silvermanRoTBand(SimRaw.wages_obs(:));
h(3) = silvermanRoTBand(SimRaw.measures_obs(:,1));

bgrid = -1:.02:1;
llgridTrim = zeros(Compparams.nparams,length(bgrid));
for i = 1:Compparams.nparams
    ical = igrid;
    ical(i)=[];
    b_cal = b_true(ical);
    for j = 1:length(bgrid)
        b_est = b_true(i)*exp(bgrid(j));
        llgridTrim(i,j) = loglikepr(b_est,b_cal,ical, SimRaw,h,Initparams);
    end
end

