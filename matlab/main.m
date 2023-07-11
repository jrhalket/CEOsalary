myStream = RandStream('mlfg6331_64');
RandStream.setGlobalStream(myStream)

Compparams.n_firms=length(FullData.up_data_obs(:,1)); 
Compparams.n_sim=10;
Compparams.trim_percent=0; 
Compparams.hmethod=4; 
Compparams.nparams=18; 
Compparams.logcompdum=0; 
Compparams.dist='Normal';

%hbcv2 = GetH_BCV2(FullData,Compparams.n_firms,Compparams.n_sim,3);
%Use Silverman's Rule of thumb
h(1) = silvermanRoTBand(FullData.down_data_obs(:,1));
h(2) = silvermanRoTBand(FullData.wages_obs(:));
h(3) = silvermanRoTBand(FullData.measures_obs(:,1))

Initparams = Compparams;
Initparams.n_firms = length(InitData.up_data_obs(:,1));

igrid = 1:Compparams.nparams;

lb = [];
ub = [];

b_init = .1*ones(Compparams.nparams);
b_init(16)=log(mean(InitData.wages_obs));
b_init(17)=0.01;
b_init(18)=0.01;


for i=  1:Compparams.nparams
    i_cal = (i+1):Compparams.nparams;
    if i==Compparams.nparams
        i_cal = [];
    end
    b_cal = b_init(i_cal);
    b_est0 = b_init(1:i);
    [lb(i),ub(i)]=returnBounds(i);
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    fun = @(b_est)(-loglikepr(b_est, b_cal, i_cal, InitData,h,Initparams)); 
    b_init(1:i) = fmincon(fun,b_est0,A,b,Aeq,beq,lb,ub) ;

end
