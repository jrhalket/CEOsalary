igrid = 1:Compparams.nparams;
Initparams.n_sim=100;
Initparams.dist = 'Cauchy';
lb = [];
ub = [];


b_true = .1*rand(Compparams.nparams,1);
b_true(17)=log(mean(InitData.wages_obs));
b_true(10)=-1;
b_true(11)=0.05;
%b_true(8) = -9;
%b_true(12:18)=0;
params_true = StrucParams(b_true);
[SimRaw.up_data_obs, SimRaw.down_data_obs, SimRaw.wages_obs, SimRaw.measures_obs]=SimData(params_true,i,Initparams) ;


for i=  1:Compparams.nparams-7
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
    fun = @(b_est)(-loglikepr(b_est, b_cal, i_cal, SimRaw,h,Initparams)); 
    %b_init(1:i,1) = fmincon(fun,b_est0,A,b,Aeq,beq,lb,ub) ;
    optimset('MaxFunEvals',10000000*i)
    b_init(1:i) = fminsearch(fun,b_est0) ;
    b_estOut(i,:) = b_init';
end