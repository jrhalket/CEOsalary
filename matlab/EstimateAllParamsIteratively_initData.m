igrid = 1:Compparams.nparams;

lb = [];
ub = [];

b_init = .1*ones(Compparams.nparams,1);
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
    %b_init(1:i,1) = fmincon(fun,b_est0,A,b,Aeq,beq,lb,ub) ;
    optimset('MaxFunEvals',10000*i)
    b_init(1:i,1) = fminsearch(fun,b_est0) ;
end