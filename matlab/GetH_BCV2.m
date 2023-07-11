function h = GetH_BCV2(Data,n_firms,n_sims,nh)
    % Optimize over choice of h
    
    lb=zeros(1,nh);
    ub = [];
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    fun = @(h)bcv2_fun(h,Data,n_firms,n_sims);
    h = fmincon(fun,rand(1,nh),A,b,Aeq,beq,lb,ub) ;
end
