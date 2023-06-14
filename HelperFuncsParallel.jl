@everywhere function StrucParams(b::Vector{Float64})
     
        #Set up thetas
      
        theta_ge_1 = copy([b[1:3]'; b[4:6]'; b[7:9]'])
        theta_ge_2 = copy([b[10:12]'; b[13:15]'; b[16:18]'])
        theta_ce = copy([b[19:21]'; b[22:24]'; b[25:27]'])
        theta_ga_1 = copy([b[28:30]'; b[31:33]'; b[34:36]'])
        theta_ga_2 = copy([b[37:39]'; b[40:42]'; b[43:45]'])
        theta_ca_1 = copy([b[46:48]'; b[49:51]'; b[52:54]'])
        theta_ca_2 = copy([b[55:57]'; b[58:60]'; b[61:63]'])
        
        #set up Cov matrix for shocks to measures
    
        sigma_1 = exp(b[64])
        sigma_2 = exp(b[66])
        cov12 = -sqrt(sigma_1)sqrt(sigma_2) + 2.0*sqrt(sigma_1)sqrt(sigma_2)*(exp(b[65]))/(1.0+exp(b[65])) #ensuring pos semi def cov matrix
    
        Σ = copy([sigma_1 cov12;
              cov12 sigma_2])
    
        mean_price = exp(copy(b[67]))
    
        rm = copy(b[68:70])
        rf = copy(b[71:73])
    
        risk = copy(exp(b[74]))  #exp to keep risk coefficient positive

        zeta_1 = copy([b[75:77]'; b[78:80]'; b[81:83]'])
        zeta_2 = copy([b[84:86]'; b[87:89]'; b[90:92]'])
 
    p = (
      
        theta_ge_1 = theta_ge_1,
        theta_ge_2 = theta_ge_2,
        theta_ce = theta_ce,
        theta_ga_1 = theta_ga_1,
        theta_ga_2 = theta_ga_2,
        theta_ca_1 = theta_ca_1,
        theta_ca_2 = theta_ca_2,
        
        Σ = Symmetric(Σ),
    
        mean_price = mean_price,
    
        rm = rm,
        rf = rf,
    
        risk = risk,
        
        zeta_1 = zeta_1,
        zeta_2 = zeta_2
        )  
    return p
end

@everywhere function returnBounds(n)
 
    #takes in parameter number (1 to nparams) and returns lower and upper bound
        lowerbnd = -10.0^5;
        upperbnd = 10.0^5;
    
    return lowerbnd, upperbnd
end

@everywhere struct Coefs
    ge_1 :: Float64 
    ge_2 :: Float64
    ce :: Float64
    ga_1 :: Float64
    ga_2 :: Float64
    ca_1 :: Float64
    ca_2 :: Float64
end

@everywhere function makeCoefs(updata,downdata,nfirms,p_in)

    C = Array{Any}(undef,nfirms)
    
    for i in 1:nfirms
        ge_1 = exp.(view(updata,:,i)'*p_in.theta_ge_1*view(downdata,:,i));
        ge_2 = exp.(view(updata,:,i)'*p_in.theta_ge_2*view(downdata,:,i));
        ce = exp.(view(updata,:,i)'*p_in.theta_ce*view(downdata,:,i));
        ga_1 = exp.(view(updata,:,i)'*p_in.theta_ga_1*view(downdata,:,i));
        ga_2 = exp.(view(updata,:,i)'*p_in.theta_ga_2*view(downdata,:,i));
        ca_1 = exp.(view(updata,:,i)'*p_in.theta_ca_1*view(downdata,:,i));
        ca_2 = exp.(view(updata,:,i)'*p_in.theta_ca_2*view(downdata,:,i));
        C[i] = Coefs(copy(ge_1),copy(ge_2),copy(ce),copy(ga_1),copy(ga_2),copy(ca_1),copy(ca_2))
    end
    
    return C
end


@everywhere function ScalcAll(up::SVector{3,Float64},dn::SVector{3,Float64},rΣ,parm::NamedTuple)
    
    ats = SVector{2, Float64}
    
    γ_e_1 = exp.(up'*parm.theta_ge_1*dn);
    γ_e_2 = exp.(up'*parm.theta_ge_2*dn);
    ce = exp.(up'*parm.theta_ce*dn);
    γ_a_1 = exp.(up'*parm.theta_ga_1*dn);
    γ_a_2 = exp.(up'*parm.theta_ga_2*dn);
    ca_1 = exp.(up'*parm.theta_ca_1*dn);
    ca_2 = exp.(up'*parm.theta_ca_2*dn);
 
    ats = [γ_e_1/ce + γ_a_1/ca_1;   γ_e_2/ce + γ_a_2/ca_2];
    T = eltype(rΣ) # make sure it's the type of the output
    F = lu([γ_e_1/ce; γ_e_2/ce]*[γ_e_1/ce; γ_e_2/ce]' +Diagonal(SA[(γ_a_1)^2/ca_1, (γ_a_2)^2/ca_2]) + rΣ, check = false) # avoids the error
    issuccess(F) || return T(Inf) # assuming minimization, this step will be rejected
    ret = 0.5 * ats' * (F \ ats)
    return ret[1]
end   

@everywhere struct SimOutputs
    down_match :: Array{Float64,2} 
    wages_match :: Array{Float64,1} 
    measures_match :: Array{Float64,2}
end


 
@everywhere function ComputeSMatrix(params,up_data_obs_in,down_data_obs_in,p_in::NamedTuple)
    CR = params;
    rΣ = Matrix{Float64}(undef,2,2)
    rΣ = p_in.risk * p_in.Σ
    S_sim = zeros(CR.n_firms,CR.n_firms);
    for n2 in 1:CR.n_firms;
        for n1 in 1:CR.n_firms;
            S_sim[n1,n2] = ScalcAll(SVector{3, Float64}(view(up_data_obs_in,:,n1)),SVector{3, Float64}(view(down_data_obs_in,:,n2)),rΣ,p_in)
        end
    end
    return S_sim

end

function bcv2_fun(h,Data,n_firms,n_sims,logcompdum)
    h=abs.(h)
    local N = n_firms*n_sims;
    ll = 0.0
    for i = 1:n_firms
        for j=1:n_firms
            if (j!=i)
                if logcompdum ==0
                    expr_1 = ((Data.down_data_obs[i,1]-Data.down_data_obs[j,1])/h[1])^2 + 
                         ((Data.down_data_obs[i,2]-Data.down_data_obs[j,2])/h[2])^2 + 
                         ((Data.wages_obs[i]-Data.wages_obs[j])/h[3])^2 +
                         ((Data.measures_obs[i,1]-Data.measures_obs[j,1])/h[4])^2 +
                         ((Data.measures_obs[i,2]-Data.measures_obs[j,2])/h[5])^2
                    expr_2 =  pdf(Normal(),(Data.down_data_obs[i,1]-Data.down_data_obs[j,1])/h[1]) * 
                          pdf(Normal(),(Data.down_data_obs[i,2]-Data.down_data_obs[j,2])/h[2]) * 
                          pdf(Normal(),(Data.wages_obs[i]-Data.wages_obs[j])/h[3]) *
                          pdf(Normal(),(Data.measures_obs[i,1]-Data.measures_obs[j,1])/h[4]) *
                          pdf(Normal(),(Data.measures_obs[i,2]-Data.measures_obs[j,2])/h[5])
                else
                    expr_1 = ((Data.down_data_obs[i,1]-Data.down_data_obs[j,1])/h[1])^2 + 
                         ((Data.down_data_obs[i,2]-Data.down_data_obs[j,2])/h[2])^2 + 
                         (((Data.wages_obs[i]-Data.wages_obs[j])/Data.wages_obs[i])/h[3])^2 +
                        ((Data.measures_obs[i,1]-Data.measures_obs[j,1])/h[4])^2 +
                        ((Data.measures_obs[i,2]-Data.measures_obs[j,2])/h[5])^2
                     expr_2 =  pdf(Normal(),(Data.down_data_obs[i,1]-Data.down_data_obs[j,1])/h[1]) * 
                        pdf(Normal(),(Data.down_data_obs[i,2]-Data.down_data_obs[j,2])/h[2]) * 
                        pdf(Normal(),((Data.wages_obs[i]-Data.wages_obs[j])/Data.wages_obs[i])/h[3]) *
                        pdf(Normal(),(Data.measures_obs[i,1]-Data.measures_obs[j,1])/h[4]) *
                        pdf(Normal(),(Data.measures_obs[i,2]-Data.measures_obs[j,2])/h[5])
                end
                ll += (expr_1 - (2*5 +4)*expr_1 + (5^2 +2*5))*expr_2
            end
        end
    end
    val = ((sqrt(2*pi))^5 * N *h[1]*h[2]*h[3]*h[4]*h[5])^(-1) +
                            ((4*N*(N-1))*h[1]*h[2]*h[3]*h[4]*h[5])^(-1) * ll
    println("band: ",h," val: ", val)
    return val
end

