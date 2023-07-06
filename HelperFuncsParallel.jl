@everywhere function StrucParams(b::Vector{Float64})
     
        #Set up thetas
      
        theta_ga_1 = copy([b[1:2]'; b[3:4]'])
        theta_ca_1 = copy([b[5:6]'; b[7:8]'])
        
        #set up Cov matrix for shocks to measures
    
        sigma_1 = exp(b[9])
        
        Σ = copy(sigma_1)
    
        mean_price = exp(copy(b[10]))
    
        rm = copy(b[11:12])
        rf = copy([b[13]; 0.0])
    
        risk = copy(exp(b[14]))  #exp to keep risk coefficient positive

        zeta_1 = copy([b[15:16]'; b[17:18]'])
     
    p = (
        theta_ga_1 = theta_ga_1,
        theta_ca_1 = theta_ca_1,
        
        Σ = Σ,
    
        mean_price = mean_price,
    
        rm = rm,
        rf = rf,
    
        risk = risk,
        
        zeta_1 = zeta_1,
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
    ga_1 :: Float64
    ca_1 :: Float64
end

@everywhere function makeCoefs(updata,downdata,nfirms,p_in)

    C = Array{Any}(undef,nfirms)
    
    for i in 1:nfirms
        ga_1 = exp(view(updata,:,i)'*p_in.theta_ga_1*view(downdata,:,i));
        ca_1 = exp(view(updata,:,i)'*p_in.theta_ca_1*view(downdata,:,i));
        C[i] = Coefs(copy(ga_1),copy(ca_1))
    end
    
    return C
end


@everywhere function ScalcAll(up::SVector{2,Float64},dn::SVector{2,Float64},rΣ,parm::NamedTuple)
    
    ats = SVector{2, Float64}
    
    γa1 = exp(up'*parm.theta_ga_1*dn);
    ca_1 = exp(up'*parm.theta_ca_1*dn);
 
    ats = γa1/ca_1
    F= (γa1).^2 / ca_1 + rΣ
    #println(γa1, " ", ca_1, " ", rΣ," ", F)
    ret = 0.5 * ats*ats/F
    return ret
end   

@everywhere struct SimOutputs
    down_match :: Array{Float64,2} 
    wages_match :: Array{Float64,1} 
    measures_match :: Array{Float64,1}
end


 
@everywhere function ComputeSMatrix(params,up_data_obs_in,down_data_obs_in,p_in::NamedTuple)
    CR = params;
    #rΣ = Matrix{Float64}(undef,1,1)
    rΣ = p_in.risk * p_in.Σ
    S_sim = zeros(CR.n_firms,CR.n_firms);
    for n2 in 1:CR.n_firms;
        for n1 in 1:CR.n_firms;
            S_sim[n1,n2] = ScalcAll(SVector{2, Float64}(view(up_data_obs_in,:,n1)),SVector{2, Float64}(view(down_data_obs_in,:,n2)),rΣ,p_in)
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
                         ((Data.wages_obs[i]-Data.wages_obs[j])/h[2])^2 +
                         ((Data.measures_obs[i,1]-Data.measures_obs[j,1])/h[3])^2 ;
                    expr_2 =  pdf(Normal(),(Data.down_data_obs[i,1]-Data.down_data_obs[j,1])/h[1]) * 
                          pdf(Normal(),(Data.wages_obs[i]-Data.wages_obs[j])/h[2]) *
                          pdf(Normal(),(Data.measures_obs[i,1]-Data.measures_obs[j,1])/h[3]) 
                elseif logcompdum ==1
                    expr_1 = ((Data.down_data_obs[i,1]-Data.down_data_obs[j,1])/h[1])^2 + 
                         (((Data.wages_obs[i]-Data.wages_obs[j])/Data.wages_obs[i])/h[2])^2 +
                        ((Data.measures_obs[i,1]-Data.measures_obs[j,1])/h[3])^2 
                    expr_2 =  pdf(Normal(),(Data.down_data_obs[i,1]-Data.down_data_obs[j,1])/h[1]) * 
                        pdf(Normal(),((Data.wages_obs[i]-Data.wages_obs[j])/Data.wages_obs[i])/h[2]) *
                        pdf(Normal(),(Data.measures_obs[i,1]-Data.measures_obs[j,1])/h[3]) 
                else 
                    expr_1 = ((Data.down_data_obs[i,1]-Data.down_data_obs[j,1])/h[1])^2 + 
                             ((log(Data.wages_obs[i])-log(Data.wages_obs[j]))/h[2])^2 +
                            ((Data.measures_obs[i,1]-Data.measures_obs[j,1])/h[3])^2 
                    expr_2 =  pdf(Normal(),(Data.down_data_obs[i,1]-Data.down_data_obs[j,1])/h[1]) * 
                            pdf(Normal(),((log(Data.wages_obs[i])-log(Data.wages_obs[j]))/Data.wages_obs[i])/h[2]) *
                            pdf(Normal(),(Data.measures_obs[i,1]-Data.measures_obs[j,1])/h[3]) 
                    end
                ll += (expr_1 - (2*3 +4)*expr_1 + (3^2 +2*3))*expr_2
            end
        end
    end
    val = ((sqrt(2*pi))^3 * N *h[1]*h[2]*h[3])^(-1) +
                            ((4*N*(N-1))*h[1]*h[2]*h[3])^(-1) * ll
    println("band: ",h," val: ", val)
    return val
end

