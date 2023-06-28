@everywhere function loglikepr(b_est, b_cal, i_cal, Data,h,quasi_seed,LLC;multithread)
    # inputs are:
    # b_est = parameter values of parameters to be estimated
    # b_cal = parameters values of parameters to be calibrated
    # i_cal = index in b for parameters to be calibrated
    # Data.up_data_obs - the observable part of up_data
    # Data.down_data_obs - the obs part of down_data
    # Data.wages_obs - obs wages
    # Data.measures_obs = obs measures
    # SimIn = structure with some allocations
   # println(b_est)
    ic = 1;
    ie = 1;
    nb = length(b_est)+length(b_cal);
    b = zeros(nb);
    for ib in 1:nb
        if ic<=length(i_cal)      
            if ib == i_cal[ic];
                b[ib] = b_cal[ic];
                ic += 1;
            else 
                b[ib] = b_est[ie]; 
                ie += 1;
            end
        else
            b[ib] = b_est[ie];
            ie += 1;
        end
    end

    LLp = StrucParams(b);
    if !isposdef(LLp.Σ)  #check positive def cov matrix
        llnf = -2.0^50
        return llnf
    end

    S_sim = Matrix{Float64}(undef,LLC.n_firms,LLC.n_firms);
    S_sim = ComputeSMatrix(LLC,Data.up_data_obs',Data.down_data_obs',LLp);
    if multithread 
        sim_dat = pmap(i -> sim_data_like(Data.up_data_obs', Data.down_data_obs', LLp, S_sim, i, quasi_seed,LLC), 1:LLC.n_sim, batch_size=cld(LLC.n_sim,nworkers()))
    else 
        sim_dat = map(i -> sim_data_like(Data.up_data_obs', Data.down_data_obs', LLp, S_sim, i, quasi_seed,LLC), 1:LLC.n_sim)
    end
    
    #if any(isnan(sim_dat[1:LLC.n_sim].down_match[1,1]))
    #    llnf = -2.0^50
    #else
            #parallel version: sim_dat = pmap(solve_draw, 1:n_sim)
        #outputs from sim_data_like: up_data, down_match_sim, wages_match, measures_match
        ll=0.0;   
    if (LLC.hmethod<4)
        # Bandwidths for y, wages, measures, cov of measures
        dataWithSim = zeros(LLC.n_firms*LLC.n_sim,5);
        #    data = zeros(LLC.n_firms,5);
        #    data[:,1] = view(down_data_obs,:,1);
        #    data[:,2] = view(down_data_obs,:,2);
        #    data[:,3] = wages_obs;
        #    data[:,4] = view(measures_obs,:,1);
        #    data[:,5] = view(measures_obs,:,2);
        for i in 1:LLC.n_sim
            dataWithSim[1+(i-1)*LLC.n_firms:i*LLC.n_firms,1] = Data.down_data_obs[:,1] .- sim_dat[i].down_match[:,1];
            dataWithSim[1+(i-1)*LLC.n_firms:i*LLC.n_firms,2] = Data.down_data_obs[:,2] .- sim_dat[i].down_match[:,2];
            dataWithSim[1+(i-1)*LLC.n_firms:i*LLC.n_firms,3] = (Data.wages_obs) .- (sim_dat[i].wages_match[:]);
            dataWithSim[1+(i-1)*LLC.n_firms:i*LLC.n_firms,4] = Data.measures_obs[:,1] .- sim_dat[i].measures_match[:,1];
            dataWithSim[1+(i-1)*LLC.n_firms:i*LLC.n_firms,5] = Data.measures_obs[:,2] .- sim_dat[i].measures_match[:,2];
        end
        if (LLC.hmethod==1)
            #Silverman's Rule of thumb:
            Iqrnorm = 1.34;
            A = [min(std(dataWithSim[:,1]),StatsBase.iqr(dataWithSim[:,1])/Iqrnorm),
                min(std(dataWithSim[:,2]),StatsBase.iqr(dataWithSim[:,2])/Iqrnorm),
                min(std(dataWithSim[:,3]),StatsBase.iqr(dataWithSim[:,3])/Iqrnorm),
                min(std(dataWithSim[:,4]),StatsBase.iqr(dataWithSim[:,4])/Iqrnorm),
                min(std(dataWithSim[:,5]),StatsBase.iqr(dataWithSim[:,5])/Iqrnorm)];
            h=0.9 * A * (LLC.n_firms*LLC.n_sim)^(-0.2);
        elseif (LLC.hmethod==2)
            #Using Sheather and Jones (1991) bandwidth
            #h = zeros(5);
            h = [KernelDensitySJ.bwsj(dataWithSim[:,1]), 
                KernelDensitySJ.bwsj(dataWithSim[:,2]),
                KernelDensitySJ.bwsj(dataWithSim[:,3]),
                KernelDensitySJ.bwsj(dataWithSim[:,4]),
                KernelDensitySJ.bwsj(dataWithSim[:,5])];
        elseif (LLC.hmethod==3)
            # Scott(1992)'s rule for normal multivariate densities
            #h_i = (4/(d + 2))^(1/(d+4))  * σ_i  n^(−1/(d+4))  where
            # n = num simulations
            # d = number of variables (i.e. 2 obs match, watch and 2 measures --> d=5)
            # σ_i is the std dev of each variable i 
            h = (4/(5+2)* 1/LLC.n_sim)^(1/(5+4)) .* 
                    [std(dataWithSim[:,1]),
                    std(dataWithSim[:,2]),
                    std(dataWithSim[:,3]),
                    std(dataWithSim[:,4]),
                    std(dataWithSim[:,5])];                                                    
        end
    end
    like = zeros(LLC.n_firms);
    hall = prod(h);
    
    havg = hall^(1/5);
  
    for i in 1:LLC.n_firms
        for j in 1:LLC.n_sim
            if LLC.logcompdum ==0
                like[i]+=exp( 
                logpdf(LLC.dist(),(Data.down_data_obs[i,1] - sim_dat[j].down_match[i,1])/h[1])
                + logpdf(LLC.dist(),(Data.down_data_obs[i,2] - sim_dat[j].down_match[i,2])/h[2])
                + logpdf(LLC.dist(),(Data.wages_obs[i] - sim_dat[j].wages_match[i])/h[3])
                + logpdf(LLC.dist(),(Data.measures_obs[i,1] - sim_dat[j].measures_match[i,1])/h[4])
                + logpdf(LLC.dist(),(Data.measures_obs[i,2] - sim_dat[j].measures_match[i,2])/h[5])
                )
            else 
                like[i]+=exp( 
                logpdf(LLC.dist(),(Data.down_data_obs[i,1] - sim_dat[j].down_match[i,1])/h[1])
                + logpdf(LLC.dist(),(Data.down_data_obs[i,2] - sim_dat[j].down_match[i,2])/h[2])
                + logpdf(LLC.dist(),((Data.wages_obs[i] - sim_dat[j].wages_match[i])/Data.wages_obs[i])/h[3])
                + logpdf(LLC.dist(),(Data.measures_obs[i,1] - sim_dat[j].measures_match[i,1])/h[4])
                + logpdf(LLC.dist(),(Data.measures_obs[i,2] - sim_dat[j].measures_match[i,2])/h[5])
                ) 
            end
        end
    end
  
    if all(isfinite,like)    
        if LLC.trim_percent > 0
            #set up trimming function following Fermanian and Salanie
        
            delta_trim = log(percentile(like,LLC.trim_percent))/log(havg); 
            #println(percentile(like,LLC.trim_percent),", ",log(percentile(like,LLC.trim_percent)),", ",log(hall),", ",delta_trim)
            if isinf(abs(delta_trim)) #if above trim fn yields computational zeros, then we need less smooth trim fn        
                return llnf =  -2.0^50
            else  
                hd =  havg^delta_trim; #ensure hd, hd3, hd4 are not computationally zero
                hd3 = hd^3.0;
                hd4 = hd^4.0;
                tau_trim = zeros(LLC.n_firms)
                tau_trim .= 4 .*(like.-hd).^3 ./ hd3 .- 3.0 .*(like.-hd).^4.0 ./ hd4;
                tau_trim[like .< hd] .=0.0
                tau_trim[like .> 2.0*hd] .=1.0
                maxtau = findmax(tau_trim)
            end
        else 
            tau_trim = zeros(LLC.n_firms)
            tau_trim.=1.0
            maxtau = findmax(tau_trim)
        end

        ll = 0 
        if tau_trim==0.0 || isnan(maxtau[1]) || maxtau[1]<=0.00001
            ll =  -2.0^50
        else
            for i in 1:LLC.n_firms
                if tau_trim[i]>0.0    
                    ll+=tau_trim[i]*(log(like[i])-(LLC.n_sim*hall))             
                end
            end
        end
        #println("parameter: ", round.(b_est, digits=5), " likelihood: ", ll/LLC.n_firms)
        println( " likelihood: ", ll/LLC.n_firms)
        if ll == 0
            println(maxtau)
            println(hall," ", delta_trim, " ", hd) 
        end
        llnf = ll/LLC.n_firms
        if isnan(llnf)
            llnf = -2.0^50
        end
    else
        llnf = -2.0^50
    end
   return llnf 

end
