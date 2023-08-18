function ll = loglikepr(b_est, b_cal, i_cal, Data,h,LLC)
    % inputs are:
    % b_est = parameter values of parameters to be estimated
    % b_cal = parameters values of parameters to be calibrated
    % i_cal = index in b for parameters to be calibrated
    % Data.up_data_obs - the observable part of up_data
    % Data.down_data_obs - the obs part of down_data
    % Data.wages_obs - obs wages
    % Data.measures_obs = obs measures
    % SimIn = structure with some allocations
   
    ic = 1;
    ie = 1;
    nb = length(b_est)+length(b_cal);
    b = zeros(nb,1);
    for ib= 1:nb
        if ic<=length(i_cal)      
            if ib == i_cal(ic)
                b(ib) = b_cal(ic);
                ic = ic+ 1;
            else 
                b(ib) = b_est(ie); 
                ie = ie+ 1;
            end
        else
            b(ib) = b_est(ie);
            ie =ie+ 1;
        end
    end

    LLp = StrucParams(b);
  
    S_sim = ComputeSMatrix(LLC,Data.up_data_obs,Data.down_data_obs,LLp);
    if anynan(S_sim)
        ll = -2.0^50;
        return;
    end
    sim_dat.down_match = zeros([LLC.n_sim size(Data.down_data_obs)]);
    sim_dat.wages_match = zeros([LLC.n_sim size(Data.wages_obs)]);
    sim_dat.measures_match = zeros([LLC.n_sim size(Data.measures_obs)]);

    for i=1:LLC.n_sim
          [ sim_dat.down_match(i,:,:), sim_dat.wages_match(i,:), sim_dat.measures_match(i,:,:) ]= sim_data_like(Data.up_data_obs, Data.down_data_obs, LLp, S_sim, i,LLC);
    end
        
    % if multithread 
    %    sim_dat = pmap(i -> sim_data_like(Data.up_data_obs', Data.down_data_obs', LLp, S_sim, i, quasi_seed,LLC), 1:LLC.n_sim, batch_size=cld(LLC.n_sim,nworkers()))
    %else 
    %    sim_dat = map(i -> sim_data_like(Data.up_data_obs', Data.down_data_obs', LLp, S_sim, i, quasi_seed,LLC), 1:LLC.n_sim)
    %end
    
    %if any(isnan(sim_dat(1:LLC.n_sim).down_match(1,1)))
    %    llnf = -2.0^50
    %else
            
    ll=0.0;   
    like = zeros(LLC.n_firms,1);
    hall = prod(h);
    
    havg = hall^(1/3);
  if LLC.dist == 'Normal'
    for i= 1:LLC.n_firms
        for j = 1:LLC.n_sim
            like(i) = like(i) + exp(...
                  logmvnpdf((Data.down_data_obs(i,1) - sim_dat.down_match(j,i,1))/h(1),0,1)....
                + logmvnpdf((Data.wages_obs(i) - sim_dat.wages_match(j,i))/h(2),0,1) ....
                + logmvnpdf((Data.measures_obs(i,1) - sim_dat.measures_match(j,i))/h(3),0,1)....
                );
            
        end
    end
  elseif LLC.dist == 'Cauchy'
    for i= 1:LLC.n_firms
        for j = 1:LLC.n_sim
            like(i) = like(i) + exp(... 
                  -log(pi*(1+(Data.down_data_obs(i,1) - sim_dat.down_match(j,i,1))/h(1)).^2)...
                + -log(pi*(1+(Data.wages_obs(i) - sim_dat.wages_match(j,i))/h(2)).^2) ...
                + -log(pi*(1+(Data.measures_obs(i,1) - sim_dat.measures_match(j,i))/h(3)).^2)...
                );
            
        end
    end
  end
  
  if all(isfinite(like(:)))    
      if LLC.trim_percent > 0
            %set up trimming function following Fermanian and Salanie
        
            delta_trim = log(prctile(like,LLC.trim_percent))/log(havg); 
            if isinf(abs(delta_trim)) %if above trim fn yields computational zeros, then we need less smooth trim fn        
                ll =  -2.0^50
                return;
            else  
                hd =  havg^delta_trim; %ensure hd, hd3, hd4 are not computationally zero
                hd3 = hd^3.0;
                hd4 = hd^4.0;
                tau_trim = zeros(LLC.n_firms,1);
                tau_trim = 4 *(like-hd).^3 ./ hd3 - 3.0 *(like-hd).^4.0 ./ hd4;
                tau_trim(like < hd) =0.0;
                tau_trim(like > 2.0*hd) =1.0;
                [~,maxtau] = max(tau_trim);
            end
        else 
            tau_trim = zeros(LLC.n_firms,1);
            tau_trim(:) = 1.0;
            maxtau = 1;
        end

        ll = 0; 
        if ( isnan(maxtau(1)) || (maxtau(1)<=0.00001))
            ll =  -2.0^50;
        else
            for i = 1:LLC.n_firms
                if tau_trim(i)>0.0    
                    ll = ll + tau_trim(i)*(log(like(i))-log(LLC.n_sim*hall)) ;            
                end
            end
        end
        %println("parameter: ", round.(b_est, digits=5), " likelihood: ", ll/LLC.n_firms)
        fprintf( " likelihood: %d \n", ll/LLC.n_firms)
        if ll == 0
            disp(maxtau)
            disp(hall," ", delta_trim, " ", hd) 
        end
        ll = ll/LLC.n_firms;
        if isnan(ll)
            ll = -2.0^50;
        end
    else
        ll = -2.0^50;
    end
end
