@everywhere function sim_data_like(up_data_obs_in,down_data_obs_in,p_in::NamedTuple,S_sim,i,quasi_seed,CR)
    #rΣ = Matrix{Float64}(undef,2,2)
    rΣ = p_in.risk * p_in.Σ
 
    # Types
    # Upstream x_1, x_2, delta^m
    # Downstream y_1, y_2, delta^f

    rng = MersenneTwister(quasi_seed) 
    brn = rand(rng)
    c= floor(brn * 10^3)
    Random.seed!(Int(c+i))  

   
    
    up_data_sim = rand(Normal(0.0,1.0), CR.n_firms);
    down_data_sim = rand(Normal(0.0,1.0), CR.n_firms);
  
    rho_m_sim = up_data_sim*(p_in.rm'*down_data_obs_in);
    rho_f_sim = (up_data_obs_in'*p_in.rf)*down_data_sim';

    C_sim = -S_sim - rho_m_sim - rho_f_sim; #negative of pairwise surplus 

    match_sim, up_profit_sim, down_profit_sim = find_best_assignment(C_sim)

    down_match_sim = similar(up_data_sim);
    down_match_obs = similar(down_data_obs_in);
    try 
        match_sim[1][2]
    catch
        fill!(down_match_sim,NaN)
        wages_match_sim = zeros(CR.n_firms);
        fill!(wages_match_sim,NaN);
        measures_match_sim = zeros(CR.n_firms);
        fill!(measures_match_sim,NaN);
        Simout = SimOutputs(down_match_obs', wages_match_sim, measures_match_sim);
        return Simout
    end
    for i in 1:CR.n_firms
        down_match_sim[i] = down_data_sim[match_sim[i][2]];
        down_match_obs[:,i] = down_data_obs_in[:,match_sim[i][2]];
    end
    
    down_match_profit_sim =  similar(down_match_sim)
    for i in 1:CR.n_firms
        down_match_profit_sim[i] = down_profit_sim[match_sim[i][2]];
    end
 
    # minimum downstream profit =0 
    profit_diff_sim = findmin(down_match_profit_sim)[1];
    down_match_profit_sim .= down_match_profit_sim .+ profit_diff_sim;
    up_profit_sim .= up_profit_sim .- profit_diff_sim;
    
    
    SC = makeCoefs(up_data_obs_in,down_match_obs,CR.n_firms,p_in)
    bstar_match_sim = zeros(CR.n_firms);
    measures_match_sim = zeros(CR.n_firms);
    S_match_sim = zeros(CR.n_firms);
    rho_m_match_sim = similar(S_match_sim);
    for n1 in 1:CR.n_firms;
        bstar_match_sim[n1] = SC[n1].ga_1/SC[n1].ca_1/(SC[n1].ga_1^2/SC[n1].ca_1 + rΣ);
  
       #simulate matched measures
       measures_match_sim[n1] = SC[n1].ga_1^2 / SC[n1].ca_1 * bstar_match_sim[n1] +rand(Normal(0.0,sqrt(p_in.Σ)));
       S_match_sim[n1] = 0.5 * SC[n1].ga_1 / SC[n1].ca_1 * bstar_match_sim[n1];
       rho_m_match_sim[n1] = view(down_match_obs,:,n1)' * p_in.rm * up_data_sim[n1];    

    end
    up_valuation_sim = -S_match_sim + rho_m_match_sim; #manager gets -S_match + unobs value, net of transfers

    # eq transfers
    up_prices_sim = up_profit_sim - up_valuation_sim;

    #down_prices_sim= γ_sim'*up_prices_sim;

    #simulate wages:
    wages_match_sim = zeros(CR.n_firms)
    for n1 in 1:CR.n_firms;
        wages_match_sim[n1] = p_in.mean_price + up_prices_sim[n1] + bstar_match_sim[n1]*measures_match_sim[n1];
    end    
    
    down_match = copy(down_match_obs)'
    wages_match = copy(wages_match_sim)
    measures_match = copy(measures_match_sim)
    for n1 in 1:CR.n_firms;
        measures_match[n1] = up_data_obs_in[:,n1]'*p_in.zeta_1*down_match_obs[:,n1] + measures_match_sim[n1];
    end
    Simout = SimOutputs(down_match, wages_match, measures_match);
    return Simout
    
end