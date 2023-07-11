function [downMatchobs, wages_match, measures_match]=sim_data_like(upData,downData,params,S_sim,i,CompParams)

    rSigma = params.risk * params.Sigma;

    %set random stream
    myStream.Substream = i ;

    up_data_sim = randn(CompParams.n_firms,1);
    down_data_sim = randn(CompParams.n_firms,1);
  
    rho_m_sim = up_data_sim*(params.rm*downData');
    rho_f_sim = (upData*params.rf')*down_data_sim';

    C_sim = S_sim + rho_m_sim + rho_f_sim; %pairwise surplus 

    [upMatchTodown,downMatchToup,gain,down_profit,up_profit]=assign2D(C_sim,true); %true --> max

    %per notes in assign2d, must renormalize down_profit
    down_profit = down_profit + max(C_sim(:));

    %put down elements in order of their up partners.
    downMatchobs = downData(downMatchToup,:);
    downMatchsim = down_data_sim(downMatchToup);
    downMatchProfit = down_profit(downMatchToup);

    %make coefficients for matches
    ga_1 = diag(exp(upData*params.theta_ga_1*downMatchobs'));
    ca_1 = diag(exp(upData*params.theta_ca_1*downMatchobs'));

    
    bstar_match = ga_1./ca_1 ./(ga_1.^2./ca_1 + rSigma);
    measures_match = ga_1.^2./ca_1 .* bstar_match + sqrt(params.Sigma)*randn(CompParams.n_firms,1);
    S_match = 0.5* ga_1./ca_1 .* bstar_match;
    rho_m_match = diag(up_data_sim*(params.rm*downMatchobs'));

    up_val = -S_match + rho_m_match; %manager gets -S_match + unobs value, net of transfers

    % eq transfers
    up_prices = up_profit - up_val;

    wages_match = params.mean_price + up_prices + bstar_match.*measures_match;

    measures_obs = diag(upData*params.zeta_1*downMatchobs') + measures_match;
end