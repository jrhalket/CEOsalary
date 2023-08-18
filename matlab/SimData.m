function [up_data, downMatchobs, wages_match, measures_match]=SimData(params,i,CompParams) 
%creates a simulated raw data set

    rSigma = params.risk * params.Sigma;

    %set random stream
    myStream.Substream = i ;

    up_data(:,1) = rand(CompParams.n_firms,1); 
    down_data(:,1) = rand(CompParams.n_firms,1); 
    up_data(:,2) = ones(CompParams.n_firms,1);
    down_data(:,2) = ones(CompParams.n_firms,1);
    S_sim = ComputeSMatrix(CompParams,up_data,down_data,params);
    
    up_data_sim = randn(CompParams.n_firms,1);
    down_data_sim = randn(CompParams.n_firms,1);
  
    rho_m_sim = up_data_sim*(params.rm*down_data');
    rho_f_sim = (up_data*params.rf')*down_data_sim';

    C_sim = S_sim + rho_m_sim + rho_f_sim; %pairwise surplus 

    [upMatchTodown,downMatchToup,gain,down_profit,up_profit]=assign2D(C_sim,true); %true --> max

    %per notes in assign2d, must renormalize down_profit
    down_profit = down_profit + max(C_sim(:));
    up_profit = up_profit + max(C_sim(:));

    %put down elements in order of their up partners.
    downMatchobs = down_data(upMatchTodown,:);
    downMatchsim = down_data_sim(upMatchTodown);
    downMatchProfit = down_profit(upMatchTodown);

    %make coefficients for matches
    ga_1 = diag(exp(up_data*params.theta_ga_1*downMatchobs'));
    ca_1 = diag(exp(up_data*params.theta_ca_1*downMatchobs'));

    
    bstar_match = ga_1./ca_1 ./(1./ca_1 + rSigma);
    measures_match = 1./ca_1 .* bstar_match + sqrt(params.Sigma)*randn(CompParams.n_firms,1);
    S_match = 0.5* ga_1./ca_1 .* bstar_match;
    rho_m_match = diag(up_data_sim*(params.rm*downMatchobs'));

    up_val = -S_match + rho_m_match; %manager gets -S_match + unobs value, net of transfers

    % eq transfers
    up_prices = up_profit - up_val;

    wages_match =  up_prices + bstar_match.*measures_match;
