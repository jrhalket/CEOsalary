function  S_sim= ComputeSMatrix(CP,up_data,down_data,params)
    %CP are computational parameters
    
    rSigma = params.risk * params.Sigma;
    S_sim = zeros(CP.n_firms,CP.n_firms);
    for n2 = 1:CP.n_firms
        for n1 = 1:CP.n_firms
            S_sim(n1,n2) = ScalcAll(up_data(n1,:),down_data(n2,:),rSigma,params);
        end
    end
end