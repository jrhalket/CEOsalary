function p= StrucParams(b)
     global ParamScale
        %Set up thetas
      
        p.theta_ga_1 = [b(1:2) b(3:4)] ;
        p.theta_ca_1 = [b(5:6) b(7:8)] ;
        %set up Cov matrix for shocks to measures
      %  p.mean_price = ParamScale; %exp(b(9));
    
        p.Sigma = exp(b(10))%*ParamScale.^2;
            
        p.risk = exp(b(11));  %exp to keep risk coefficient positive
        p.zeta_1 = [b(12:13) b(14:15)] ;
        p.rm = [b(16) b(17)]*ParamScale;
        p.rf = [b(18) b(9)]*ParamScale;
        
        

end