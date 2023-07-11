function p= StrucParams(b)
     
        %Set up thetas
      
        p.theta_ga_1 = [b(1:2); b(3:4)] ;
        p.theta_ca_1 = [b(5:6); b(7:8)] ;
        p.zeta_1 = [b(9:10); b(11:12)] ;
        p.rm = b(13:14);
        p.rf = [b(15) 0.0] ;
        
        %set up Cov matrix for shocks to measures
        p.mean_price = exp(b(16));
    
        p.Sigma = exp(b(17));
            
        p.risk = exp(b(18));  %exp to keep risk coefficient positive

end