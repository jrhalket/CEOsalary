function ret=ScalcAll(up,dn,rS,parm)
        
    ga_1 = exp(up*parm.theta_ga_1*dn');
    ca_1 = exp(up*parm.theta_ca_1*dn');
    
    ats = ga_1/ca_1;
    F= (ga_1).^2 / ca_1 + rS;
    ret = 0.5 * ats*ats/F;
end   