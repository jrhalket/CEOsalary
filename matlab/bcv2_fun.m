function val =bcv2_fun(h,Data,n_firms,n_sims)
    h=abs(h);
    N = n_firms;
    ll = 0.0;
    for i = 1:n_firms
        for j=1:n_firms
            if (j~=i)
                expr_1 = ((Data.down_data_obs(i,1)-Data.down_data_obs(j,1))/h(1))^2 + ....
                         ((Data.wages_obs(i)-Data.wages_obs(j))/h(2))^2 + ....
                         ((Data.measures_obs(i,1)-Data.measures_obs(j,1))/h(3))^2 ;
                expr_2 =  exp(logmvnpdf((Data.down_data_obs(i,1)-Data.down_data_obs(j,1))/h(1),0,1) + ....
                          logmvnpdf((Data.wages_obs(i)-Data.wages_obs(j))/h(2),0,1) +....
                          logmvnpdf((Data.measures_obs(i,1)-Data.measures_obs(j,1))/h(3),0,1));
                ll = ll + (expr_1 - (2*3 +4)*expr_1 + (3^2 +2*3))*expr_2;
            end
        end
    end
    val = ((sqrt(2*pi))^3 * N *h(1)*h(2)*h(3))^(-1) + ...
                            ((4*N*(N-1))*h(1)*h(2)*h(3))^(-1) * ll ;
    fprintf('band: %d,%d,%d , val: %d \n',h, val);
end