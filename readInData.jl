function ReadInData(renormDummy)


    rawdataallvars = CSV.File("Combined_All2013.csv"; header=true) |> DataFrame;

    rawdataselectvars = rawdataallvars[:,[:CEOpos_NumJobs_NumInd, :revenue_residual, 
                        :size_LogTotalAsset, :num_seg, :tot_yr_comp, :annual_StockReturn, :adj_roa]];
    
    rawdata_nomissing = dropmissing(rawdataselectvars);
    
    
    
    RawData = (up_data_obs = [copy(rawdata_nomissing.CEOpos_NumJobs_NumInd) ones(length(rawdata_nomissing.revenue_residual))], 
        down_data_obs = [copy(rawdata_nomissing.size_LogTotalAsset) ones(length(rawdata_nomissing.revenue_residual))], 
        wages_obs = copy(rawdata_nomissing.tot_yr_comp), 
        measures_obs = [copy(rawdata_nomissing.adj_roa)]);
    
    if renormDummy ==1
        absmax = (CEOpos_NumJobs_NumInd = maximum([abs.([maximum(rawdata_nomissing.CEOpos_NumJobs_NumInd)]),
                                        abs.([minimum(rawdata_nomissing.CEOpos_NumJobs_NumInd)])]),
            revenue_residual = maximum([abs.([maximum(rawdata_nomissing.revenue_residual)]),
                                        abs.([minimum(rawdata_nomissing.revenue_residual)])]),
            size_LogTotalAsset = maximum([abs.([maximum(rawdata_nomissing.size_LogTotalAsset)]),
                                        abs.([minimum(rawdata_nomissing.size_LogTotalAsset)])]),                           
            num_seg = maximum([abs.([maximum(rawdata_nomissing.num_seg)]),
                                        abs.([minimum(rawdata_nomissing.num_seg)])]))                           
    
        RenormData =  (up_data_obs = [copy(rawdata_nomissing.CEOpos_NumJobs_NumInd)./absmax.CEOpos_NumJobs_NumInd ones(length(rawdata_nomissing.revenue_residual))], 
                down_data_obs = [copy(rawdata_nomissing.size_LogTotalAsset)./absmax.size_LogTotalAsset ones(length(rawdata_nomissing.revenue_residual))], 
                wages_obs = copy(rawdata_nomissing.tot_yr_comp), 
                measures_obs = [copy(rawdata_nomissing.annual_StockReturn)]);
    
        return RenormData
    else 
        return RawData
    end
end
function CutData(cutlength,FullData)
    InitData = (up_data_obs = copy(FullData.up_data_obs[1:cutlength,:]), 
                down_data_obs = copy(FullData.down_data_obs[1:cutlength,:]), 
                wages_obs = copy(FullData.wages_obs[1:cutlength]), 
                measures_obs = copy(FullData.measures_obs[1:cutlength,:]));
    
    return InitData
end 

function GetH_BCV2(DataIn,n_firms,n_sim,logcompdum)
    # Optimize over choice of h
    #res_bcv = Optim.optimize(h -> bcv2_fun(h,RenormData,Compparams.n_firms,Compparams.logcompdum), rand(5))
    # res_bcv = Optim.optimize(bcv2_fun, rand(3),BFGS(),autodiff = :forward)
                  
    #h = abs.(Optim.minimizer(res_bcv))
    make_closuresH(Data,n_firms,n_sim,logcompdum) = h -> bcv2_fun(h,Data,n_firms,n_sim,logcompdum)
    hfun = make_closuresH(DataIn,n_firms,n_sim,logcompdum)

    res_CMAE = CMAEvolutionStrategy.minimize(hfun,rand(3), 1.,
    lower = zeros(3),
     upper = 5000*ones(3),
     noise_handling = 1.,
     callback = (object, inputs, function_values, ranks) -> nothing,
     parallel_evaluation = false,
     multi_threading = false,
     verbosity = 1,
     seed = rand(UInt),
     maxtime = 10000,
     maxiter = nothing,
     maxfevals = nothing,
     ftarget = nothing,
     xtol = nothing,
     ftol = 1e-3)
    # # # Estimated parameters: 
     h = xbest(res_CMAE)
    return h
end
