


rawdataallvars = CSV.File("Combined_All2013.csv"; header=true) |> DataFrame;

rawdataselectvars = rawdataallvars[:,[:CEOpos_NumJobs_NumInd, :revenue_residual, :size_LogTotalAsset, :num_seg, :tot_yr_comp, :annual_StockReturn, :Tobin_Qf]];
rawdata_nomissing = dropmissing(rawdataselectvars);
rawout = (up_data = [copy(rawdata_nomissing.CEOpos_NumJobs_NumInd) copy(rawdata_nomissing.revenue_residual)]', 
    down_match_ret = [copy(rawdata_nomissing.size_LogTotalAsset) copy(rawdata_nomissing.num_seg)]', 
    wages_match = copy(rawdata_nomissing.tot_yr_comp), 
    measures_match = [copy(rawdata_nomissing.annual_StockReturn) copy(rawdata_nomissing.Tobin_Qf)]')
