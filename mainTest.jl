using Pkg
Pkg.instantiate()
using ClusterManagers,Distributed, BenchmarkTools 
# Add in the cores allocated by the scheduler as workers
#addprocs(4)
#addprocs(SlurmManager(parse(Int,ENV["SLURM_NTASKS"])-1))

#print("Added workers: ")

#println(nworkers())
#setting up the linear program

# Required Packages
#using Pkg
#Pkg.add(url="https://github.com/mcreel/Econometrics.git")

using Distributions, Random, LinearAlgebra, SparseArrays, StaticArrays, GLPK, Cbc, Optim, FiniteDiff,GaussianDistributions
using StatsBase, KernelDensitySJ, ThreadsX, FLoops, JuMP, BlackBoxOptim
#use the package at  https://github.com/amirkp/AssignmentDual.jl.git to add Assigment
# Pkg.add(url="https://github.com/amirkp/AssignmentDual.jl.git")
using Assignment, Folds, NLopt, MultistartOptimization, Econometrics
using Distributed, CSV, Tables, DataFrames
using CMAEvolutionStrategy # Global Optimizer
@everywhere using Distributions, Random, LinearAlgebra, SparseArrays, StaticArrays, GLPK, Cbc, Optim
@everywhere using StatsBase, KernelDensitySJ, ThreadsX, FLoops, JuMP, Econometrics, Statistics,GaussianDistributions
@everywhere using Assignment, Folds, NLopt, Distributed, CSV, Tables


# load files
# ############

include("sim_data_like_Parallel.jl")
include("loglikefn.jl")
#include("readInData.jl")
include("HelperFuncsParallel.jl")
include("ExamineResults.jl")

rawdataallvars = CSV.File("Combined_All2013.csv"; header=true) |> DataFrame;

rawdataselectvars = rawdataallvars[:,[:CEOpos_NumJobs_NumInd, :revenue_residual, 
                    :size_LogTotalAsset, :num_seg, :tot_yr_comp, :annual_StockReturn, :adj_roa]];

rawdata_nomissing = dropmissing(rawdataselectvars);

Compparams = (n_firms=length(rawdata_nomissing.CEOpos_NumJobs_NumInd), n_sim=10, trim_percent=0, 
                hmethod=4, nparams=92, logcompdum=1, dist=Normal);

RawData = (up_data_obs = [copy(rawdata_nomissing.CEOpos_NumJobs_NumInd) copy(rawdata_nomissing.revenue_residual) ones(Compparams.n_firms)], 
    down_data_obs = [copy(rawdata_nomissing.size_LogTotalAsset) copy(rawdata_nomissing.num_seg) ones(Compparams.n_firms)], 
    wages_obs = copy(rawdata_nomissing.tot_yr_comp), 
    measures_obs = [copy(rawdata_nomissing.annual_StockReturn) copy(rawdata_nomissing.adj_roa)]);

absmax = (CEOpos_NumJobs_NumInd = maximum([abs.([maximum(rawdata_nomissing.CEOpos_NumJobs_NumInd)]),
                                    abs.([minimum(rawdata_nomissing.CEOpos_NumJobs_NumInd)])]),
        revenue_residual = maximum([abs.([maximum(rawdata_nomissing.revenue_residual)]),
                                    abs.([minimum(rawdata_nomissing.revenue_residual)])]),
        size_LogTotalAsset = maximum([abs.([maximum(rawdata_nomissing.size_LogTotalAsset)]),
                                    abs.([minimum(rawdata_nomissing.size_LogTotalAsset)])]),                           
        num_seg = maximum([abs.([maximum(rawdata_nomissing.num_seg)]),
                                    abs.([minimum(rawdata_nomissing.num_seg)])]))                           

RenormData = (up_data_obs = [copy(rawdata_nomissing.CEOpos_NumJobs_NumInd)./absmax.CEOpos_NumJobs_NumInd  copy(rawdata_nomissing.revenue_residual)./absmax.revenue_residual  ones(Compparams.n_firms)], 
            down_data_obs = [copy(rawdata_nomissing.size_LogTotalAsset)./absmax.size_LogTotalAsset copy(rawdata_nomissing.num_seg)./absmax.num_seg ones(Compparams.n_firms)], 
            wages_obs = copy(rawdata_nomissing.tot_yr_comp), 
            measures_obs = [copy(rawdata_nomissing.annual_StockReturn) copy(rawdata_nomissing.adj_roa)]);

quasi_seed = rand(UInt64) + hash(time_ns());  
# create a seed based on the current time and a random number
# create a random number generator with the seed
cutlength = 300; #take first 300 obs to spead up for now. 
InitData = (up_data_obs = copy(RenormData.up_data_obs[1:cutlength,:]), 
            down_data_obs = copy(RenormData.down_data_obs[1:cutlength,:]), 
            wages_obs = copy(RenormData.wages_obs[1:cutlength]), 
            measures_obs = copy(RenormData.measures_obs[1:cutlength,:]));


Initparams = (n_firms=length(rawdata_nomissing.CEOpos_NumJobs_NumInd[1:cutlength]), n_sim=10, trim_percent=10, hmethod=4, nparams=92, logcompdum=1, dist=Normal)

h = zeros(5);
#hmethod==4
if (Compparams.hmethod==4)
    # Optimize over choice of h
    #res_bcv = Optim.optimize(h -> bcv2_fun(h,RenormData,Compparams.n_firms,Compparams.logcompdum), rand(5))
    # res_bcv = Optim.optimize(bcv2_fun, rand(3),BFGS(),autodiff = :forward)
                  
    #h = abs.(Optim.minimizer(res_bcv))
    make_closuresH(Data,n_firms,n_sim,logcompdum) = h -> bcv2_fun(h,Data,n_firms,n_sim,logcompdum)
    hfun = make_closuresH(RenormData,Initparams.n_firms,Initparams.n_sim,Initparams.logcompdum)

    res_CMAE = CMAEvolutionStrategy.minimize(hfun,rand(5), 1.,
    lower = zeros(5),
     upper = 5000*ones(5),
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
else 
    h = zeros(5)
end 



igrid = Vector{Integer}(1:Compparams.nparams);              

lb= Array{Float64}(undef,0) 
ub= Array{Float64}(undef,0)

b_init = rand(Compparams.nparams);
b_init[Compparams.nparams-19]=0.0;
b_init[Compparams.nparams-25]=log(mean(InitData.wages_obs));
b_init[Compparams.nparams-18] =10.0;

for i=1:Compparams.nparams-1;
        lowerbnd,upperbnd = returnBounds(i)
        push!(lb,lowerbnd)
        push!(ub,upperbnd)
end
i_cal = [Compparams.nparams-19];
b_cal = [0.0];
igrid = Vector{Integer}(1:Compparams.nparams);              
i_est = deleteat!(igrid,Compparams.nparams-19);
b_est = b_init
b_est = parse.(Float64,split(last(eachline("TestNewIterate0613Normal.csv")),","));
deleteat!(b_est,Compparams.nparams-19);
make_closures(b_cal,i_cal,Data,h,quasi_seed,params;multithread) = b_est -> -loglikepr(b_est,b_cal,i_cal,Data,h,quasi_seed,params;multithread)
nll = make_closures(b_cal,i_cal,InitData,h,quasi_seed, Initparams;multithread=true)
res_CMAE = CMAEvolutionStrategy.minimize(nll,b_est, 1.,
        lower = lb,
         upper = ub,
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
b_init = rand(Compparams.nparams);
b_init[1:Compparams.nparams-20] = xbest(res_CMAE)[1:Compparams.nparams-20]
b_init[Compparams.nparams-19]=0.0;
b_init[Compparams.nparams-18:Compparams.nparams]=xbest(res_CMAE)[Compparams.nparams-19:Compparams.nparams-1]
        
        # P = MinimizationProblem(b_est -> -loglikepr(b_est,b_cal,i_cal,InitData,h,quasi_seed,Initparams;multithread=false), lb, ub)
        # local_method = NLoptLocalMethod(NLopt.LN_BOBYQA)
        # multistart_method = TikTak(100)
        # p = multistart_minimization(multistart_method, local_method, P)
        # p.location, p.value
        # b_init[1:21]=p.location[1:21]
        # b_init[22]=0.0;
        # b_init[23]=p.location[22]
CSV.write("TestNewIterate0613Normal.csv", Tables.table(b_init'), append=true) 
        #CSV.write("saminResults.csv", Tables.table(conv'), append=true) 
    
Initparams = (n_firms=length(rawdata_nomissing.CEOpos_NumJobs_NumInd[1:cutlength]), n_sim=10, trim_percent=10, hmethod=4, nparams=92, logcompdum=1, dist=Cauchy)
nll2 = make_closures(b_cal,i_cal,InitData,h,quasi_seed, Initparams;multithread=true)

        res_CMAE = CMAEvolutionStrategy.minimize(nll2,b_est, 1.,
        lower = lb,
         upper = ub,
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
        b_init = rand(Compparams.nparams);
        b_init[1:Compparams.nparams-20] = xbest(res_CMAE)[1:Compparams.nparams-20]
        b_init[Compparams.nparams-19]=0.0;
        b_init[Compparams.nparams-18:Compparams.nparams]=xbest(res_CMAE)[Compparams.nparams-19:Compparams.nparams-1]
        
        # P = MinimizationProblem(b_est -> -loglikepr(b_est,b_cal,i_cal,InitData,h,quasi_seed,Initparams;multithread=false), lb, ub)
        # local_method = NLoptLocalMethod(NLopt.LN_BOBYQA)
        # multistart_method = TikTak(100)
        # p = multistart_minimization(multistart_method, local_method, P)
        # p.location, p.value
        # b_init[1:21]=p.location[1:21]
        # b_init[22]=0.0;
        # b_init[23]=p.location[22]
        CSV.write("TestNewIterate0613Cauchy.csv", Tables.table(b_init'), append=true) 
        #CSV.write("saminResults.csv", Tables.table(conv'), append=true) 



resParams = parse.(Float64,split(last(eachline("TestNewIterate0613Normal.csv")),","));
SimP = StrucParams(resParams)
AvgSimData, SimData = CreateNewSimData(InitData,quasi_seed,Initparams,SimP;multithread=false)
using Plots
PlotResults(InitData,AvgSimData)