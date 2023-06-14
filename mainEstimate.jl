using ClusterManagers,Distributed, BenchmarkTools 
# Add in the cores allocated by the scheduler as workers
#addprocs(4)
#addprocs(SlurmManager(parse(Int,ENV["SLURM_NTASKS"])-1))

#print("Added workers: ")

#println(nworkers())
#setting up the linear program

# Required Packages
#using Pkg
#Pkg.add("https://github.com/mcreel/Econometrics.git")

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

rawdataselectvars = rawdataallvars[:,[:CEOpos_NumJobs_NumInd, :revenue_residual, :size_LogTotalAsset, :num_seg, :tot_yr_comp, :annual_StockReturn, :adj_roa]];
rawdata_nomissing = dropmissing(rawdataselectvars);
Compparams = (n_firms=length(rawdata_nomissing.CEOpos_NumJobs_NumInd), n_sim=10, trim_percent=0, hmethod=4, nparams=74, logcompdum=1, dist=Normal);

RawData = (up_data_obs = [copy(rawdata_nomissing.CEOpos_NumJobs_NumInd) copy(rawdata_nomissing.revenue_residual) ones(Compparams.n_firms)], 
    down_data_obs = [copy(rawdata_nomissing.size_LogTotalAsset) copy(rawdata_nomissing.num_seg) ones(Compparams.n_firms)], 
    wages_obs = copy(rawdata_nomissing.tot_yr_comp), 
    measures_obs = [copy(rawdata_nomissing.annual_StockReturn) copy(rawdata_nomissing.adj_roa)]);


quasi_seed = rand(UInt64) + hash(time_ns());  
# create a seed based on the current time and a random number
# create a random number generator with the seed


h = zeros(5);
#hmethod==4
if (Compparams.hmethod==4)
    # Optimize over choice of h
    res_bcv = Optim.optimize(h -> bcv2_fun(h,RawData,Compparams.n_firms,Compparams.logcompdum), rand(5))
    # res_bcv = Optim.optimize(bcv2_fun, rand(3),BFGS(),autodiff = :forward)
                  
    h = abs.(Optim.minimizer(res_bcv))
else 
    h = zeros(5)
end 

cutlength = 300; #take first 300 obs to spead up for now. 
InitData = (up_data_obs = [copy(rawdata_nomissing.CEOpos_NumJobs_NumInd[1:cutlength]) copy(rawdata_nomissing.revenue_residual[1:cutlength]) ones(cutlength)], 
down_data_obs = [copy(rawdata_nomissing.size_LogTotalAsset[1:cutlength]) copy(rawdata_nomissing.num_seg[1:cutlength]) ones(cutlength)], 
wages_obs = copy(rawdata_nomissing.tot_yr_comp[1:cutlength]), 
measures_obs = [copy(rawdata_nomissing.annual_StockReturn[1:cutlength]) copy(rawdata_nomissing.adj_roa[1:cutlength])]);
Initparams = (n_firms=length(rawdata_nomissing.CEOpos_NumJobs_NumInd[1:cutlength]), n_sim=10, trim_percent=0, hmethod=4, nparams=74, logcompdum=1, dist=Normal)


igrid = Vector{Integer}(1:Compparams.nparams);              

lb= Array{Float64}(undef,0) 
ub= Array{Float64}(undef,0)

b_init = rand(23);
b_init[22]=0.0;
for i=1:Compparams.nparams;
    if (i<22)
        i_cal = deleteat!(igrid,1);
        b_cal = b_init[i_cal];
        b_est = b_init[1:i];
        lowerbnd,upperbnd = returnBounds(i)
        push!(lb,lowerbnd)
        push!(ub,upperbnd)
        make_closures2(b_cal,i_cal,Data,h,quasi_seed,params;multithread) = b_est -> -loglikepr(b_est,b_cal,i_cal,Data,h,quasi_seed,params;multithread)
        nll = make_closures2(b_cal,i_cal,InitData,h,quasi_seed, Initparams;multithread=false)
        res = Optim.optimize(nll, b_est)
        #res = Optim.optimize(nll, b_est, LBFGS())
        #res = Optim.optimize(nll, lb,ub, b_est, SAMIN(; rt=0.95))
        #res = Optim.optimize(nll, lb,ub, b_est, ParticleSwarm(; n_particles=10))
        b_init[1:i]=Optim.minimizer(res)
        #res,obj,conv,details = samin(nll,b_est,lb,ub, rt=0.95)
        #b_init[1:i]=res
        # res_CMAE = CMAEvolutionStrategy.minimize(nll,b_init[1:i], 1.,
        # lower = lb,
        #  upper = ub,
        #  noise_handling = 1.,
        #  callback = (object, inputs, function_values, ranks) -> nothing,
        #  parallel_evaluation = false,
        #  multi_threading = false,
        #  verbosity = 1,
        #  seed = rand(UInt),
        #  maxtime = 10000,
        #  maxiter = nothing,
        #  maxfevals = nothing,
        #  ftarget = nothing,
        #  xtol = nothing,
        #  ftol = 1e-3)
        # # # # Estimated parameters: 
        # b_init[1:i] = xbest(res_CMAE)
        
        
        # P = MinimizationProblem(b_est -> -loglikepr(b_est,b_cal,i_cal,InitData,h,quasi_seed,Initparams;multithread=false), lb, ub)
        # local_method = NLoptLocalMethod(NLopt.LN_BOBYQA)
        # multistart_method = TikTak(100)
        # p = multistart_minimization(multistart_method, local_method, P)
        # p.location, p.value
        # b_init[1:i]=p.location
        CSV.write("TestNewIterate0525.csv", Tables.table(b_init'), append=true) 
        #CSV.write("saminResults.csv", Tables.table(conv'), append=true) 
    end
    if (i>22)
        i_cal = [22];
        b_cal = [0.0];
        igrid = Vector{Integer}(1:Compparams.nparams);              
        i_est = deleteat!(igrid,22);
        b_est = b_init[1:21]
        push!(b_est,b_init[23])
        lowerbnd,upperbnd = returnBounds(i)
        push!(lb,lowerbnd)
        push!(ub,upperbnd)
        make_closures(b_cal,i_cal,Data,h,quasi_seed,params;multithread) = b_est -> -loglikepr(b_est,b_cal,i_cal,Data,h,quasi_seed,params;multithread)
        nll = make_closures(b_cal,i_cal,InitData,h,quasi_seed, Initparams;multithread=false)
        res = Optim.optimize(nll, b_est)
        b_init[1:21]=Optim.minimizer(res)[1:21]
        b_init[22]=0.0;
        b_init[23]=Optim.minimizer(res)[22]
        #res = Optim.optimize(nll, b_est, LBFGS())
        #  res = Optim.optimize(nll, lb,ub, b_est, SAMIN(; rt=0.95))
        #res = Optim.optimize(nll, lb,ub, b_est, ParticleSwarm(; n_particles=10))
        #res,obj,conv,details = samin(nll,b_est,lb,ub, rt=0.95)
        #b_init[1:17]=res[1:17]
        #b_init[18]=0.0;
        #b_init[19]=res[18]
        # res_CMAE = CMAEvolutionStrategy.minimize(nll,b_est, 1.,
        # lower = lb,
        #  upper = ub,
        #  noise_handling = 1.,
        #  callback = (object, inputs, function_values, ranks) -> nothing,
        #  parallel_evaluation = false,
        #  multi_threading = false,
        #  verbosity = 1,
        #  seed = rand(UInt),
        #  maxtime = 10000,
        #  maxiter = nothing,
        #  maxfevals = nothing,
        #  ftarget = nothing,
        #  xtol = nothing,
        #  ftol = 1e-3)
        # # # # Estimated parameters: 
        # b_init[1:17] = xbest(res_CMAE)[1:17]
        # b_init[18]=0.0;
        # b_init[19]=xbest(res_CMAE)[18]
        
        # P = MinimizationProblem(b_est -> -loglikepr(b_est,b_cal,i_cal,InitData,h,quasi_seed,Initparams;multithread=false), lb, ub)
        # local_method = NLoptLocalMethod(NLopt.LN_BOBYQA)
        # multistart_method = TikTak(100)
        # p = multistart_minimization(multistart_method, local_method, P)
        # p.location, p.value
        # b_init[1:21]=p.location[1:21]
        # b_init[22]=0.0;
        # b_init[23]=p.location[22]
        CSV.write("TestNewIterate0525.csv", Tables.table(b_init'), append=true) 
        #CSV.write("saminResults.csv", Tables.table(conv'), append=true) 
    end
end
    
# b_final= parse.(Float64,split(last(eachline("IterativeParams2013_6.csv")),","));
# b_final[22]=0;
# FinalParams = StrucParams(b_final);
# AvgSimData = CreateNewSimData(InitData,quasi_seed,Initparams,FinalParams;multithread=false)

PlotResults(InitData,AvgSimData)