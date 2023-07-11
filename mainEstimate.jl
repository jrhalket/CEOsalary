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
include("readInData.jl")
include("HelperFuncsParallel.jl")
include("ExamineResults.jl")

FullData = ReadInData(0)
Compparams = (n_firms=length(FullData.up_data_obs[:,1]), n_sim=10, trim_percent=0, 
                    hmethod=4, nparams=18, logcompdum=0, dist=Normal);

cutlength = 100; #take first 300 obs to spead up for now.
InitData = CutData(cutlength,FullData)

quasi_seed = rand(UInt64) + hash(time_ns());  
# create a seed based on the current time and a random number
# create a random number generator with the seed



#hmethod==4
if (Compparams.hmethod==4)
    #h = GetH_BCV2(FullData,Compparams.n_firms,Compparams.n_sim,Compparams.logcompdum)
    h = [KernelDensitySJ.bwsj(FullData.down_data_obs[:,1]), 
    KernelDensitySJ.bwsj(FullData.wages_obs),
    KernelDensitySJ.bwsj(FullData.measures_obs[:,1])];
else 
    h = zeros(3)
end 

Initparams = (n_firms=length(InitData.up_data_obs[:,1]), n_sim=10, trim_percent=0, hmethod=4, nparams=18, logcompdum=0, dist=Cauchy)

using MATLAB
writeMatlabDataFrame(h,InitData,FullData)

igrid = Vector{Integer}(1:Compparams.nparams);              

lb= Array{Float64}(undef,0) 
ub= Array{Float64}(undef,0)

b_init = .1*ones(Compparams.nparams);#rand(Compparams.nparams);
b_init[16]=log(mean(InitData.wages_obs));
b_init[17]=0.01;
b_init[18]=0.01;

#optional ways to initialize parameters if files available:
#b_init = parse.(Float64,split(last(eachline("TestNewIterate0613Cauchy.csv")),","));
b_init = parse.(Float64,split(last(eachline("Test0707full.csv")),","));
#########

for i in 1:Compparams.nparams;
    i_cal = deleteat!(igrid,1);
    b_cal = b_init[i_cal];
    b_est = b_init[1:i];
    lowerbnd,upperbnd = returnBounds(i)
    push!(lb,lowerbnd)
    push!(ub,upperbnd)
    make_closures2(b_cal,i_cal,Data,h,quasi_seed,params;multithread) = b_est -> -loglikepr(b_est,b_cal,i_cal,Data,h,quasi_seed,params;multithread)
    nll = make_closures2(b_cal,i_cal,InitData,h,quasi_seed, Initparams;multithread=false)
        #res = Optim.optimize(nll, b_est)
        #res = Optim.optimize(nll, b_est, LBFGS())
        #res = Optim.optimize(nll, lb,ub, b_est, SAMIN(; rt=0.95))
        #res = Optim.optimize(nll, lb,ub, b_est, ParticleSwarm(; n_particles=10))
        #b_init[1:i]=Optim.minimizer(res)
        #res,obj,conv,details = samin(nll,b_est,lb,ub, rt=0.95)
        #b_init[1:i]=res
        res_CMAE = CMAEvolutionStrategy.minimize(nll,b_init[1:i], 1.,
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
        b_init[1:i] = xbest(res_CMAE)
        CSV.write("Test0708full.csv", Tables.table(b_init'), append=true) 
        
        
        # # P = MinimizationProblem(b_est -> -loglikepr(b_est,b_cal,i_cal,InitData,h,quasi_seed,Initparams;multithread=false), lb, ub)
        # # local_method = NLoptLocalMethod(NLopt.LN_BOBYQA)
        # # multistart_method = TikTak(100)
        # # p = multistart_minimization(multistart_method, local_method, P)
        # # p.location, p.value
        # # b_init[1:i]=p.location
        # CSV.write("TestNewIterate0618.csv", Tables.table(b_init'), append=true) 
        # #CSV.write("saminResults.csv", Tables.table(conv'), append=true) 
end
    
resParams = parse.(Float64,split(last(eachline("Test0708full.csv")),","));
SimP = StrucParams(resParams)
AvgSimData, SimData = CreateNewSimData(InitData,quasi_seed,Initparams,SimP;multithread=false)
using Plots
PlotResults(InitData,AvgSimData)


i_cal = [];
b_cal = [];
igrid = Vector{Integer}(1:Compparams.nparams);              
i_est = igrid #deleteat!(igrid,Compparams.nparams-19);
b_est = b_init
b_est = parse.(Float64,split(last(eachline("TestNewIterate0630.csv")),","));
@run loglikepr(b_est,b_cal,i_cal,InitData,h,quasi_seed,Initparams;multithread=false) 