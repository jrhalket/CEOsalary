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

FullData = ReadInData()
Compparams = (n_firms=length(FullData.up_data_obs[:,1]), n_sim=10, trim_percent=0, 
                    hmethod=4, nparams=92, logcompdum=0, dist=Normal);

cutlength = 30; #take first 300 obs to spead up for now.
InitData = CutData(cutlength,FullData)

quasi_seed = rand(UInt64) + hash(time_ns());  
# create a seed based on the current time and a random number
# create a random number generator with the seed



#hmethod==4
if (Compparams.hmethod==4)
    h = GetH_BCV2(FullData,Compparams.n_firms,Compparams.n_sim,Compparams.logcompdum)
else 
    h = zeros(5)
end 

Initparams = (n_firms=length(InitData.up_data_obs[:,1]), n_sim=10, trim_percent=10, hmethod=4, nparams=92, logcompdum=0, dist=Cauchy)


igrid = Vector{Integer}(1:Compparams.nparams);              

lb= Array{Float64}(undef,0) 
ub= Array{Float64}(undef,0)

b_init = rand(Compparams.nparams);
b_init[Compparams.nparams-19]=0.0;
b_init[Compparams.nparams-25]=log(mean(InitData.wages_obs));
b_init[Compparams.nparams-18] =10.0;
#optional ways to initialize parameters if files available:
b_init = parse.(Float64,split(last(eachline("TestNewIterate0613Cauchy.csv")),","));
b_init = parse.(Float64,split(last(eachline("TestNewIterate0618.csv")),","));
#########

for i=1:Compparams.nparams;
    if (i<Compparams.nparams-19)
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
        
        
        # # P = MinimizationProblem(b_est -> -loglikepr(b_est,b_cal,i_cal,InitData,h,quasi_seed,Initparams;multithread=false), lb, ub)
        # # local_method = NLoptLocalMethod(NLopt.LN_BOBYQA)
        # # multistart_method = TikTak(100)
        # # p = multistart_minimization(multistart_method, local_method, P)
        # # p.location, p.value
        # # b_init[1:i]=p.location
        # CSV.write("TestNewIterate0618.csv", Tables.table(b_init'), append=true) 
        # #CSV.write("saminResults.csv", Tables.table(conv'), append=true) 
    end
    if (i>Compparams.nparams-19)
        i_cal = deleteat!(igrid,2);
        b_cal = b_init[i_cal];
        b_est = b_init[1:Compparams.nparams-20];
        for j = Compparams.nparams-19+1:i
            push!(b_est,b_init[j])
        end
        lowerbnd,upperbnd = returnBounds(i)
        push!(lb,lowerbnd)
        push!(ub,upperbnd)
        make_closures(b_cal,i_cal,Data,h,quasi_seed,params;multithread) = b_est -> -loglikepr(b_est,b_cal,i_cal,Data,h,quasi_seed,params;multithread)
        nll = make_closures(b_cal,i_cal,InitData,h,quasi_seed, Initparams;multithread=false)
        # res = Optim.optimize(nll, b_est)
        # b_init[1:Compparams.nparams-19]=Optim.minimizer(res)[1:Compparams.nparams-19]
        # b_init[Compparams.nparams-19]=0.0;
        # b_init[Compparams.nparams-18:i]=Optim.minimizer(res)[Compparams.nparams-20:i-1]
        #res = Optim.optimize(nll, b_est, LBFGS())
        #  res = Optim.optimize(nll, lb,ub, b_est, SAMIN(; rt=0.95))
        #res = Optim.optimize(nll, lb,ub, b_est, ParticleSwarm(; n_particles=10))
        #res,obj,conv,details = samin(nll,b_est,lb,ub, rt=0.95)
        #b_init[1:17]=res[1:17]
        #b_init[18]=0.0;
        #b_init[19]=res[18]
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
        b_init[1:Compparams.nparams-20]=xbest(res_CMAE)[1:Compparams.nparams-20]
        b_init[Compparams.nparams-19]=0.0;
        b_init[Compparams.nparams-18:i]=xbest(res_CMAE)[Compparams.nparams-19:i-1]
        
        # P = MinimizationProblem(b_est -> -loglikepr(b_est,b_cal,i_cal,InitData,h,quasi_seed,Initparams;multithread=false), lb, ub)
        # local_method = NLoptLocalMethod(NLopt.LN_BOBYQA)
        # multistart_method = TikTak(100)
        # p = multistart_minimization(multistart_method, local_method, P)
        # p.location, p.value
        # b_init[1:21]=p.location[1:21]
        # b_init[22]=0.0;
        # b_init[23]=p.location[22]
        CSV.write("TestNewIterate0630.csv", Tables.table(b_init'), append=true) 
        #CSV.write("saminResults.csv", Tables.table(conv'), append=true) 
    end
end
    
resParams = parse.(Float64,split(last(eachline("TestNewIterate0630.csv")),","));
SimP = StrucParams(resParams)
AvgSimData, SimData = CreateNewSimData(InitData,quasi_seed,Initparams,SimP;multithread=false)
using Plots
PlotResults(InitData,AvgSimData)


i_cal = [Compparams.nparams-19];
b_cal = [0.0];
igrid = Vector{Integer}(1:Compparams.nparams);              
i_est = deleteat!(igrid,Compparams.nparams-19);
b_est = b_init
b_est = parse.(Float64,split(last(eachline("TestNewIterate0630.csv")),","));
deleteat!(b_est,Compparams.nparams-19);
@run loglikepr(b_est,b_cal,i_cal,InitData,h,quasi_seed,Initparams;multithread=false) 