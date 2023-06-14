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
b_true = rand(Compparams.nparams)
b_true[67] = log(mean(InitData.wages_obs));
bgrid = -1:.02:1;
llgrid = zeros(Compparams.nparams,length(bgrid));
for i in eachindex(b_true)
    igrid = Vector{Integer}(1:length(b_true));
    ical = deleteat!(igrid,i);
    b_cal = b_true[ical];
    for j in eachindex(bgrid)
        b_est = [b_true[i]+bgrid[j]];
        llgrid[i,j] = loglikepr(b_est,b_cal,ical,InitData,h,quasi_seed, Initparams;multithread=false)
    end
end

