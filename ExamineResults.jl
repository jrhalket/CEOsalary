@everywhere function CreateNewSimData(Data,quasi_seed,SimC,Simparams;multithread)
    S_sim = Matrix{Float64}(undef,SimC.n_firms,SimC.n_firms);
    S_sim = ComputeSMatrix(SimC,Data.up_data_obs',Data.down_data_obs',Simparams);
    println(size(Simparams.rf))
    if multithread 
        sim_dat = pmap(i -> sim_data_like(Data.up_data_obs', Data.down_data_obs', Simparams, S_sim, i, quasi_seed,SimC), 1:SimC.n_sim, batch_size=cld(SimC.n_sim,nworkers()))
    else 
        sim_dat = map(i -> sim_data_like(Data.up_data_obs', Data.down_data_obs', Simparams, S_sim, i, quasi_seed,SimC), 1:SimC.n_sim)
    end
    

    avg_sim_data= (down_match = zeros(SimC.n_firms,1),
                    wages_match = zeros(SimC.n_firms),
                    measures_match = zeros(SimC.n_firms,1));

    #take averages
    for i in 1:SimC.n_firms
        for j in 1:SimC.n_sim
            avg_sim_data.down_match[i,1] += sim_dat[j].down_match[i,1]/SimC.n_sim
            avg_sim_data.wages_match[i] += sim_dat[j].wages_match[i]/SimC.n_sim
            avg_sim_data.measures_match[i,1] += sim_dat[j].measures_match[i,1]/SimC.n_sim
         
        end
    end

    return avg_sim_data, sim_dat
end

function PlotResults(Data,AvgSimData)
    
    display(scatter(Data.down_data_obs[:,1], AvgSimData.down_match[:,1],
        markersize=2, color=:red))
    display(scatter(Data.wages_obs[:], AvgSimData.wages_match[:],
        markersize=2))
    display(scatter(Data.measures_obs[:,1], AvgSimData.measures_match[:,1],
        markersize=2, color=:red))
end 


