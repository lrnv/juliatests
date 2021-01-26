import Random, DatagenCopulaBased, Distributions, ThorinDistributions, Optim, Plots
import Serialization
using Plots, StatsPlots, KernelDensity, StatsBase
setprecision(512)
DoubleType = BigFloat
ArbType = BigFloat


function MultivExperiment(;N=100000,dist_name,Time_ps=3600,Time_lbfgs=1800,n_gammas,m,copula,marginals, shifts, with_regul = false)

    model_name = "N$(N)_m$(m)_n$(n_gammas)_Tpso$(Time_ps)_Tpolish$(Time_lbfgs)"
    
    # Simulate the dataset : 
    Random.seed!(123)
    sample = 1 .- DatagenCopulaBased.simulate_copula(N,copula)
    for i in 1:size(sample,2)
        sample[:,i] = Distributions.quantile.(marginals[i],sample[:,i]) .- shifts[i]
    end
    sample = DoubleType.(transpose(sample))

    # Compute empirical coefs: 
    E = DoubleType.(ThorinDistributions.empirical_coefs(ArbType.(sample),m))

    println("Launching ParticleSwarm...")

    n_par = with_regul ? 3n_gammas+1 : 3n_gammas

    par = DoubleType.((Random.rand(n_par) .- 1/2))
    tol = DoubleType(0.1)^(22)
    if with_regul
        obj = x -> ThorinDistributions.L2ObjectiveWithPenalty(x,E)
    else
        obj = x -> ThorinDistributions.L2Objective(x,E)
    end
    opt = Optim.Options(g_tol=tol,
                    x_tol=tol,
                    f_tol=tol,
                    time_limit=DoubleType(Time_ps), # 30min
                    show_trace = true,
                    allow_f_increases = true,
                    iterations = 10000000)
    algo = Optim.ParticleSwarm()
    program = Optim.optimize(obj, par, algo, opt)
    print(program)
    par = Optim.minimizer(program)

    # Switch to ArbType for the precision run : 
    if ArbType != DoubleType
        par = ArbType.(par)
        tol = ArbType.(tol)
        sample = ArbType.(sample)
        E = ThorinDistributions.empirical_coefs(sample,m)
    end

    println("Polishing with LBFGS...")
    opt2 = Optim.Options(g_tol=tol,
                    x_tol=tol,
                    f_tol=tol,
                    time_limit=ArbType(Time_lbfgs), # 30min
                    show_trace = true,
                    allow_f_increases = true,
                    iterations = 10000000)
    algo2 = Optim.BFGS()
    program2 = Optim.optimize(obj, par, algo2, opt2)
    print(program2)
    par = Optim.minimizer(program2)

    
    # Extracting the solution for a plot
    alpha = par[1:n_gammas] .^2 #make them positives
    scales = reshape(par[(n_gammas+1):3n_gammas],(n_gammas,2)) .^ 2 # make them positives
    rez = hcat(alpha,scales)
    rez = rez[sortperm(-rez[:,1]),:]
    display(rez)


    # Save stuff :
    if !isdir("multiv/$dist_name")
        mkdir("multiv/$dist_name")
    end
    to_serialize = (alpha,scales,sample,N,n_gammas,m, copula, dist_name, marginals, shifts,model_name, E)
    Serialization.serialize("multiv/$dist_name/$model_name.model",to_serialize)

end
function MultivPlot(filename)

    alpha,scales,sample,N,n_gammas,m, copula, dist_name, marginals, shifts,model_name, E = Serialization.deserialize(filename)
    # generate data for plots:
    dist = ThorinDistributions.MultivariateGammaConvolution(BigFloat.(alpha),BigFloat.(scales))
    simu_sample = BigFloat.(deepcopy(sample))
    Random.rand!(dist,simu_sample)

    simu_sample = Float64.(simu_sample)
    sample = Float64.(sample)
    N_plot = N

    log_sample = log.(sample)
    log_simu = log.(simu_sample)

    
    # p1 = marginalkde(log_sample[1,1:N_plot],log_sample[2,1:N_plot];levels=100)
    # p2 = marginalkde(log_simu[1,1:N_plot],log_simu[2,1:N_plot]; levels=100)
    # p = plot(p1,p2,layout=(1,2),size=[1600,900],link=:both)

    # p3 = marginalkde(sample[1,1:N_plot],sample[2,1:N_plot];levels=100)
    # p4 = marginalkde(simu_sample[1,1:N_plot],simu_sample[2,1:N_plot]; levels=100)
    # pp = plot(p3,p4,layout=(1,2),size=[1600,900],link=:both)

    # p5 = scatter(log_sample[1,1:N_plot],log_sample[2,1:N_plot])
    # p6 = scatter(log_simu[1,1:N_plot],log_simu[2,1:N_plot])
    # ppp = plot(p5,p6,layout=(1,2),size=[1600,900],link=:both)

    # tpl = (2, n_gammas)
    # x = y = 0:0.5:10
    # coefs = ThorinDistributions.get_coefficients(alpha,scales,m)
    # fE = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(ArbType,x), convert(ArbType,y)], E))
    # g = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(ArbType,x), convert(ArbType,y)], coefs))
    # pppp1 = Plots.plot(x, y, fE, legend=false, title = "Projection on L_$m", seriestype=:wireframe)
    # pppp2 = Plots.plot(x, y, g, legend=false, title = "Estimation in G_$tpl", seriestype=:wireframe)
    # pppp = Plots.plot(pppp1,pppp2, layout = (1,2), size=[1600,900])


    sample_cop = transpose([ordinalrank(sample[1,:]) ordinalrank(sample[2,:])]/(N+1))
    simu_cop = transpose([ordinalrank(simu_sample[1,:]) ordinalrank(simu_sample[2,:])]/(N+1))
    new_p2 = qqplot(log_sample[1,1:N_plot],log_simu[1,1:N_plot],qqline = :R, title = "Qqplot (log-scale) of the first marginal", left_margin = 10Plots.mm)
    new_q2 = qqplot(log_sample[2,1:N_plot],log_simu[2,1:N_plot],qqline = :R, title = "Qqplot (log-scale) of the second marginal")
    new_p3 = plot(kde(transpose(sample_cop)), title="KDE of original copula")
    new_q3 = plot(kde(transpose(simu_cop)), title="KDE of estimated copula", left_margin = 10Plots.mm)
    y = ones(3)
    title = Plots.scatter(y,
                            marker=0,
                            markeralpha=0,
                            annotations=(2,
                                         y[2],
                                         Plots.text("Convolution of $n_gammas bivariates gammas fitted on $N samples of a $dist_name (m = $(m))")),
                            axis=nothing,
                            legend=false,
                            border=:none,
                            size=(200,100))            

    new_p = plot(
        title,
        plot(new_p2,new_p3,new_q3,new_q2,layout=(2,2)),
        layout=Plots.grid(2,1,heights=[0.01,0.99]),
        size=[1600,900]
    )

    # Save stuff :
    # Plots.savefig(p,"multiv/$dist_name/$(model_name)_log.pdf") # Beautifull 
    # Plots.savefig(pp,"multiv/$dist_name/$(model_name).pdf") # -> Useless because does not render correctly. 
    # Plots.savefig(ppp,"multiv/$dist_name/$(model_name)_log_scatter.pdf") # Not very pleasant, because points go wild on the extremals...
    # Plots.savefig(pppp,"multiv/$dist_name/$(model_name)_density.png") # Very nice plot if m is big enough. 
    Plots.savefig(new_p,"multiv/$dist_name/$(model_name).pdf") # Beautifull, lacks a title. Augment the number of levels in the KDE ? 
end
function PlotAllMultiv(folder="multiv/")
    for (root, dirs, files) in walkdir(folder)
        for file in files
            path = joinpath(root, file) # path to files
            if split(path,".")[end] == "model"
                println("Plotting the model: $path")
                MultivPlot(path)
            end
        end
    end
end

# Copulas
clayton7 = DatagenCopulaBased.Clayton_cop(2,7.0)
gauss05 = DatagenCopulaBased.Gaussian_cop([1. 0.5; 0.5 1.])

# Marginals
Ln083 = Distributions.LogNormal(0,0.83)
Ln01 = Distributions.LogNormal(0,1)
Pa25 = Distributions.Pareto(2.5,1)
Pa1 = Distributions.Pareto(1,1)

MultivExperiment(;dist_name="Clayton(7)_Par(1,1)_LN(0,0.83)",n_gammas=10,m=(10,10),copula=clayton7,marginals=[Pa1, Ln083], shifts=[1,0])
MultivExperiment(;dist_name="Clayton(7)_Par(1,1)_LN(0,0.83)",n_gammas=20,m=(10,10),copula=clayton7,marginals=[Pa1, Ln083], shifts=[1,0])
MultivExperiment(;dist_name="Clayton(7)_Par(1,1)_LN(0,0.83)",n_gammas=20,m=(20,20),copula=clayton7,marginals=[Pa1, Ln083], shifts=[1,0])

MultivExperiment(;dist_name="Clayton(7)_LN(0,0.83)_Par(1,1)",n_gammas=10,m=(10,10),copula=clayton7,marginals=[Ln083, Pa1], shifts=[0,1])
MultivExperiment(;dist_name="Clayton(7)_LN(0,0.83)_Par(1,1)",n_gammas=20,m=(10,10),copula=clayton7,marginals=[Ln083, Pa1], shifts=[0,1])
MultivExperiment(;dist_name="Clayton(7)_LN(0,0.83)_Par(1,1)",n_gammas=20,m=(20,20),copula=clayton7,marginals=[Ln083, Pa1], shifts=[0,1])

MultivExperiment(;dist_name="Clayton(7)_Par(2.5,1)_LN(0,0.83)",n_gammas=10,m=(10,10),copula=clayton7,marginals=[Pa25, Ln083], shifts=[1,0])
MultivExperiment(;dist_name="Clayton(7)_Par(2.5,1)_LN(0,0.83)",n_gammas=20,m=(10,10),copula=clayton7,marginals=[Pa25, Ln083], shifts=[1,0])
MultivExperiment(;dist_name="Clayton(7)_Par(2.5,1)_LN(0,0.83)",n_gammas=20,m=(20,20),copula=clayton7,marginals=[Pa25, Ln083], shifts=[1,0])

MultivExperiment(;dist_name="MLN(0.5)_LN(0,1)_LN(0,1)",n_gammas=10,m=(10,10),copula=gauss05,marginals=[Ln01,Ln01], shifts=[0,0])
MultivExperiment(;dist_name="MLN(0.5)_LN(0,1)_LN(0,1)",n_gammas=20,m=(20,20),copula=gauss05,marginals=[Ln01,Ln01], shifts=[0,0])
MultivExperiment(;dist_name="MLN(0.5)_LN(0,1)_LN(0,1)",n_gammas=20,m=(20,20),copula=gauss05,marginals=[Ln01,Ln01], shifts=[0,0])

PlotAllMultiv()
