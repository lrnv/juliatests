import Random, DatagenCopulaBased, Distributions, ThorinDistributions, Optim, Plots
import Serialization
using Plots, StatsPlots, KernelDensity
setprecision(512)
DoubleType = BigFloat
ArbType = BigFloat

function MultivariateTest(;N=100000,dist_name,Time_ps=7200,Time_lbfgs=7200,n_gammas,m,copula,marginals, shifts, m_plot=(50,50))

    model_name = "N$(N)_m$(m)_Tpso$(Time_ps)_Tpolish$(Time_lbfgs)"
    
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
    par = DoubleType.((Random.rand(3n_gammas) .- 1/2))
    tol = DoubleType(0.1)^(22)
    obj = x -> ThorinDistributions.L2Objective(x,E)
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
    
    # generate data for plots:
    dist = ThorinDistributions.MultivariateGammaConvolution(BigFloat.(alpha),BigFloat.(scales))
    simu_sample = BigFloat.(deepcopy(sample))
    Random.rand!(dist,simu_sample)

    simu_sample = Float64.(simu_sample)
    sample = Float64.(sample)
    N_plot = N

    log_sample = log.(sample)
    log_simu = log.(simu_sample)

    
    p1 = marginalkde(log_sample[1,1:N_plot],log_sample[2,1:N_plot];levels=100)
    p2 = marginalkde(log_simu[1,1:N_plot],log_simu[2,1:N_plot]; levels=100)
    p = plot(p1,p2,layout=(1,2),size=[1920,1024],link=:both)

    p3 = marginalkde(sample[1,1:N_plot],sample[2,1:N_plot];levels=100)
    p4 = marginalkde(simu_sample[1,1:N_plot],simu_sample[2,1:N_plot]; levels=100)
    pp = plot(p3,p4,layout=(1,2),size=[1920,1024],link=:both)

    p5 = scatter(log_sample[1,1:N_plot],log_sample[2,1:N_plot])
    p6 = scatter(log_simu[1,1:N_plot],log_simu[2,1:N_plot])
    ppp = plot(p5,p6,layout=(1,2),size=[1920,1024],link=:both)

    tpl = (2, n_gammas)
    E_plot = DoubleType.(ThorinDistributions.empirical_coefs(ArbType.(sample),m_plot))
    coefs_plot = ThorinDistributions.get_coefficients(alpha,scales,m_plot)
    x = y = 0:0.5:10
    fE = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(ArbType,x), convert(ArbType,y)], E_plot))
    g = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(ArbType,x), convert(ArbType,y)], coefs_plot))
    pppp1 = Plots.plot(x, y, fE, legend=false, title = "Projection on L_$m", seriestype=:wireframe)
    pppp2 = Plots.plot(x, y, g, legend=false, title = "Estimation in G_$tpl", seriestype=:wireframe)
    pppp = Plots.plot(pppp1,pppp2, layout = (1,2), size=[1920,1024])
    
    # Save stuff :
    if !isdir(dist_name)
        mkdir(dist_name)
    end

    Plots.savefig(p,"$dist_name/$(model_name)_log.pdf")
    Plots.savefig(pp,"$dist_name/$(model_name).pdf")
    Plots.savefig(ppp,"$dist_name/$(model_name)_log_scatter.pdf")
    Plots.savefig(pppp,"$dist_name/$(model_name)_density.png")

    Serialization.serialize("$dist_name/$model_name.model",(alpha,scales))

end


# Copulas
clayton7 = DatagenCopulaBased.Clayton_cop(2,7.0)
gauss05 = DatagenCopulaBased.Gaussian_cop([1. 0.5; 0.5 1.])
Ln083 = Distributions.LogNormal(0,0.83)
Ln01 = Distributions.LogNormal(0,1)
Pa25 = Distributions.Pareto(2.5,1)
Pa1 = Distributions.Pareto(1,1)

MultivariateTest(;dist_name="Clayton(7)_Par(1,1)_LN(0,0.83)",n_gammas=10,m=(10,10),copula=clayton7,marginals=[Pa1, Ln083], shifts=[1,0])
MultivariateTest(;dist_name="Clayton(7)_Par(1,1)_LN(0,0.83)",n_gammas=20,m=(10,10),copula=clayton7,marginals=[Pa1, Ln083], shifts=[1,0])
MultivariateTest(;dist_name="Clayton(7)_Par(1,1)_LN(0,0.83)",n_gammas=20,m=(20,20),copula=clayton7,marginals=[Pa1, Ln083], shifts=[1,0])

MultivariateTest(;dist_name="Clayton(7)_Par(2.5,1)_LN(0,0.83)",n_gammas=10,m=(10,10),copula=clayton7,marginals=[Pa25, Ln083], shifts=[1,0])
MultivariateTest(;dist_name="Clayton(7)_Par(2.5,1)_LN(0,0.83)",n_gammas=20,m=(10,10),copula=clayton7,marginals=[Pa25, Ln083], shifts=[1,0])
MultivariateTest(;dist_name="Clayton(7)_Par(2.5,1)_LN(0,0.83)",n_gammas=20,m=(20,20),copula=clayton7,marginals=[Pa25, Ln083], shifts=[1,0])

MultivariateTest(;dist_name="MLN(0.5)_LN(0,1)_LN(0,1)",n_gammas=10,m=(10,10),copula=gauss05,marginals=[Ln01,Ln01], shifts=[0,0])
MultivariateTest(;dist_name="MLN(0.5)_LN(0,1)_LN(0,1)",n_gammas=20,m=(20,20),copula=gauss05,marginals=[Ln01,Ln01], shifts=[0,0])
MultivariateTest(;dist_name="MLN(0.5)_LN(0,1)_LN(0,1)",n_gammas=20,m=(20,20),copula=gauss05,marginals=[Ln01,Ln01], shifts=[0,0])





