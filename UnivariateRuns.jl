using ThorinDistributions, DoubleFloats, HypothesisTests, MultiFloats
import Optim, Plots, Random, Distributions, KernelDensity, Serialization, StatsPlots


DoubleType = Double64
setprecision(256)
ArbType = BigFloat


function Experiment(;N = 100000,dist_name,dist,Time_ps = 300,Time_lbfgs = 300,m,n_gammas,seed = 123, N_ks_tests = 100, shift = 0)

    Total_time = Int(floor(Time_lbfgs+Time_ps))

    Random.seed!(seed)
    sample = Array{Float64}(undef, 1,N)
    Random.rand!(dist,sample)
    sample .-= shift
    sample = DoubleType.(sample)
    println("Computing empirical coefs of $dist_name, $N samples (may take some time..)")
    E = ThorinDistributions.empirical_coefs(sample,m)
    println("Done")

    println("Launching ParticleSwarm...")
    par = Random.rand(2n_gammas) .- 1/2
    tol = 0.1^22

    obj = x -> ThorinDistributions.L2Objective(DoubleType.(x),E)

    opt = Optim.Options(g_tol=tol,
                    x_tol=tol,
                    f_tol=tol,
                    time_limit=DoubleType(Time_ps), # 30min
                    show_trace = true,
                    allow_f_increases = true,
                    iterations = 10000000)
    algo = Optim.ParticleSwarm()
    program = Optim.optimize(obj, par, algo, opt; autodiff = :forward)
    print(program)
    par = Optim.minimizer(program)


    println("Polishing with LBFGS...")
    if ArbType != DoubleType
        sample = ArbType.(sample)
        E = ThorinDistributions.empirical_coefs(sample,m)
        par = ArbType.(par)
        tol = ArbType(tol)
    end

    opt2 = Optim.Options(g_tol=tol,
                    x_tol=tol,
                    f_tol=tol,
                    time_limit=ArbType(Time_lbfgs), # 30min
                    show_trace = true,
                    allow_f_increases = true,
                    iterations = 10000000)
    algo2 = Optim.LBFGS()
    program2 = Optim.optimize(obj, par, algo2, opt2)
    print(program2)
    par = Optim.minimizer(program2)

    # Back to double : 
    par = ArbType.(par)
    E = ArbType.(E)

    # Extracting the solution for a plot
    alpha = par[1:n_gammas] .^2 #make them positives
    scales = reshape(par[(n_gammas+1):2n_gammas],(n_gammas,)) .^ 2 # make them positives
    rez = hcat(alpha,scales)
    rez = rez[sortperm(-rez[:,1]),:]
    display(rez)
    coefs = ThorinDistributions.get_coefficients(alpha,scales,m)
    x = ArbType.(0:0.01:7.5)

    model_name = "N$(N)_m$(m[1])_Tpso$(Time_ps)_Tpolish$(Time_lbfgs)"
    true_density = (x) -> convert(Float64,Distributions.pdf(dist,x+shift))
    moschdist = ThorinDistributions.UnivariateGammaConvolution(alpha,scales)
    fMosch = (x) -> convert(Float64,Distributions.pdf(moschdist,x))
    y = KernelDensity.kde(convert.(Float64,sample[1,:]))
    kern = x -> KernelDensity.pdf(y,convert(Float64,x))
    fE = (x)->convert(Float64,ThorinDistributions.laguerre_density(x, E))
    f = (x)->convert(Float64,ThorinDistributions.laguerre_density(x, coefs))
    

    plotMat = hcat(true_density.(x),
                   kern.(x),
                   fE.(x),
                   f.(x))
                   #mosh)
    x = Float64.(x)
    # Let's do some KS testing : 
    new_sample = deepcopy(sample)
    Random.rand!(moschdist,new_sample)
    new_samples = [new_sample for i in 1:N_ks_tests]

    p_values = zeros(N_ks_tests)

    println("Computing KS...")
    Threads.@threads for i in 1:N_ks_tests
        Random.rand!(moschdist,new_samples[i])
        p_values[i] = pvalue(ApproximateTwoSampleKSTest(vec(sample),vec(new_samples[i])))
        print("KS : $i","\n")
    end

   diffMat = deepcopy(plotMat)
   for i in 3:size(diffMat,2)
       diffMat[:,i] -= diffMat[:,1]
   end
   diffMat = diffMat[:,3:end]

    density_label = ["Theoretical $dist_name" "Gaussian kernel of input data" "L_$m estimation" "L_$m projection of the estimated GC"]
    diff_label = ["L_$m estimation" "L_$m projection of the estimated GC"]

    println("Plotting...")
    p0 = Plots.plot(x,plotMat,title = "Densities",label =density_label)
    p1 = Plots.plot(x,diffMat,title = "Difference to the true density",label =diff_label)
    y = ones(3)
    title = Plots.scatter(y,
                            marker=0,
                            markeralpha=0,
                            annotations=(2,
                                         y[2],
                                         Plots.text("Univariate $dist_name, $N samples, $n_gammas gammas, m = $(m[1]), $(Total_time) seconds")),
                            axis=nothing,
                            legend=false,
                            border=:none,
                            size=(200,100))

    new_sample = deepcopy(sample)
    Random.rand!(moschdist,new_sample)
    p2 = StatsPlots.qqplot(Float64.(vec(log.(sample))[1:Int(N//100)]), Float64.(vec(log.(new_sample))[1:Int(N//100)]), qqline = :fit, title="Empirical QQplot")

    p3 = StatsPlots.ea_histogram(p_values,bins=25,legend=nothing,title="KS test: histogram of p-values of $(N_ks_tests) resamples", yaxis=nothing)
                            

    p = Plots.plot(
        title,
        Plots.plot(p0, p2, p1, p3,layout = (2,2)),
        layout=Plots.grid(2,1,heights=[0.01,0.99]),
        size=[1920,1080]
    )
    Plots.display(p)

    # Save stuff :
    if !isdir(dist_name)
        mkdir(dist_name)
    end
    Plots.savefig(p,"$(dist_name)/$model_name.pdf")
    Serialization.serialize("$(dist_name)/$model_name.model",(alpha,scales,p_values))

    print("Experiment finished !")
    return p_values
end

Experiment(; dist_name = "Weibull(1.5,1)", dist = Distributions.Weibull(3/2,1), m = (5,), n_gammas = 2)
Experiment(; dist_name = "Weibull(0.75,1)", dist = Distributions.Weibull(3/4,1), m = (5,), n_gammas = 2)

Experiment(; dist_name = "LogNormal(0,0.83)", dist = Distributions.LogNormal(0,0.83), m = (21,), n_gammas = 10)
Experiment(; dist_name = "Pareto(2.5,1)", dist = Distributions.Pareto(2.5,1), m = (21,), n_gammas = 10, shift = 1)
Experiment(; dist_name = "Pareto(1.5,1)", dist = Distributions.Pareto(1.5,1), m = (21,), n_gammas = 10, shift = 1)
Experiment(; dist_name = "Pareto(1,1)",   dist = Distributions.Pareto(1.0,1), m = (21,), n_gammas = 10, shift = 1)
Experiment(; dist_name = "Pareto(0.5,1)", dist = Distributions.Pareto(0.5,1), m = (21,), n_gammas = 10, shift = 1)
Experiment(; dist_name = "Weibull(1.5,1)", dist = Distributions.Weibull(3/2,1), m = (21,), n_gammas = 10)
Experiment(; dist_name = "Weibull(0.75,1)", dist = Distributions.Weibull(3/4,1), m = (21,), n_gammas = 10)

Experiment(; dist_name = "LogNormal(0,0.83)", dist = Distributions.LogNormal(0,0.83), m = (41,), n_gammas = 20)
Experiment(; dist_name = "Pareto(2.5,1)", dist = Distributions.Pareto(2.5,1), m = (41,), n_gammas = 20, shift = 1)
Experiment(; dist_name = "Pareto(1.5,1)", dist = Distributions.Pareto(1.5,1), m = (41,), n_gammas = 20, shift = 1)
Experiment(; dist_name = "Pareto(1,1)",   dist = Distributions.Pareto(1.0,1), m = (41,), n_gammas = 20, shift = 1)
Experiment(; dist_name = "Pareto(0.5,1)", dist = Distributions.Pareto(0.5,1), m = (41,), n_gammas = 20, shift = 1)
Experiment(; dist_name = "Weibull(1.5,1)", dist = Distributions.Weibull(3/2,1), m = (41,), n_gammas = 20)
Experiment(; dist_name = "Weibull(0.75,1)", dist = Distributions.Weibull(3/4,1), m = (41,), n_gammas = 20)

Experiment(; dist_name = "LogNormal(0,0.83)", dist = Distributions.LogNormal(0,0.83), m = (81,), n_gammas = 40)
Experiment(; dist_name = "Pareto(2.5,1)", dist = Distributions.Pareto(2.5,1), m = (81,), n_gammas = 40, shift = 1)
Experiment(; dist_name = "Pareto(1.5,1)", dist = Distributions.Pareto(1.5,1), m = (81,), n_gammas = 40, shift = 1)
Experiment(; dist_name = "Pareto(1,1)",   dist = Distributions.Pareto(1.0,1), m = (81,), n_gammas = 40, shift = 1)
Experiment(; dist_name = "Pareto(0.5,1)", dist = Distributions.Pareto(0.5,1), m = (81,), n_gammas = 40, shift = 1)
Experiment(; dist_name = "Weibull(1.5,1)", dist = Distributions.Weibull(3/2,1), m = (81,), n_gammas = 40)
Experiment(; dist_name = "Weibull(0.75,1)", dist = Distributions.Weibull(3/4,1), m = (81,), n_gammas = 40)