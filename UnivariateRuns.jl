using ThorinDistributions, DoubleFloats
import Optim, Plots, Random, Distributions, KernelDensity, Serialization

function Experiment(;N,dist_name,dist,Time_ps,Time_lbfgs,m,n_gammas,seed, shift)

    Total_time = Int(floor(Time_lbfgs+Time_ps))

    Random.seed!(seed)
    sample = Array{Float64}(undef, 1,N)
    Random.rand!(dist,sample)
    sample .-= shift
    println("Computing empirical coefs of $dist_name, $N samples (may take some time..)")
    E = ThorinDistributions.empirical_coefs(sample,m)
    println("Done")

    println("Launching ParticleSwarm...")
    par = Double64.(Random.rand(2n_gammas) .- 1/2)
    tol = Double64(10)^(-22)

    obj = x -> ThorinDistributions.L2Objective(x,E)

    opt = Optim.Options(g_tol=tol,
                    x_tol=tol,
                    f_tol=tol,
                    time_limit=Double64(Time_ps), # 30min
                    show_trace = true,
                    allow_f_increases = true,
                    iterations = 10000000)
    algo = Optim.ParticleSwarm()
    program = Optim.optimize(obj, par, algo, opt; autodiff = :forward)
    print(program)
    par = Optim.minimizer(program)

    println("Polishing with LBFGS...")
    opt2 = Optim.Options(g_tol=tol,
                    x_tol=tol,
                    f_tol=tol,
                    time_limit=Double64(Time_lbfgs), # 30min
                    show_trace = true,
                    allow_f_increases = true,
                    iterations = 10000000)
    algo2 = Optim.LBFGS()
    program2 = Optim.optimize(obj, par, algo2, opt2)
    print(program2)
    par = Optim.minimizer(program2)

    # Extracting the solution for a plot
    alpha = par[1:n_gammas] .^2 #make them positives
    scales = reshape(par[(n_gammas+1):2n_gammas],(n_gammas,)) .^ 2 # make them positives
    rez = hcat(alpha,scales)
    rez = rez[sortperm(-rez[:,1]),:]
    display(rez)
    coefs = ThorinDistributions.get_coefficients(alpha,scales,m)
    x = 0:0.01:7.5

    model_name = "N$(N)_m$(m[1])_Tpso$(Time_ps)_Tpolish$(Time_lbfgs)"
    true_density = (x) -> Distributions.pdf(dist,x+shift)
    moschdist = ThorinDistributions.UnivariateGammaConvolution(alpha,scales)
    fMosch = (x) -> convert(Float64,Distributions.pdf(moschdist,x))
    y = KernelDensity.kde(convert.(Float64,sample[1,:]))
    kern = x -> KernelDensity.pdf(y,x)
    fE = (x)->convert(Float64,ThorinDistributions.laguerre_density(x, E))
    f = (x)->convert(Float64,ThorinDistributions.laguerre_density(x, coefs))
    plotMat = hcat(true_density.(x),
                   kern.(x),
                   fE.(x),
                   f.(x),
                   fMosch.(x))

   mosho_density = plotMat[:,5]
   replace!(mosho_density, Inf=>0)
   moscho_failed = sum(mosho_density) < N * 10.0^(-3.0)

   diffMat = deepcopy(plotMat)
   for i in 2:size(diffMat,2)
       diffMat[:,i] -= diffMat[:,1]
   end
   diffMat = diffMat[:,2:end]
   if !moscho_failed
       density_label = ["Theoretical $dist_name" "Gaussian kernel of input data" "L_$m estimation" "L_$m projection of the estimated GC" "Moshopoulos Density of the esitmated GC"]
       diff_label = ["Gaussian kernel of input data" "L_$m estimation" "L_$m projection of the estimated GC" "Moshopoulos Density of the esitmated GC"]
       small_diff_label = ["L_$m projection of the estimated GC" "Moshopoulos Density of the esitmated GC"]
       small_density_label = ["Theoretical $dist_name" "Moshopoulos Density of the esitmated GC"]
       p2 = Plots.plot(x,hcat(plotMat[:,1],plotMat[:,5]),title = "Densities",label =small_density_label)
       p3 = Plots.plot(x,diffMat[:,3:4], title = "Diff", label =small_diff_label)
   else
       print("Moschopoulos returned only zeros...\n")
       plotMat = plotMat[:,begin:(end-1)]
       diffMat = diffMat[:,begin:(end-1)]
       density_label = ["Theoretical $dist_name" "Gaussian kernel of input data" "L_$m estimation" "L_$m projection of the estimated GC"]
       diff_label = ["Gaussian kernel of input data" "L_$m estimation" "L_$m projection of the estimated GC"]
       small_density_label = ["Theoretical $dist_name" "L_$m projection of the estimated GC"]
       small_diff_label = ["L_$m projection of the estimated GC"]
       p2 = Plots.plot(x,hcat(plotMat[:,1],plotMat[:,4]),title = "Densities",label =small_density_label)
       p3 = Plots.plot(x,diffMat[:,3:3], title = "Diff", label =small_diff_label)
   end

    p0 = Plots.plot(x,plotMat,title = "Densities",label =density_label)
    p1 = Plots.plot(x,diffMat,title = "Diff",label =diff_label)
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

    p = Plots.plot(
        title,
        Plots.plot(p0, p1, p2, p3,layout = (2,2)),
        layout=Plots.grid(2,1,heights=[0.01,0.99]),
        size=[1920,1080]
    )
    Plots.display(p)


    # Save stuff :
    if !isdir(dist_name)
        mkdir(dist_name)
    end
    Plots.savefig(p,"$(dist_name)/$model_name.pdf")
    Serialization.serialize("$(dist_name)/$model_name.model",(alpha,scales))
    print("Experiment finished !")
    #return (alpha,scales)
end


T_PSO = 12
T_LBFGS = 12



# Some Lognormals
#alpha, scale=
# dist = ThorinDistributions.UnivariateGammaConvolution(alpha, scale)
# sample = Array{Float64}(undef, 1,10000)
# Random.rand!(Distributions.LogNormal(0,0.83),sample)
# fMosch = (x) -> convert.(Float64,Distributions.pdf.(dist,x))
#
# vals = fMosch.(sample)

Experiment(; N = 100000, dist_name = "LogNormal(0,0.83)", dist = Distributions.LogNormal(0,0.83),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (21,), n_gammas = 10, seed = 123, shift = 0)
Experiment(; N = 100000, dist_name = "LogNormal(0,0.83)", dist = Distributions.LogNormal(0,0.83),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (41,), n_gammas = 20, seed = 123, shift = 0)

# Some Pareto
Experiment(; N = 100000, dist_name = "Pareto(2.5,1)", dist = Distributions.Pareto(2.5,1),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (21,), n_gammas = 10, seed = 123, shift = 1)
Experiment(; N = 100000, dist_name = "Pareto(2.5,1)", dist = Distributions.Pareto(2.5,1),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (41,), n_gammas = 20, seed = 123, shift = 1)
# Some Pareto
Experiment(; N = 100000, dist_name = "Pareto(1.5,1)", dist = Distributions.Pareto(1.5,1),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (21,), n_gammas = 10, seed = 123, shift = 1)
Experiment(; N = 100000, dist_name = "Pareto(1.5,1)", dist = Distributions.Pareto(1.5,1),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (41,), n_gammas = 20, seed = 123, shift = 1)

# Some more Pareto
Experiment(; N = 100000, dist_name = "Pareto(1,1)", dist = Distributions.Pareto(1,1),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (21,), n_gammas = 10, seed = 123, shift = 1)
Experiment(; N = 100000, dist_name = "Pareto(1,1)", dist = Distributions.Pareto(1,1),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (41,), n_gammas = 20, seed = 123, shift = 1)

# Some more Pareto
Experiment(; N = 100000, dist_name = "Pareto(0.5,1)", dist = Distributions.Pareto(0.5,1),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (21,), n_gammas = 10, seed = 123, shift = 1)
Experiment(; N = 100000, dist_name = "Pareto(0.5,1)", dist = Distributions.Pareto(0.5,1),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (41,), n_gammas = 20, seed = 123, shift = 1)

# Two experiments to compare to furmann
Experiment(; N = 100000, dist_name = "Weibull(1.5,1)", dist = Distributions.Weibull(3/2,1),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (5,), n_gammas = 2, seed = 123, shift=0)
Experiment(; N = 100000, dist_name = "Weibull(0.75,1)", dist = Distributions.Weibull(3/4,1),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (5,), n_gammas = 2, seed = 123, shift = 0)

# Same, but more realistic
Experiment(; N = 100000, dist_name = "Weibull(1.5,1)", dist = Distributions.Weibull(3/2,1),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (41,), n_gammas = 20, seed = 123, shift = 0)
Experiment(; N = 100000, dist_name = "Weibull(0.75,1)", dist = Distributions.Weibull(3/4,1),
            Time_ps = T_PSO, Time_lbfgs = T_LBFGS, m = (41,), n_gammas = 20, seed = 123, shift = 0)
