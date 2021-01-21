using ThorinDistributions, HypothesisTests
import Optim, Plots, Random, Distributions, KernelDensity, Serialization, StatsPlots

setprecision(512)
DoubleType = BigFloat
ArbType = BigFloat

function build_sample(N,dist,shift,m)
    sample = Array{Float64}(undef, 1,N)
    Random.rand!(dist,sample)
    sample .-= shift
    sample = DoubleType.(sample)
    E = ThorinDistributions.empirical_coefs(sample,m)
    return (sample,E)
end
function UnivExperiment(;N = 100000,dist_name,dist,Time_ps = 600,Time_lbfgs = 600,m,n_gammas,seed = 123, shift = 0)

    Total_time = Int(floor(Time_lbfgs+Time_ps))

    Random.seed!(seed)
    sample,E = build_sample(N,dist,shift,m)

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


    # Now that we have the solution, we need to save the model, to be able to reload it later to plot it. 
    # Save stuff :
    model_name = "N$(N)_m$(m[1])_Tpso$(Time_ps)_Tpolish$(Time_lbfgs)"
    model = (model_name,alpha,scales,N,dist_name,dist,Time_ps,Time_lbfgs,m,n_gammas,seed,shift,sample,E)

    
    # Save stuff :
    if !isdir("univ/$dist_name")
        mkdir("univ/$(dist_name)")
    end

    Serialization.serialize("univ/$(dist_name)/$model_name.model",model)
    return nothing
end
function UnivPlot(filename;N_ks_tests = 250)
    model_name,alpha,scales,N,dist_name,dist,Time_ps,Time_lbfgs,m,n_gammas,seed,shift,sample,E = Serialization.deserialize(filename)

    #sample,E = build_sample(N,seed,dist,shift,m)

    coefs = ThorinDistributions.get_coefficients(alpha,scales,m)
    x = ArbType.(0:0.01:7.5)

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
    new_sample = sample = Array{BigFloat}(undef, 1,Int(N//10))
    Random.rand!(moschdist,new_sample)
    new_samples = [new_sample for i in 1:N_ks_tests]

    p_values = zeros(N_ks_tests)

    println("Computing KS...")
    Threads.@threads for i in 1:N_ks_tests
        Random.rand!(moschdist,new_samples[i])
        p_values[i] = pvalue(ExactOneSampleKSTest(vec(new_samples[i]) .+ shift,dist))
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
                                         Plots.text("Univariate $dist_name, $N samples, $n_gammas gammas, m = $(m[1])")),
                                         #Plots.text("Univariate $dist_name, $N samples, $n_gammas gammas, m = $(m[1]), $(Total_time) seconds")),
                            axis=nothing,
                            legend=false,
                            border=:none,
                            size=(200,100))

    new_sample = deepcopy(sample)
    Random.rand!(moschdist,new_sample)
    p2 = StatsPlots.qqplot(Float64.(vec(log.(sample))), Float64.(vec(log.(new_sample))), qqline = :fit, title="Empirical QQplot")

    p3 = StatsPlots.histogram(p_values,bins=20,legend=nothing,title="KS test: histogram of p-values of $(N_ks_tests) resamples", yaxis=nothing)
                            

    p = Plots.plot(
        title,
        Plots.plot(p0, p2, p1, p3,layout = (2,2)),
        layout=Plots.grid(2,1,heights=[0.01,0.99]),
        size=[1920,1080]
    )
    Plots.display(p)

    Plots.savefig(p,"univ/$(dist_name)/$model_name.pdf")
    return nothing
end
function PlotAllUniv(folder="univ/",N_ks_tests=250)
    for (root, dirs, files) in walkdir(folder)
        for file in files
            path = joinpath(root, file) # path to files
            if split(path,".")[end] == "model"
                println("Plotting the model: $path")
                UnivPlot(path;N_ks_tests)
            end
        end
    end
end

# Distributions: 
Weib15 = Distributions.Weibull(3/2,1)
Weib75 = Distributions.Weibull(3/4,1)
Pa05 = Distributions.Pareto(0.5,1)
Pa10 = Distributions.Pareto(1.0,1)
Pa15 = Distributions.Pareto(1.5,1)
Pa25 = Distributions.Pareto(2.5,1)
Ln = Distributions.LogNormal(0,0.83)


UnivExperiment(; dist_name = "Weibull(1.5,1)", dist = Weib15, m = (5,), n_gammas = 2)
UnivExperiment(; dist_name = "Weibull(1.5,1)", dist = Weib15, m = (21,), n_gammas = 10)
UnivExperiment(; dist_name = "Weibull(1.5,1)", dist = Weib15, m = (41,), n_gammas = 20)
UnivExperiment(; dist_name = "Weibull(1.5,1)", dist = Weib15, m = (81,), n_gammas = 40)

UnivExperiment(; dist_name = "LogNormal(0,0.83)", dist = Ln, m = (21,), n_gammas = 10)
UnivExperiment(; dist_name = "LogNormal(0,0.83)", dist = Ln, m = (41,), n_gammas = 20)
UnivExperiment(; dist_name = "LogNormal(0,0.83)", dist = Ln, m = (81,), n_gammas = 40)

UnivExperiment(; dist_name = "Weibull(0.75,1)", dist = Weib75, m = (5,), n_gammas = 2)
UnivExperiment(; dist_name = "Weibull(0.75,1)", dist = Weib75, m = (21,), n_gammas = 10)
UnivExperiment(; dist_name = "Weibull(0.75,1)", dist = Weib75, m = (41,), n_gammas = 20)
UnivExperiment(; dist_name = "Weibull(0.75,1)", dist = Weib75, m = (81,), n_gammas = 40)

UnivExperiment(; dist_name = "Pareto(0.5,1)", dist = Pa05, m = (21,), n_gammas = 10, shift = 1)
UnivExperiment(; dist_name = "Pareto(0.5,1)", dist = Pa05, m = (41,), n_gammas = 20, shift = 1)
UnivExperiment(; dist_name = "Pareto(0.5,1)", dist = Pa05, m = (81,), n_gammas = 40, shift = 1)

UnivExperiment(; dist_name = "Pareto(1,1)",   dist = Pa10, m = (21,), n_gammas = 10, shift = 1)
UnivExperiment(; dist_name = "Pareto(1,1)",   dist = Pa10, m = (41,), n_gammas = 20, shift = 1)
UnivExperiment(; dist_name = "Pareto(1,1)",   dist = Pa10, m = (81,), n_gammas = 40, shift = 1)

UnivExperiment(; dist_name = "Pareto(1.5,1)", dist = Pa15, m = (21,), n_gammas = 10, shift = 1)
UnivExperiment(; dist_name = "Pareto(1.5,1)", dist = Pa15, m = (41,), n_gammas = 20, shift = 1)
UnivExperiment(; dist_name = "Pareto(1.5,1)", dist = Pa15, m = (81,), n_gammas = 40, shift = 1)

UnivExperiment(; dist_name = "Pareto(2.5,1)", dist = Pa25, m = (21,), n_gammas = 10, shift = 1)
UnivExperiment(; dist_name = "Pareto(2.5,1)", dist = Pa25, m = (41,), n_gammas = 20, shift = 1)
UnivExperiment(; dist_name = "Pareto(2.5,1)", dist = Pa25, m = (81,), n_gammas = 40, shift = 1)


# Plot everything : 


PlotAllUniv()