using ThorinDistributions, Distributions, DoubleExponentialFormulas, LinearAlgebra, PolynomialRoots, HypothesisTests, StatsPlots, Serialization, CSV, Tables
import Random, Plots, Optim

setprecision(2048)
qde = QuadDE(BigFloat)

function compute_g(dist,n)
    g = Array{BigFloat}(undef, 1,2n+1)
    residuals_g = deepcopy(g)
    for i in 0:(2n)
        g[i+1],residuals_g[i+1] = qde(x -> (-x)^(i) * pdf(dist,x) * exp(-x), 0, +Inf)
        print("g_{",i,"} = ",Float64(g[i+1]),", rez = ",Float64(residuals_g[i+1]),"\n")
    end
    return g
end
function MFK_Projection(g_integrals,n_gammas)

    s = Array{BigFloat}(undef, 2n_gammas)
    s[1] = -g_integrals[2]/g_integrals[1]
    for k in 1:(length(s)-1)
        s[k+1] = g_integrals[k+2] / factorial(big(k))
        for i in 0:(k-1)
            s[k+1] += s[i+1] * g_integrals[k-i+1] / factorial(big(k-i))
        end
        s[k+1] = - s[k+1]/g_integrals[1]
    end

    S = Array{BigFloat}(undef, n_gammas,n_gammas)
    for i in 0:(n_gammas-1)
        for j in 0:(n_gammas-1)
            S[i+1,j+1] = s[i+j+1]
        end
    end

    sol_b = Symmetric(S) \ (-s[(n_gammas+1):end])
    b = deepcopy(sol_b)
    append!(b,1)
    b = reverse(b)
    b_deriv = reverse(sol_b) .* (1:n_gammas)

    a = Array{BigFloat}(undef, n_gammas)
    a[1] = s[1]
    for k in 1:(n_gammas-1)
        a[k+1] = s[k+1]
        for i in 0:(k-1)
            a[k+1] = a[k+1] + b[i+2] * s[k-i]
        end
    end

    z = real.(roots(b, polish=true))

    beta = -z .-1
    alpha = deepcopy(beta)

    for i in 1:length(alpha)
        rez_num = 0
        rez_denom = 0
        for k in 1:length(a)
            rez_num += a[k] * z[i]^(k-1)
        end
        for k in 1:length(b_deriv)
            rez_denom += b_deriv[k] * z[i]^(k-1)
        end
        alpha[i] = rez_num/rez_denom
    end

    return ThorinDistributions.UnivariateGammaConvolution(Float64.(alpha), Float64.(1 ./ beta))
end
function E_from_g(g)
    # this function should compute laguerre coefficients from g. 
    a = deepcopy(g)
    for i in 1:length(a)
        a[i] = sqrt(2) * sum(binomial(big(i-1),big(k))*big(2)^(k)/factorial(big(k)) * g[k+1] for k in 0:(i-1))
    end
    return a
end
function Fit_my_model(E,n_gammas;t1=10,t2=600,tol=big(0.1^22))
    par = big.(Random.rand(2n_gammas) .- 1/2)
    obj = x -> ThorinDistributions.L2Objective(x,E)
    program = Optim.optimize(obj, 
                            par, 
                            Optim.ParticleSwarm(), 
                            Optim.Options(g_tol=tol,
                                        x_tol=tol,
                                        f_tol=tol,
                                        time_limit=BigFloat(t1), # 30min
                                        show_trace = true,
                                        allow_f_increases = true,
                                        iterations = 10000000);
                            autodiff = :forward)
    program2 = Optim.optimize(obj, 
                            Optim.minimizer(program),
                            Optim.LBFGS(), 
                            Optim.Options(g_tol=tol,
                                            x_tol=tol,
                                            f_tol=tol,
                                            time_limit=BigFloat(t2), # 30min
                                            show_trace = true,
                                            allow_f_increases = true,
                                            iterations = 10000000))
    par = Optim.minimizer(program2)
    alpha = par[1:n_gammas] .^2 #make them positives
    scales = reshape(par[(n_gammas+1):2n_gammas],(n_gammas,)) .^ 2 # make them positives

    return ThorinDistributions.UnivariateGammaConvolution(Float64.(alpha),Float64.(scales))
end

My_Dist_BF = Distributions.LogNormal(big(0),big(83)/big(100))
My_Dist = Distributions.LogNormal(0,0.83)
different_ns = [2,3,4,5,10,15,20,30,40] # n at wich we compute the model
max_n = maximum(different_ns)
n_repeats=100 # number of samples for each KS distance
N_simu = 100000 # Number of simulations for KS distances. 

g = compute_g(My_Dist_BF,max_n)
E = E_from_g(g)

models_furman = []
models_me = []
for n in different_ns
    print("Furman...\n")
    append!(models_furman,[MFK_Projection(g[1:(2n+1)],n)])
    print("Optim...\n")
    append!(models_me,[Fit_my_model(E[1:(2n+1)], n; t1 = 200*n)])
end

# COmpute KS distances and plot everything : 
values_n = repeat(different_ns,n_repeats)
values_ks_furman = BigFloat.(deepcopy(values_n))
values_ks_me = deepcopy(values_ks_furman)
id_models = deepcopy(values_n)
Threads.@threads for id in 1:length(values_n)
    id_models[id] = findall(x -> x == values_n[id], different_ns)[1]
    values_ks_furman[id] = ExactOneSampleKSTest(vec(Random.rand(models_furman[id_models[id]],N_simu)),My_Dist).δ
    values_ks_me[id] = ExactOneSampleKSTest(vec(Random.rand(models_me[id_models[id]],N_simu)),My_Dist).δ
    println(id)
end



model = (My_Dist,different_ns,max_n,n_repeats,N_simu,g,E,models_furman,models_me,values_n,values_ks_furman,values_ks_me)
Serialization.serialize("furman/LnApprox.model",model)

p = violin(values_n, values_ks_furman, side=:left, linewidth=0, label="Miles, Furman & Kuxnetsov")
p = violin!(values_n, values_ks_me, side=:right, linewidth=0, label="Laguerre")
# Plots.savefig(p,"FurmanViolin.pdf")
# p = boxplot(values_n, values_ks_furman, linewidth=1, label="Miles, Furman & Kuxnetsov", fillalpha=0.50, side=:right)
# p = boxplot!(values_n, values_ks_me, linewidth=1, label="Laguerre", fillalpha=0.50, side=:left)
y = ones(3)
title = Plots.scatter(y,marker=0,markeralpha=0,annotations=(2,y[2],
                      Plots.text("KS distances to a LN(0,0.83), for different number of gammas, on $n_repeats resamples")),
                      axis=nothing,legend=false,border=:none,size=(200,100))
p = Plots.plot(title,p,layout=Plots.grid(2,1,heights=[0.01,0.99]),size=[1600,900])
Plots.savefig(p,"furman/LnViolin.pdf")


# Ok we also want to export a table ith the parameters for the first ones. 
par_fur = map(x -> [x.α x.θ][sortperm(x.α),:], models_furman)
par_me = map(x -> [x.α x.θ][sortperm(x.α),:], models_me)
rez = Array{Float64}(undef,0,5)
for i in 1:length(different_ns)
    rez = vcat(rez,hcat(repeat([different_ns[i]],different_ns[i]),par_fur[i],par_me[i]))
end
CSV.write("furman/Furman_Ln_parameters_compare.csv",Tables.table(rez))



# Now we need to do the same thing for a weibull

Weib_BF = Distributions.Weibull(big(3)/big(2),big(1))
Weib = Distributions.Weibull(3/2,1)

g = compute_g(Weib_BF,max_n)
E = E_from_g(g)

models_weibull = []
for n in different_ns
    append!(models_weibull,[Fit_my_model(E[1:(2n+1)], n; t1 = 200*n)])
end

# COmpute KS distances and plot everything : 
values_ks_weibull = BigFloat.(deepcopy(values_n))
Threads.@threads for id in 1:length(values_n)
    values_ks_weibull[id] = ExactOneSampleKSTest(vec(Random.rand(models_weibull[id_models[id]],N_simu)),Weib).δ
end


p = boxplot(values_n, values_ks_weibull, linewidth=1, label="Miles, Furman & Kuxnetsov", fillalpha=0.50, side=:right)
y = ones(3)
title = Plots.scatter(y,marker=0,markeralpha=0,annotations=(2,y[2],
                      Plots.text("KS distances to a Weibull(1.5,1), for different number of gammas, on $n_repeats resamples")),
                      axis=nothing,legend=false,border=:none,size=(200,100))
p = Plots.plot(title,p,layout=Plots.grid(2,1,heights=[0.01,0.99]),size=[1600,900])
Plots.savefig(p,"furman/WeibBoxplot.pdf")
