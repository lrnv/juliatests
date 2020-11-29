import Random, DatagenCopulaBased, Distributions, ThorinDistributions, Optim, LineSearches, Plots

Random.seed!(123)
N = 10000
sample = DatagenCopulaBased.simulate_copula(N,DatagenCopulaBased.Clayton_cop(2,7.0))
kendall = DatagenCopulaBased.corkendall(sample)
spearman = DatagenCopulaBased.corspearman(sample)


# add some marginals:
marginals = [
Distributions.Pareto(1,1),
Distributions.LogNormal(0,0.83),
]
# compute quantiles :
for i in size(sample,2)
    sample[:,i] = Distributions.quantile.(marginals[i],sample[:,i])
end

sample = transpose(sample)


n_gammas = 20
m = Int.(ThorinDistributions.minimum_m(n_gammas,2))
m = (15,15)
E = ThorinDistributions.empirical_coefs(sample,m)


# Now we can launch the main algorithm :
println("Launching ParticleSwarm...")
par = ((Random.rand(3n_gammas) .- 1/2))
obj = x -> ThorinDistributions.L2Objective(x,E)
opt = Optim.Options(g_tol=1e-22,
                x_tol=1e-22,
                f_tol=1e-22,
                time_limit=3600.0, # half hour
                show_trace = true,
                allow_f_increases = true,
                iterations = 100000)
algo = Optim.ParticleSwarm()
program = Optim.optimize(obj, par, algo, opt; autodiff = :forward)
print(program)
par = Optim.minimizer(program)

println("Polishing with LBFGS...")
opt2 = Optim.Options(g_tol=1e-22,
                x_tol=1e-22,
                f_tol=1e-22,
                time_limit=3600.0, # 15mins.
                show_trace = true,
                allow_f_increases = false,
                iterations = 100000)
algo2 = Optim.LBFGS(linesearch = LineSearches.MoreThuente())
program2 = Optim.optimize(obj, par, algo2, opt2; autodiff = :forward)
print(program2)
par = Optim.minimizer(program2)

# Extracting the solution for a plot
alpha = par[1:n_gammas] .^2 #make them positives
rates = reshape(par[(n_gammas+1):3n_gammas],(n_gammas,2)) .^ 2 # make them positives
rez = hcat(alpha,rates)
rez = rez[sortperm(-rez[:,1]),:]
display(rez)


m2 = (20,20)
coefs = ThorinDistributions.get_coefficients(alpha,rates,m2)
E = ThorinDistributions.empirical_coefs(sample,m2)
x = y = 0:0.5:10
true_density = (x,y) -> Distributions.pdf(dist,[x,y])

tpl = (2, n_gammas)
fE = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(BigFloat,x), convert(BigFloat,y)], E))
g = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(BigFloat,x), convert(BigFloat,y)], coefs))
p1 = Plots.plot(x, y, fE, legend=false, title = "Projection on L_$m2", seriestype=:wireframe)
p2 = Plots.plot(x, y, g, legend=false, title = "Estimation in G_$tpl", seriestype=:wireframe)
p = Plots.plot(p1,p2, layout = (1,2), size=[1920,1024])
Plots.display(p)

Plots.savefig(p,"ClaytonParetoLN_m$(m)_n$n.png")

import Serialization
Serialization.serialize("ClaytonParetoLN_m$(m)_n$n.model",(alpha,rates))
