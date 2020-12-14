import Optim, Plots, Random, Distributions, LineSearches, KernelDensity
import ThorinDistributions
# The multivariate lognormal test case:
Random.seed!(123)
N = 10000
mu = [0,2]
rho = 0.25
sig = [1 rho; rho 1]
dist = Distributions.MvLogNormal(Distributions.MvNormal(mu, sig))
sample = Array{Float64}(undef, 2,N)
Random.rand!(dist,sample)
sample = convert.(BigFloat,sample)
m = (10,10)
println("Computing empirical coefs of standard MVLN(mu = $mu, rho = $(sig[1,2])), $N samples (may take some time..)")
E = ThorinDistributions.empirical_coefs(sample,m)
println("Done")


println("Launching ParticleSwarm...")
n_gammas = 10
x0 = big.((Random.rand(3n_gammas) .- 1/2))
obj = x -> ThorinDistributions.L2Objective(x,E)
opt = Optim.Options(g_tol=big(1e-22),
                x_tol=big(1e-22),
                f_tol=big(1e-22),
                time_limit=big(15.0), # half hour
                show_trace = true,
                allow_f_increases = true,
                iterations = 100000)
algo = Optim.ParticleSwarm()
program = Optim.optimize(obj, x0, algo, opt; autodiff = :forward)
print(program)
par = Optim.minimizer(program)

println("Polishing with LBFGS...")
opt2 = Optim.Options(g_tol=big(1e-22),
                x_tol=big(1e-22),
                f_tol=big(1e-22),
                time_limit=big(15.0), # 15mins.
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

coefs = ThorinDistributions.get_coefficients(alpha,rates,m)
x = y = 0:0.5:10
true_density = (x,y) -> Distributions.pdf(dist,[x,y])

tpl = (2, n_gammas)
fE = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(BigFloat,x), convert(BigFloat,y)], E))
g = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(BigFloat,x), convert(BigFloat,y)], coefs))
p0 = Plots.plot(x,y,true_density,legend=false,title="True MLN(μ=0, σ= 1, ρ=$rho)",seriestype=:wireframe)
p1 = Plots.plot(x, y, fE, legend=false, title = "Projection on L_$m", seriestype=:wireframe)
p2 = Plots.plot(x, y, g, legend=false, title = "Estimation in G_$tpl", seriestype=:wireframe)
p = Plots.plot(p0,p1,p2, layout = (1,3), size=[1920,1024])
Plots.display(p)

Plots.savefig(p,"MLNTest_$m_2.png")

import Serialization
Serialization.serialize("MLNTest_$m.model",(alpha,rates))
