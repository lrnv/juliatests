using ThorinDistributions
import Optim, Plots, Random, Distributions, LineSearches, KernelDensity
# The multivariate lognormal test case:
Random.seed!(12)
N = 100000
sigma = BigFloat("0.83")
dist = Distributions.LogNormal(0,BigFloat(sigma))
sample = Array{BigFloat}(undef, 1,N)
Random.rand!(dist,sample)
m = (80,)
println("Computing empirical coefs of standard Log normal, $N samples (may take some time..)")
E = ThorinDistributions.empirical_coefs(sample,m)
display(transpose(E))
println("Done")


println("Launching ParticleSwarm...")
n_gammas = 10

par = big.((Random.rand(2n_gammas) .- 1/2))

#Testing with 5 m only :
obj = x -> MGCLaguerre.L2Objective(x,E)
opt = Optim.Options(g_tol=big(1e-22),
                x_tol=big(1e-22),
                f_tol=big(1e-22),
                time_limit=big(3600.0), # 30 min
                show_trace = true,
                allow_f_increases = true,
                iterations = 1000000)
algo = Optim.ParticleSwarm()
program = Optim.optimize(obj, par, algo, opt; autodiff = :forward)
print(program)
par = Optim.minimizer(program)

println("Polishing with LBFGS...")
opt2 = Optim.Options(g_tol=big(1e-22),
                x_tol=big(1e-22),
                f_tol=big(1e-22),
                time_limit=big(1800.0), # 15 min
                show_trace = true,
                allow_f_increases = false,
                iterations = 100000)
algo2 = Optim.LBFGS(linesearch = LineSearches.MoreThuente())
program2 = Optim.optimize(obj, par, algo2, opt2; autodiff = :forward)
print(program2)
par = Optim.minimizer(program2)

# Extracting the solution for a plot
alpha = par[1:n_gammas] .^2 #make them positives
rates = reshape(par[(n_gammas+1):2n_gammas],(n_gammas,)) .^ 2 # make them positives
rez = hcat(alpha,rates)
rez = rez[sortperm(-rez[:,1]),:]
display(rez)

coefs = MGCLaguerre.get_coefficients(alpha,rates,m)
x = 0:0.01:10

true_density = (x) -> Distributions.pdf(dist,x)
y = KernelDensity.kde(convert.(Float64,sample[1,:]))
kern = x -> KernelDensity.pdf(y,x)
fE = (x)->convert(Float64,MGCLaguerre.laguerre_density(convert(BigFloat,x), E))
f = (x)->convert(Float64,MGCLaguerre.laguerre_density(convert(BigFloat,x), coefs))
plotMat = hcat(true_density.(x), kern.(x), fE.(x), f.(x))
p0 = Plots.plot(x,plotMat, title = "$N samples from a Univariate Ln(0,$sigma), n_gammas = $n_gammas, m = $(m[1])", size=[1920,1024], label = ["Theoretical Ln(0,$sigma)" "KDE of input data" "Laguerre projection" "Gamma projection"])
Plots.display(p0)
Plots.savefig(p0,"LNTest_$m.png")


import Serialization
Serialization.serialize("LNTest_$m.model",(alpha,rates))
