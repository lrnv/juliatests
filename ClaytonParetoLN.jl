import Random, DatagenCopulaBased, Distributions, ThorinDistributions, Optim, Plots
import Serialization
using MultiFloats
setprecision(256)


DoubleType = Float64x2
ArbType = BigFloat

Base.rand(::Type{MultiFloat{Float64,2}}) = MultiFloat{Float64,2}(rand(Float64))
function Base.:^(x::MultiFloat{Float64,2}, y::MultiFloat{Float64,2}) 
    return MultiFloat{Float64,2}(BigFloat(x)^BigFloat(y))
end
Base.round(x::MultiFloat{Float64,2}, y::RoundingMode{:Up}) = round(BigFloat(x),y)


dist_name = "Clayton(7)_Par(1,1)_LN(0,0.83)"
N = 100000
m = (40,40)
Time_ps = 7200
Time_lbfgs = 7200
model_name = "N$(N)_m$(m)_Tpso$(Time_ps)_Tpolish$(Time_lbfgs)"
n_gammas = 20

# Simulate the dataset : 
Random.seed!(123)
sample = DatagenCopulaBased.simulate_copula(N,DatagenCopulaBased.Clayton_cop(2,7.0))
marginals = [Distributions.Pareto(1,1), Distributions.LogNormal(0,0.83)]
for i in 1:size(sample,2)
    sample[:,i] = Distributions.quantile.(marginals[i],sample[:,i])
end
sample = DoubleType.(transpose(sample))

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
par = ArbType.(par)
tol = ArbType.(tol)
sample = ArbType.(sample)
E = ThorinDistributions.empirical_coefs(sample,m)

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
rates = reshape(par[(n_gammas+1):3n_gammas],(n_gammas,2)) .^ 2 # make them positives
rez = hcat(alpha,rates)
rez = rez[sortperm(-rez[:,1]),:]
display(rez)

coefs = ThorinDistributions.get_coefficients(alpha,rates,m)
x = y = ArbType.(0:0.5:10)
true_density = (x,y) -> Distributions.pdf(dist,[x,y])

tpl = (2, n_gammas)
fE = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(ArbType,x), convert(ArbType,y)], E))
g = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(ArbType,x), convert(ArbType,y)], coefs))
p1 = Plots.plot(x, y, fE, legend=false, title = "Projection on L_$m", seriestype=:wireframe)
p2 = Plots.plot(x, y, g, legend=false, title = "Estimation in G_$tpl", seriestype=:wireframe)
p = Plots.plot(p1,p2, layout = (1,2), size=[1920,1024])
Plots.display(p)

# Save stuff :
if !isdir(dist_name)
    mkdir(dist_name)
end
Plots.savefig(p,"$dist_name/$model_name.pdf")
Serialization.serialize("$dist_name/$model_name.model",(alpha,rates))
