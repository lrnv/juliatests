import Random, DatagenCopulaBased, Distributions, ThorinDistributions, Optim, Plots
import Serialization
using MultiFloats
setprecision(256)


DoubleType = BigFloat
ArbType = BigFloat

Base.rand(::Type{MultiFloat{Float64,5}}) = MultiFloat{Float64,5}(rand(Float64))
function Base.:^(x::MultiFloat{Float64,5}, y::MultiFloat{Float64,5}) 
    return MultiFloat{Float64,5}(BigFloat(x)^BigFloat(y))
end
Base.round(x::MultiFloat{Float64,5}, y::RoundingMode{:Up}) = MultiFloat{Float64,5}(round(BigFloat(x),y))

dist_name = "Clayton(7)_Par(1,1)_LN(0,0.83)"
N = 100000
m = (10,10)
Time_ps = 3600
Time_lbfgs = 1800
model_name = "N$(N)_m$(m)_Tpso$(Time_ps)_Tpolish$(Time_lbfgs)"
n_gammas = 10

# Simulate the dataset : 
Random.seed!(123)
sample = 1 .- DatagenCopulaBased.simulate_copula(N,DatagenCopulaBased.Clayton_cop(2,7.0))
marginals = [Distributions.Pareto(2.5,1), Distributions.LogNormal(0,0.83)]
shifts = [1,0]
for i in 1:size(sample,2)
    sample[:,i] = Distributions.quantile.(marginals[i],sample[:,i]) .- shifts[i]
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


#true_density = (x,y) -> Distributions.pdf(dist,[x,y])

tpl = (2, n_gammas)
# fE = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(ArbType,x), convert(ArbType,y)], E))
# g = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(ArbType,x), convert(ArbType,y)], coefs))



# generate some data from the MGC : 
dist = ThorinDistributions.MultivariateGammaConvolution(BigFloat.(alpha),BigFloat.(scales))
simu_sample = BigFloat.(deepcopy(sample))
Random.rand!(dist,simu_sample)

simu_sample = Float64.(simu_sample)
sample = Float64.(sample)
N_plot = N

log_sample = log.(sample)
log_simu = log.(simu_sample)

using Plots, StatsPlots, KernelDensity
p1 = marginalkde(log_sample[1,1:N_plot],log_sample[2,1:N_plot];levels=100)
p2 = marginalkde(log_simu[1,1:N_plot],log_simu[2,1:N_plot]; levels=100)
# p1 = scatter(sample[1,1:N_plot],sample[2,1:N_plot])
# p2 = scatter(simu_sample[1,1:N_plot],simu_sample[2,1:N_plot])
# p = plot(p1,p2,layout=(1,2),size=[1920,1024])

#p1 = scatter(log_sample[1,1:N_plot],log_sample[2,1:N_plot])
#p2 = scatter(log_simu[1,1:N_plot],log_simu[2,1:N_plot])
p = plot(p1,p2,layout=(1,2),size=[1920,1024],link=:both)

p3 = marginalkde(sample[1,1:N_plot],sample[2,1:N_plot];levels=100)
p4 = marginalkde(simu_sample[1,1:N_plot],simu_sample[2,1:N_plot]; levels=100)
# p1 = scatter(sample[1,1:N_plot],sample[2,1:N_plot])
# p2 = scatter(simu_sample[1,1:N_plot],simu_sample[2,1:N_plot])
# p = plot(p1,p2,layout=(1,2),size=[1920,1024])

#p1 = scatter(log_sample[1,1:N_plot],log_sample[2,1:N_plot])
#p2 = scatter(log_simu[1,1:N_plot],log_simu[2,1:N_plot])
pp = plot(p3,p4,layout=(1,2),size=[1920,1024],link=:both)

p5 = scatter(log_sample[1,1:N_plot],log_sample[2,1:N_plot])
p6 = scatter(log_simu[1,1:N_plot],log_simu[2,1:N_plot])
ppp = plot(p5,p6,layout=(1,2),size=[1920,1024],link=:both)


# p1 = Plots.plot(x, y, fE, legend=false, title = "Projection on L_$m", seriestype=:wireframe)
# #p2 = Plots.plot(x, y, g, legend=false, title = "Estimation in G_$tpl", seriestype=:wireframe)
# p = Plots.plot(p1,p2, layout = (1,2), size=[1920,1024])
# Plots.display(p)

# Save stuff :
if !isdir(dist_name)
    mkdir(dist_name)
end
Plots.savefig(ppp,"$dist_name/$(model_name)_3.pdf")
Plots.savefig(pp,"$dist_name/$(model_name).pdf")
Plots.savefig(p,"$dist_name/$(model_name)_2.pdf")
Serialization.serialize("$dist_name/$model_name.model",(alpha,scales))



