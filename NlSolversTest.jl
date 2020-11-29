using NLSolvers
using ThreadsX
using Base.Threads

include("MGC.jl")
setprecision(MGCLaguerre.P.PREC)
import Optim, Plots, Random, Distributions, LineSearches, KernelDensity

# The multivariate lognormal test case:
Random.seed!(123)
N = 1000
dist = Distributions.LogNormal(0,1)
sample = Array{Float64}(undef, 1,N)
Random.rand!(dist,sample)
sample = convert.(BigFloat,sample)
m = (20,)
println("Computing empirical coefs of standard Log normal, $N samples (may take some time..)")
E = MGCLaguerre.empirical_coefs(sample,m)
println("Done")


println("Launching ParticleSwarm...")
n_gammas = 40
x0 = big.((Random.rand(2n_gammas) .- 1/2))


function himmelblau!(x)
   fx = MGCLaguerre.L2Objective(x, E)
   println(convert(Float64,fx))
   return fx
end


function himmelblau_batched!(F, X)
   ThreadsX.map!(himmelblau!, F, X)
   return F
end

scalar_obj = ScalarObjective(himmelblau!, nothing, nothing, nothing, nothing, nothing,  himmelblau_batched!, nothing)

prob = OptimizationProblem(scalar_obj)
opts = OptimizationOptions(x_abstol=0.0, x_reltol=0.0, x_norm=x->norm(x, Inf),
             g_abstol=1e-8, g_reltol=0.0, g_norm=x->norm(x, Inf),
             f_limit=-Inf, f_abstol=0.0, f_reltol=0.0,
             nm_tol=1e-8, maxiter=10000, show_trace=true)
rez = solve(prob, x0, ParticleSwarm(), opts)

rez.info.X

rez.info.minimum
