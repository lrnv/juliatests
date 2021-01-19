using ThorinDistributions, DoubleFloats
import Optim, Plots, Random, Distributions, KernelDensity, Serialization

# We should do a Moshopoulos test in 1D, and an emprical test in 2D to be sure that
# the coded get_coeficients is right...

setprecision(256)
# A simple 1D test against moshopoulos & a KDE to be shure that we are good enough.
N = 100000
alpha = [10, 10]
scales = [0.5 0.1; 0.1 0.5]
alpha = BigFloat.(alpha)
scales = BigFloat.(scales)
n_gammas = 2
dist = ThorinDistributions.MultivariateGammaConvolution(alpha, scales)
m = (40,40)
seed = 123
Random.seed!(seed)
sample = Array{Float64}(undef, 2,N)
Random.rand!(dist,sample)
println("Computing empirical coefs, $N samples (may take some time..)")
E = ThorinDistributions.empirical_coefs(sample,m)
println("Done")

coefs = ThorinDistributions.get_coefficients(alpha,scales,m)

M = (prod(m),)
display([reshape(E,M) reshape(coefs,M)])

print(sum((E .- coefs)^2))