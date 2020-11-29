using ThorinDistributions, DoubleFloats
import Optim, Plots, Random, Distributions, KernelDensity, Serialization

# We should do a Moshopoulos test in 1D, and an emprical test in 2D to be sure that
# the coded get_coeficients is right...


# A simple 1D test against moshopoulos & a KDE to be shure that we are good enough.
N = 10000
alpha = [0.41163655254510817,0.29041964536214265]
scales = [2.415904044321296,0.6785192219503884]
n_gammas = 2
dist = ThorinDistributions.UnivariateGammaConvolution(alpha, scales)
m = (10,)
seed = 123
Random.seed!(seed)
sample = Array{Float64}(undef, 1,N)
Random.rand!(dist,sample)
println("Computing empirical coefs, $N samples (may take some time..)")
E = ThorinDistributions.empirical_coefs(sample,m)
println("Done")

coefs = ThorinDistributions.get_coefficients(alpha,scales,m)
x = sort(sample[1,:])

mosch = (x) -> Distributions.pdf(dist,Double64(x))
y = KernelDensity.kde(convert.(Float64,sample[1,:]))
kern = x -> KernelDensity.pdf(y,x)
fE = (x)->convert(Float64,ThorinDistributions.laguerre_density(x, E))
f = (x)->convert(Float64,ThorinDistributions.laguerre_density(x, coefs))
plotMat = hcat(mosch.(x), kern.(x), fE.(x), f.(x))#, fMosh.(x))
p0 = Plots.plot(x,
                plotMat,
                title = "$N samples from a convolution of $n_gammas gammas, m=$(m[1])",
                size=[1000,600],
                label = ["Moschopoulos density" "KDE of input data" "Laguerre Density (empirical coefs)" "Laguerre Density (theoritical coefs)"])
Plots.display(p0)
