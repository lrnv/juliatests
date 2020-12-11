# ]add github.com/lrnv/ThorinDistributions.jl
# ]add https://github.com/lrnv/MultiFloats.jl
import Random, ThorinDistributions

using DoubleFloats
using ArbNumerics
using Quadmath
using MultiFloats
setprecision(256)
setprecision(ArbFloat,256)

types = (Float64,
         Double64, 
         Float128, 
         BigFloat, 
         (MultiFloat{Float64,i} for i in 1:8)...
         )

n = 20
d = 2
m = Tuple(repeat([40],d))
N = 1000

Random.seed!(1234)
α = rand(n)
θ = reshape(10 .* rand(n*d),(n,d))

init = (0.0, 0.0, 0.0)
init_val = (zeros(m),zeros(m))

Values = Dict(T => init_val for T in types)
Errors = Dict(T => init for T in types)
for T in types
    print(T,"\n")

    alpha = T.(α);
    scales = T.(θ);
    dist = ThorinDistributions.MultivariateGammaConvolution(alpha,scales);
    samples = zeros(T,(2,N));
    Random.seed!(123);
    Random.rand!(dist,samples);

    E = ThorinDistributions.empirical_coefs(samples,m); # empirical coefs
    coefs = ThorinDistributions.get_coefficients(alpha, scales,m); # theoreticla coefs
    aloc = @allocated ThorinDistributions.get_coefficients(alpha,scales,m);
    time = Base.@elapsed(ThorinDistributions.get_coefficients(alpha,scales,m));
    

    # All these should have the same magnitude, very small.
    Errors[T] = (Float64(sum((E - coefs)^2)),time,aloc);
    Values[T] = (Float64.(E), Float64.(coefs));
    print(Errors[T],"\n")
end

# Type => (Error, time1, allocations)
display(Errors)


