# You should ]add github.com/lrnv/ThorinDistributions.jl
import ThorinDistributions
using DoubleFloats

types = (Float64, Double64, BigFloat)

setprecision(1024)

n, d = 20, 2
m = Tuple(repeat([80],d))

α = rand(n)
θ = reshape(10 .* rand(n*d),(n,d))
Values = Dict(T => zeros(m) for T in types)

for T in types
    alpha = T.(α);
    scales = T.(θ);
    coefs = ThorinDistributions.get_coefficients(alpha, scales,m); # theoreticla coefs
    Values[T] = (Float64.(coefs));
end


# Float64 error: 
for T in (Float64, Double64)
    err = (sum((Values[T] - Values[BigFloat]) .^2))
    print("Error for $T: $err \n")
end


