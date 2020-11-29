# This file is only there for testing eventual performance imporvements.
using ArbNumerics, Readables, DoubleFloats

# You can switch to BigFloat to get the true values, but pre-computations need to be re-ran.
setprecision(ArbFloat, bits=256)
const ArbT = ArbFloat{256}

struct PreComp{ArbT}
    BINS::Array{ArbT,2}
    LAGUERRE::Array{ArbT,2}
    FACTS::Array{ArbT,1}
    MAX_SUM_OF_M::ArbT
end
function PreComp(m)
    setprecision(2048)
    m = big(m)
    BINS = zeros(BigInt,(m,m))
    FACTS = zeros(BigInt,m)
    LAGUERRE = zeros(BigFloat,(m,m))
    for i in 1:m
        FACTS[i] = factorial(i-1)
    end
    for i in 1:m, j in 1:m
        BINS[j,i] = binomial(i-1,j-1)
        LAGUERRE[j,i] = -BINS[j,i]/FACTS[j]*(-big(2))^(j-1)
    end
    BINS = ArbT.(BINS)
    LAGUERRE = ArbT.(LAGUERRE)
    FACTS = ArbT.(FACTS)
    PreComp(BINS,LAGUERRE,FACTS,ArbT(m))
end
const P = PreComp(100)


function get_coefficients(α, θ, m)
    # α must be an array with size (n,)
    # θ must be an array with size (n,d)
    # max_p must be a Tuple of integers with size (d,)
    # we ensure that α and θ have same type.
    entry_type = Base.promote_eltype(α,θ,[1.0]) # At least float.
    α = ArbT.(α)
    θ = ArbT.(θ)

    # Trim down null αs values:
    are_pos = α .!= 0
    θ = θ[are_pos,:]
    α = α[are_pos]

    # Allocates ressources
    coefs = zeros(ArbT,m)
    κ = deepcopy(coefs)
    μ = deepcopy(coefs)
    n = size(θ)[1]
    d = length(m)
    I = CartesianIndices(coefs)

    # Construct the simplex expression of θ, and compute it's powers.
    S = θ ./ (ArbT(1) .+ sum(θ,dims=2))
    S_pow = [s^k for k in (0:Base.maximum(m)), s in S]


    # Starting the algorithm: there is an edge case for the Oth cumulant, 0th moment and 0th coef:
    κ[1] = sum(α .* log.(ArbT(1) .- sum(S,dims=2))) # this log fails ifsum(S,dims=2) is greater than 1, which should not happend.

    coefs[1] = μ[1] = exp(κ[1])

    for k in I[2:length(I)]
        # Indices and organisation
        k_arr = Tuple(k)
        degree = findfirst(k_arr .!= 1)
        sk = sum(k_arr)

        # Main Computations:
        for i in 1:n
            rez = α[i]
            for j in 1:d
                rez *= S_pow[k[j],i,j]
            end
            κ[k] += rez
        end
        κ[k] *= P.FACTS[sk-d]

        for j in CartesianIndices(k)
            rez_coefs = μ[j]
            if j[degree] < k[degree]
                rez_mu = μ[j] * κ[k - j + I[1]]
                for i in 1:d
                    rez_mu *= P.BINS[j[i],k[i]-Int(i==degree)]
                    rez_coefs *= P.LAGUERRE[j[i], k[i]]
                end
                μ[k] += rez_mu
            else
                for i in 1:d
                    rez_coefs *= P.LAGUERRE[j[i], k[i]]
                end
            end
            coefs[k] += rez_coefs
        end
        # for j in CartesianIndices(k)
        #     # In this version, we replaced mu_k by mu_k / k!, which simplifies a little the combinatorics.
        #     rez_coefs = μ[j]
        #     if j[degree] < k[degree]
        #         rez_mu = T(1)
        #         idx = k - deg - j + I[1]
        #         for i in 1:d
        #             rez_mu *= P.FACTS_INV[idx[i]]
        #             rez_coefs *= P.LAGUERRE_VAR[j[i],k[i]]
        #         end
        #         μ[k] += rez_mu * μ[j] * κ[k - j + I[1]] / (k[degree] - 1)
        #     else
        #         for i in 1:d
        #             rez_coefs *= P.LAGUERRE_VAR[j[i],k[i]]
        #         end
        #     end
        #
        #     coefs[k] += rez_coefs
        # end
        # for j in CartesianIndices(k)
        #     # In this version, we replaced mu_k by mu_k / k!, which simplifies a little the combinatorics.
        #     # we also replaced kappa[k] by k!/kappa[k], for the same reason.
        #     μ[k] += (k[degree]-j[degree]) * μ[j] * κ[k - j + I[1]] / (k[degree] -1)
        #
        #     rez_coefs = μ[j]
        #     for i in 1:d
        #         rez_coefs *= P.LAGUERRE_VAR[j[i],k[i]]
        #     end
        #     coefs[k] += rez_coefs
        # end
        #println((k,κ[k],μ[k],coefs[k]))
    end
    coefs *= sqrt(ArbT(2))^d
    return entry_type.(coefs)
end




# Some benchmarks :


using BenchmarkTools
println("1D test")
alpha = [10, 10, 10]
scales = [0.5; 0; 1]
alpha = Double64.(alpha)
scales = Double64.(scales)
m = (80,)
@btime coefs1 = get_coefficients(alpha, scales,m)

println("2D test")
alpha = [10, 10]
scales = [0.5 0.1; 0.1 0.5]
alpha = Double64.(alpha)
scales = Double64.(scales)
m = (40,40)
@btime coefs2 = get_coefficients(alpha, scales,m)
coefs2 = get_coefficients(alpha, scales,m);
#display(coefs2)
@profiler coefs2 = get_coefficients(alpha, scales,m)


# The bigfloat value of coefs2[25,25] is supposed to be :
true_val = big"-1.329472861215989958353148796949387245505296082743958217986291502162732173721559510994933627660600344789815286583644267318321004356203981387912906192123517290826758711637887782841271017607754253890068484817044517591221840170581028583387286205353531089134284520690406090243222613041001294915010364850669103680158e-08"
error = Float64(trunc((big(coefs2[25,25])-true_val)/true_val,sigdigits = 10))
println("% Error on the 25/25 coef : $error")


#
# println("Realistic test")
# alpha = rand(40)
# scales = reshape(rand(120),(40,3))
# alpha = Double64.(alpha)
# scales = Double64.(scales)
# m = (15,15,15)
# @time coefs3 = get_coefficients(alpha,scales,m);
#
# # I use juno's profiler :
# @profiler coefs3 = get_coefficients(alpha,scales,m);
