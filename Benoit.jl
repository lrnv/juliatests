# This file is only there for testing eventual performance imporvements.

struct PreComp{Tb, Tl, Tf, Tp, Tm}
    BINS::Tb
    LAGUERRE::Tl
    FACTS::Tf
    PREC::Tp
    MAX_SUM_OF_M::Tm
end

function PreComp(precision, m)
    setprecision(precision)
    BINS = zeros(BigInt, (m, m))
    FACTS = zeros(BigInt, m)
    LAGUERRE = zeros(BigFloat, (m, m))

    bigm = big(m)

    for i in 1:bigm
        FACTS[i] = factorial(i-1)
    end

    for i in 1:bigm
        for j in 1:bigm
            BINS[j, i] = binomial(i-1, j-1)
            LAGUERRE[j, i] = -BINS[j, i] / FACTS[j] * (-big(2.0))^(j-1)
        end
    end

    PreComp(BINS, LAGUERRE, FACTS, precision, bigm)
end

const P = PreComp(1024, 50)


function get_coefficients(alpha, scales, m)
    # alpha must be an array with size (n,)
    # scales must be an array with size (n, d)
    # max_p must be a Tuple of integers with size (d,)

    # Trim down null alphas values:
    are_pos = alpha .!= 0
    scales = scales[are_pos, :]
    alpha = alpha[are_pos]

    # Allocates ressources, construct the simplex expression of the scales and the indexes.
    coefs = zeros(eltype(alpha), m)
    kappa = zeros(eltype(alpha), size(coefs))
    mu = deepcopy(kappa)
    n = size(scales)[1]
    d = ndims(coefs)
    I = CartesianIndices(coefs)
    S = scales ./ (big(1.0) .+ sum(scales, dims=2))
    S_pow = [s^k for k in (0:maximum(m)), s in S]

    # Edge case for the Oth cumulant, 0th moment and 0th coef:
    kappa[1] = sum(alpha .* log.(big(1.0) .- sum(S, dims=2)))
    coefs[1] = mu[1] = exp(kappa[1])

    for k in I[2:length(I)]
        # Indices and organisation
        k_arr = Tuple(k)
        degree = findfirst(k_arr .!= 1)
        sk = sum(k_arr)

        # Main Computations:
        for i in 1:n
            rez = alpha[i]
            for j in 1:d
                rez *= S_pow[k[j], i, j]
            end
            kappa[k] += rez
        end
        kappa[k] *= P.FACTS[sk - d]


        for j in CartesianIndices(k)
            if j[degree] < k[degree]
                rez_mu = mu[j] * kappa[k - j + I[1]]
                rez_coefs = mu[j]
                for i in 1:d
                    rez_mu *= P.BINS[j[i], k[i]-Int(i==degree)]
                    rez_coefs *= P.LAGUERRE[j[i], k[i]]
                end

                mu[k] += rez_mu
            else
                rez_coefs = mu[j]
                for i in 1:d
                    rez_coefs *= P.LAGUERRE[j[i], k[i]]
                end
            end

            coefs[k] += rez_coefs
        end
    end

    coefs *= sqrt(big(2.0))^d
    return coefs
end

# Some benchmarks :

using BenchmarkTools
println("1D test")
alpha = [10, 10, 10]
scales = [0.5; 0; 1]
alpha = convert.(BigFloat, alpha)
scales = convert.(BigFloat, scales)
m = (20,)
@btime coefs1 = get_coefficients(alpha, scales, m)

println("2D test")
alpha = [10, 10]
scales = [0.5 0.1; 0.1 0.5]
alpha = convert.(BigFloat, alpha)
scales = convert.(BigFloat, scales)
m = (10, 10)
@btime coefs2 = get_coefficients(alpha, scales, m)
coefs2 = get_coefficients(alpha, scales, m)
display(coefs2)
println()

println("Realisatic test")
alpha = rand(10)
scales = reshape(rand(20), (10, 2))
alpha = convert.(BigFloat, alpha)
scales = convert.(BigFloat, scales)
m = (15, 15)
@btime coefs3 = get_coefficients(alpha, scales, m)

println("4D test")
alpha = [10, 10, 5, 8, 9, 10, 10]
scales = [0.5 0.0 0.0 0.0; 0.0 0.5 0.0 0.0; 0.0 0.0 0.5 0.0; 1 1 1 1; 2 2 2 2; 2 0 2 0; 0 2 0 2]
alpha = convert.(BigFloat, alpha)
scales = convert.(BigFloat, scales)
m = (3, 3, 3, 3)
@btime coefs4 = get_coefficients(alpha, scales, m)


using Profile
Profile.init(delay = 0.00001)
Profile.clear()

alpha = convert.(BigFloat, rand(10))
scales = convert.(BigFloat, reshape(rand(20), (10, 2)))
m = (15, 15)

Profile.clear_malloc_data()
coefs3 = get_coefficients(alpha, scales, m)

@profile get_coefficients(alpha, scales, m)
Profile.print(format=:flat, sortedby=:count)

#using InteractiveUtils
#@code_warntype get_coefficients(alpha, scales, m)
