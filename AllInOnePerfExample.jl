using DoubleFloats
setprecision(256)
const ArbT = BigFloat

struct PreComp{Tb,Tl,Tf,Tm}
    BINS::Tb
    LAGUERRE::Tl
    FACTS::Tf
    MAX_SUM_OF_M::Tm
end

"""
    PreComp(m)

Given a precison of computations and a tuple m which gives the size of the future laguerre basis, this fonctions precomputes certain quatities
these quatities might be needed later...
"""
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
        LAGUERRE[j,i] = BINS[j,i]/FACTS[j]*(-big(2))^(j-1)
    end
    BINS = ArbT.(BINS)
    LAGUERRE = ArbT.(LAGUERRE)
    FACTS = ArbT.(FACTS)
    PreComp(BINS,LAGUERRE,FACTS,ArbT(m))
end
const P = PreComp(200)

"""
    get_coefficients(α,θ,m)

α should be a vector of shapes of length n, θ should be a matrix of θ of size (n,d) for a MultivariateGammaConvolution with d marginals.

This function produce the coefficients of the multivariate (tensorised) expensions of the density. It is type stable and quite optimized.
it is based on some precomputations that are done in a certain precision.

"""
function get_coefficients(α, θ, m)
    # α must be an array with size (n,)
    # θ must be an array with size (n,d)
    # max_p must be a Tuple of integers with size (d,)
    entry_type = Base.promote_eltype(α,θ,[1.0]) # At least float.
    α = ArbT.(α)
    θ = ArbT.(θ)

    # Trim down null αs values:
    are_pos = α .!= 0
    θ = θ[are_pos,:]
    α = α[are_pos]

    coefs = zeros(ArbT,m)
    build_coefficients!(coefs,α,θ,ArbT(1.0),m)
    return entry_type.(coefs)
end

function build_coefficients!(coefs,α,θ,cst1,m)
    # Allocates ressources
    κ = deepcopy(coefs)
    μ = deepcopy(coefs)
    n = size(θ)[1]
    d = length(m)
    I = CartesianIndices(coefs)
    S = θ ./ (cst1 .+ sum(θ,dims=2))
    S_pow = [s^k for k in (0:Base.maximum(m)), s in S]

    # Starting the algorithm: there is an edge case for the Oth cumulant, 0th moment and 0th coef:
    κ[1] = sum(α .* log.(cst1 .- sum(S,dims=2))) # this log fails ifsum(S,dims=2) is greater than 1, which should not happend.
    μ[1] = exp(κ[1])
    coefs[1] = μ[1]

    @inbounds for k in I[2:length(I)]
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
    end
    coefs .*= sqrt(2*cst1)^d
end

function build_coefficients!(coefs,α,θ,cst1,m::NTuple{1, T}) where T <: Int

    θ = θ[:,1]
    m = m[1]
    kappa = Array{eltype(α), 1}(undef, m)
    mu = zeros(eltype(α), m)

    @. θ /= 1 + θ

    # Edge case for the Oth cumulant, 0th moment and 0th coef:
    kappa[1] = sum(α .* log.(cst1 .- θ))
    coefs[1] = mu[1] = exp(kappa[1])

    @inbounds for k in 2:m
        # Main Computations:
        α .*= θ
        kappa[k] = sum(α) * P.FACTS[k - 1]
    
        for j in 1:k-1
            mu[k] += mu[j] * kappa[k - j + 1] * P.BINS[j, k-1]
        end
        
        @views coefs[k] = sum(mu[1:k] .* P.LAGUERRE[1:k, k])
    end

    coefs .*= sqrt(2*cst1)
end

function build_coefficients!(coefs,α,θ,cst1,m::NTuple{2, T}) where T <: Int
    # Allocates ressources
    κ = Array{Base.eltype(α), 2}(undef, m)
    μ = zeros(Base.eltype(α), m)

    #n = size(θ)[1]
    d = length(m)
    I = CartesianIndices(coefs)
    θ ./= (cst1 .+ sum(θ,dims=2))
    S_pow = [s^k for k in (0:Base.maximum(m)), s in θ]
    # reduce the number of multiplications : 
    for i in 1:size(α,1)
        S_pow[:,i,:] .*= sqrt(α[i])
    end

    # Starting the algorithm: there is an edge case for the Oth cumulant, 0th moment and 0th coef:
    κ[1] = sum(α .* log.(cst1 .- sum(θ,dims=2))) # this log fails ifsum(S,dims=2) is greater than 1, which should not happend.
    μ[1] = exp(κ[1])
    coefs[1] = μ[1]

    @inbounds for k in I[2:length(I)]
        degree = k[1]!=1 ? 1 : 2
        κ[k] = sum(S_pow[k[1],:,1] .* S_pow[k[2],:,2]) * P.FACTS[sum(Tuple(k))-d]
        for j in CartesianIndices(k)
            if j[degree] < k[degree]
                μ[k] += μ[j] * κ[k - j + I[1]] *  P.BINS[j[1],k[1]-Int(1==degree)] * P.BINS[j[2],k[2]-Int(2==degree)]
            end
            coefs[k] += μ[j] * P.LAGUERRE[j[1], k[1]] * P.LAGUERRE[j[2], k[2]]
        end
    end
    coefs .*= sqrt(2*cst1)^d
end



# Some benchmarks :


using BenchmarkTools
# println("1D test")
# alpha = [10, 10, 10]
# scales = [0.5; 0; 1]
# alpha = Double64.(alpha)
# scales = Double64.(scales)
# m = (80,)
# @btime coefs1 = get_coefficients(alpha, scales,m)

println("2D test")
alpha = [10, 10]
scales = [0.5 0.1; 0.1 0.5]
alpha = Double64.(alpha)
scales = Double64.(scales)
m = (25,25)
@btime coefs2 = get_coefficients($alpha, $scales,$m)
coefs2 = get_coefficients(alpha, scales,m);
#display(coefs2)


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




# A simple 1D test against moshopoulos & a KDE to be shure that we are good enough.
# using ThorinDistributions, Random, Plots, Distributions, KernelDensity
# N = 10000
# alpha = [0.41163655254510817,0.29041964536214265]
# scales = [2.415904044321296,0.6785192219503884]
# n_gammas = 2
# dist = ThorinDistributions.UnivariateGammaConvolution(alpha, scales)
# m = (10,)
# seed = 123
# Random.seed!(seed)
# sample = Array{Float64}(undef, 1,N)
# Random.rand!(dist,sample)
# println("Computing empirical coefs, $N samples (may take some time..)")
# E = ThorinDistributions.empirical_coefs(sample,m)
# println("Done")

# coefs = get_coefficients(alpha,scales,m)
# x = sort(sample[1,1:100])

# mosch = (x) -> Distributions.pdf(dist,Double64.(x))
# y = KernelDensity.kde(convert.(Float64,sample[1,:]))
# kern = x -> KernelDensity.pdf(y,x)
# fE = (x)->convert(Float64,ThorinDistributions.laguerre_density(x, E))
# f = (x)->convert(Float64,ThorinDistributions.laguerre_density(x, coefs))
# plotMat = hcat(mosch.(x), kern.(x), fE.(x), f.(x))
# p0 = Plots.plot(x,
#                 plotMat,
#                 title = "$N samples from a convolution of $n_gammas gammas, m=$(m[1])",
#                 size=[1000,600],
#                 label = ["Moschopoulos density" "KDE of input data" "Laguerre Density (empirical coefs)" "Laguerre Density (theoritical coefs)"])
# Plots.display(p0)
