include("MGC.jl")
setprecision(MGCLaguerre.P.PREC)

d = 4
n = 10
C = 10
c = zeros(BigFloat,(d,n))
scales = zeros(BigFloat,(d,n))
for j in 1:d
    for i in 1:n
        c[j,i] = big(rand())
        scales[j,i] = big(rand())
    end
    c[j,:] = C*c[j,:]/sum(c[j,:])
end

# maintenant il faut générer une matrice A alétaoire.

A = zeros(BigFloat,Tuple(repeat([n],d)))
for i in eachindex(A)
    A[i] = rand()
end

function project!(A,c)

    d = ndims(A)
    n = size(A)[1]
    @assert(all(size(A) .== n),"A must have all dimensions of same length")
    @assert(size(c,1) == d, "A must have d dimensions, and c must have d rows")
    @assert(size(c,2) == n, "c must have n rows, A must have dimensions of length n")


    rez = 1
    while rez > 10*eps(eltype(A))
        rez = 0
        for i in 1:n
            for j in 1:d
                idx = filter(k -> k[j]==i, CartesianIndices(A))
                rez += (sum(A[idx]) - c[j,i])^2
                A[idx] = A[idx]/sum(A[idx])*c[j,i]
            end
        end
        #rez = sqrt(rez)
        println(rez)
    end
end

project!(A,c)

# then we ned to construct the right expression for (alpha, beta)
function construct_alpha_theta(A,scales)
    d = ndims(A)
    n = size(A)[1]
    α = zeros((n^d,))
    θ = zeros((n^d,d))
    j = 1
    for i in CartesianIndices(A)
        α[j] = A[i]
        for k in 1:d
            θ[j,k] = scales[k,i[k]]
        end
        j = j+1
    end
    return α,θ
end

α,θ = construct_alpha_theta(A,scales)


# then we can get coeffcients:
