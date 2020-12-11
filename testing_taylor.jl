using TaylorSeries


n = 10
d = 2
m = (10,10)

x = set_variables("x",numvars=n*(d+1)+d)
t = x[1:d]
α = x[d+1:n+d]
θ = x[(n+d+1):n*(d+1)+d]
t = x[(n*(d+1)+1+d):end]
θ = reshape(θ,(n,d))

kappa = -sum(α[i] * log(1 + sum(θ[i,:].*t)) for i in 1:n)


mu = exp(kappa)

deriv = Array{TaylorN{Float64},2}(undef,m)

deriv[1,1] = mu

I = CartesianIndices(m)


for k in I[2:end]
    if k[1] != 1
        deriv[k] = derivative(deriv[k[1]-1,k[2]],1)
    else
        deriv[k] = derivative(deriv[k[1],k[2]-1],2)
    end
end

# Now evaluate all these derivatives : 
α = rand(n)
θ = rand(n*d)
θ = reshape(θ,(n,d))
eval_pt = [repeat([-1],d)...,α...,reshape(θ,(n*d))...]

deriv_val = Array{Float64,2}(undef,m)

for i in eachindex(deriv)
    deriv_val[i] = evaluate(deriv[i],eval_pt)
end

# then compute the laguerre polynomial : 
laguerre = laguerre
