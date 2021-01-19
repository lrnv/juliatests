import Random, DatagenCopulaBased, Distributions, ThorinDistributions, Optim, Serialization
setprecision(1024)

const na = [CartesianIndex()]

dist_name = "Clayton(7)_Par(2.5,1)_LN(0,0.83)"
N_eval_pts = 1000
N = 100000
d = 2
Time_lbfgs = 3000
n_gammas = 20

# Simulate the dataset : 
Random.seed!(123)
sample = 1 .- DatagenCopulaBased.simulate_copula(N,DatagenCopulaBased.Clayton_cop(2,7.0))
#sample = 1 .- DatagenCopulaBased.simulate_copula(N,DatagenCopulaBased.Gaussian_cop([1. 0.5; 0.5 1.]))

marginals = [Distributions.Pareto(2.5,1), Distributions.LogNormal(0,0.83)]
shift = [1,0]
for i in 1:size(sample,2)
    sample[:,i] = Distributions.quantile.(marginals[i],sample[:,i]) .- shift[i]
end
sample = transpose(sample)

function generate_evaluation_points(n,d)
    n = Int(floor(n/(d+1)))

    # Version sans ajout de marges:
    E = -log.(reshape(rand(n*(d+1)),(n,d+1)))
    E = E ./ sum(E,dims=2)

    # Version avec ajout de marges: 
    # E = -log.(reshape(rand(n*(d+1)^2),(n*(d+1),d+1)))
    # for dim in 1:d
    #     E[((dim-1)*n+1):(dim*n),dim] .= 0
    # end
    # E = E ./ sum(E,dims=2)



    return -E[:,1:d]

end
function ecgf(points,sample)
    rez = zeros(size(points,1))
    Threads.@threads for i in 1:size(points,1)
        for j in 1:size(sample,1)
            rez[i] += exp(sum(points[i,:] .* sample[j,:]))
        end
        println(i)
    end
    rez ./= size(sample,1)
    return log.(rez)
end
function MGC_cgf!(rez,points,α,θ)
    @inbounds for j in 1:length(α),i in 1:size(points,1)
        rez[i] = rez[i] - α[j] * log.(1-sum(θ[j,:] .* points[i,:]))
    end
end
function MGC_cgf(points,α,θ)
    rez = zeros(Base.eltype(α),size(points,1))
    MGC_cgf!(rez,points,α,θ)
    return rez
end
function L2Obj_ecgf(α,θ,eval_points,ecgf)
    th_cgf = zeros(Base.eltype(α),size(eval_points,1))
    MGC_cgf!(th_cgf,eval_points,α,θ)
    return sum((ecgf .- th_cgf) .^2 )
end
function L2Obj_ecgf_par(par,eval_points,ecgf,d)
    n = Int((size(par)[1])/(d+1))
    α = par[1:n] .^2 #make them positives
    θ = reshape(par[(n+1):(d+1)*n],(n,d)) .^ 2 # make them positives
    return L2Obj_ecgf(α,θ,eval_points,ecgf)
end
function loss_and_grad!(F,G,x,eval_points,ecgf,d)
    n = Int((size(x)[1])/(d+1))
    α = x[1:n]
    θ = reshape(x[(n+1):(d+1)*n],(n,d))
    T = Base.eltype(x)
    N_pts = size(eval_points,1)

    α_square = α .^ 2
    θ_square = θ .^ 2

    in_log  = zeros(T,(n,N_pts))
    log_in_log = zeros(T,(n,N_pts))
    th_cgf = zeros(T,N_pts)

    @inbounds for j in 1:n,t in 1:N_pts
        in_log[j,t] = 1-sum(θ_square[j,:] .* eval_points[t,:])
        log_in_log[j,t] = log(in_log[j,t])
        th_cgf[t] -= α_square[j] * log_in_log[j,t]
    end

    value = (ecgf .- th_cgf)

    if G != nothing
        G[1:n] = 4 .* α .* sum(log_in_log .* value[na,:],dims=2)
        @inbounds for j in 1:n, i in 1:d
            #print((i,j,θ[j,i],par[n + n*(i-1) + j]),"\n")
            G[n + n*(i-1) + j] = -4 * α_square[j] * θ[j,i] * sum(eval_points[:,i] .* value ./ in_log[j,:])
        end 
    end
    if F != nothing
        return sum(value .^2)
    end
end


# Pre-computations: 
Eval_pts = generate_evaluation_points(N_eval_pts,2)
ecgf_to_fit = ecgf(Eval_pts,transpose(sample))

# Instentiate objecive function:
function obj(x)
    L2Obj_ecgf_par(x,Eval_pts,ecgf_to_fit,d)
end
function obj_only_fg!(F,G,x)
    loss_and_grad!(F,G,x,Eval_pts,ecgf_to_fit,d)
end


par =Random.rand(3n_gammas) .- 1/2
tol = 10^(-10)

println("Polishing with LBFGS...")
opt2 = Optim.Options(g_tol=tol,
                x_tol=tol,
                f_tol=tol,
                time_limit=Time_lbfgs,
                show_trace = true,
                allow_f_increases = true,
                iterations = 10000000)
algo2 = Optim.LBFGS()
program2 = Optim.optimize(Optim.only_fg!(obj_only_fg!), par, algo2, opt2)
print(program2)
par = Optim.minimizer(program2)


# Extracting the solution for a plot
alpha = par[1:n_gammas] .^2 #make them positives
scales = reshape(par[(n_gammas+1):3n_gammas],(n_gammas,2)) .^ 2 # make them positives
rez = hcat(alpha,scales)
rez = rez[sortperm(-rez[:,1]),:]
display(rez)


# Sample some data : 
N_plot = N
# generate some data from the MGC : 
dist = ThorinDistributions.MultivariateGammaConvolution(BigFloat.(alpha),BigFloat.(scales))
simu_sample = BigFloat.(deepcopy(sample[:,1:N_plot]))
Random.rand!(dist,simu_sample)

# Take logs and go back to floats : 
log_sample = log.(sample)
log_simu_sample = log.(simu_sample)
simu_sample = Float64.(simu_sample)
sample = Float64.(sample)
log_sample = Float64.(log_sample)
log_simu_sample = Float64.(log_simu_sample)


# Let's try to plot the copula. 

using StatsBase
sample_cop = transpose([ordinalrank(sample[1,:]) ordinalrank(sample[2,:])]/(N+1))
simu_cop = transpose([ordinalrank(simu_sample[1,:]) ordinalrank(simu_sample[2,:])]/(N+1))

using Plots, StatsPlots, KernelDensity
p1 = marginalkde(log_sample[1,1:N_plot],log_sample[2,1:N_plot];levels=50)
q1 = marginalkde(log_simu_sample[1,1:N_plot],log_simu_sample[2,1:N_plot]; levels=50)
p2 = qqplot(sample[1,1:N_plot],simu_sample[1,1:N_plot],qqline = :R)
q2 = qqplot(sample[2,1:N_plot],simu_sample[2,1:N_plot],qqline = :R)
p3 = plot(kde(transpose(sample_cop)))#marginalkde(sample_cop[1,1:N_plot],sample_cop[2,1:N_plot];levels=100)
q3 = plot(kde(transpose(simu_cop)))#marginalkde(simu_cop[1,1:N_plot],simu_cop[2,1:N_plot];levels=100)

p = plot(p1,p2,p3,q1,q2,q3,layout=(2,3),size=[1920,1024])
Plots.savefig(p,"test_ecgf_clayton_20.png")
