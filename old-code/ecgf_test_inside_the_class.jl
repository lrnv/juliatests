import Random, ThorinDistributions, Optim, Serialization

const na = [CartesianIndex()]

N_eval_pts = 10000
N = 100000
d = 2
Time_ps = 30
Time_lbfgs = 60
n_gammas = 20

# Simulate the dataset : 
Random.seed!(123)
α = 2*rand(n_gammas)
θ = 10*reshape(rand(n_gammas*d),(n_gammas,d))
dist = ThorinDistributions.MultivariateGammaConvolution(α,θ)
sample = zeros((d,N))
Random.rand!(dist,sample)

function generate_evaluation_points(n,d; add_zeros=false)
    if add_zeros
        n = Int(floor(n/(d+1)))
        E = -log.(reshape(rand(n*(d+1)^2),(n*(d+1),d+1)))
        for dim in 1:d
            E[((dim-1)*n+1):(dim*n),dim] .= 0
        end
        E = E ./ sum(E,dims=2)
    else
        E = -log.(reshape(rand(n*(d+1)),(n,d+1)))
        E = E ./ sum(E,dims=2)
    end
    return -E[:,1:d]
end
function ecgf(points,sample)
    rez = zeros(size(points,1))
    Threads.@threads for i in 1:size(points,1)
        for j in 1:size(sample,1)
            rez[i] += exp(sum(points[i,:] .* sample[j,:]))
        end
        print(i,"\n")
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
        G ./= N_pts
    end
    if F != nothing
        return sum(value .^2) ./ N_pts
    end
end


# Pre-computations: 
Eval_pts = generate_evaluation_points(N_eval_pts,2; add_zeros=false)
ecgf_to_fit = ecgf(Eval_pts,transpose(sample))

# Instentiate objecive function:
function obj(x)
    L2Obj_ecgf_par(x,Eval_pts,ecgf_to_fit,d)
end
function obj_only_fg!(F,G,x)
    loss_and_grad!(F,G,x,Eval_pts,ecgf_to_fit,d)
end
function model(x,p)
    n = Int((size(p)[1])/(d+1))
    α = p[1:n] .^ 2 #make them positives
    θ = reshape(p[(n+1):(d+1)*n],(n,d)) .^ 2 # make them positives
    rez = zeros(Base.eltype(α),size(x,2))
    return MGC_cgf(transpose(x),α,θ)
end


par =Random.rand(3n_gammas) .- 1/2
tol = 10^(-10)

# println("Launching ParticleSwarm...")
# opt = Optim.Options(g_tol=tol,
#                 x_tol=tol,
#                 f_tol=tol,
#                 time_limit=Time_ps, # 30min
#                 show_trace = true,
#                 allow_f_increases = true,
#                 iterations = 10000000)
# algo = Optim.ParticleSwarm()
# program = Optim.optimize(obj, par, algo, opt)
# print(program)
# par = Optim.minimizer(program)

println("Polishing with LBFGS...")
opt2 = Optim.Options(g_tol=tol,
                x_tol=tol,
                f_tol=tol,
                time_limit=Time_lbfgs, # 30min
                show_trace = true,
                allow_f_increases = true,
                iterations = 10000000)
algo2 = Optim.BFGS() #LBFGS
program2 = Optim.optimize(Optim.only_fg!(obj_only_fg!), par, algo2, opt2)
par = Optim.minimizer(program2)


# using LsqFit
# pts = convert(typeof(Eval_pts), transpose(Eval_pts))
# goal = ecgf_to_fit
# fit = curve_fit(model, pts, goal, par; show_trace = true)


# Extracting the solution for a plot
alpha = par[1:n_gammas] .^2 #make them positives
scales = reshape(par[(n_gammas+1):3n_gammas],(n_gammas,2)) .^ 2 # make them positives
rez = hcat(alpha,scales)
rez = rez[sortperm(-rez[:,1]),:]
display(rez)

# coefs = ThorinDistributions.get_coefficients(BigFloat.(alpha),BigFloat.(scales),m)
# E = ThorinDistributions.empirical_coefs(BigFloat.(sample),m)
# x = y = 0:0.2:10

# tpl = (2, n_gammas)
# f = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(BigFloat,x), convert(BigFloat,y)], E))
# g = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([convert(BigFloat,x), convert(BigFloat,y)], coefs))
# p1 = Plots.plot(x, y, f, legend=false, title = "Projection on the laguerre basis", seriestype=:wireframe)
# p2 = Plots.plot(x, y, g, legend=false, title = "Estimation in G_$tpl", seriestype=:wireframe)
# p = Plots.plot(p1,p2, layout = (1,2), size=[1920,1024])
# Plots.display(p)

# # Save stuff :
# if !isdir(dist_name)
#     mkdir(dist_name)
# end
# Plots.savefig(p,"$dist_name/$model_name.pdf")
# Serialization.serialize("$dist_name/$model_name.model",(alpha,ratscaleses))



# generate some data from the MGC : 
est_dist = ThorinDistributions.MultivariateGammaConvolution(BigFloat.(alpha),BigFloat.(scales))
simu_sample = BigFloat.(deepcopy(sample))
Random.rand!(est_dist,simu_sample)

simu_sample = Float64.(simu_sample)
sample = Float64.(sample)

log_sample = log.(sample)
log_simu_sample = log.(simu_sample)

N_plot = 5000

using Plots, StatsPlots, KernelDensity
p1 = marginalkde(log_sample[1,1:N_plot],log_sample[2,1:N_plot];levels=100)
p2 = marginalkde(log_simu_sample[1,1:N_plot],log_simu_sample[2,1:N_plot]; levels=100)
p = plot(p1,p2,layout=(1,2),size=[1920,1024])

p1 = marginalkde(sample[1,1:N_plot],sample[2,1:N_plot];levels=100)
p2 = marginalkde(simu_sample[1,1:N_plot],simu_sample[2,1:N_plot]; levels=100)
p = plot(p1,p2,layout=(1,2),size=[1920,1024])

plot(
    qqplot(sample[1,:],simu_sample[1,:],qqline = :fit),
    qqplot(sample[2,:],simu_sample[2,:],qqline = :fit),
    size=[1920,1080]
)
