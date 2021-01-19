import Random, DatagenCopulaBased, Distributions, ThorinDistributions, Optim, Serialization
setprecision(1024)
using HypothesisTests

const na = [CartesianIndex()]

dist_name = "Pareto(2.5,1)"
N_eval_pts = 100
N = 100000
d = 1
Time_lbfgs = 3000
n_gammas = 20

# Simulate the dataset : 
Random.seed!(123)
dist = Distributions.Pareto(2.5,1)
sample = Array{Float64}(undef, 1,N)
Random.rand!(dist,sample)
sample .-= 1 # shift

function generate_evaluation_points(n,d)
    

    # Version sans ajout de marges:
    E = -log.(reshape(rand(n*(d+1)),(n,d+1)))
    E = E ./ sum(E,dims=2)

    # Version avec ajout de marges: 
    # n = Int(floor(n/(d+1)))
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


par =Random.rand((d+1)*n_gammas) .- 1/2
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
scales = reshape(par[(n_gammas+1):((d+1)*n_gammas)],(n_gammas,d)) .^ 2 # make them positives
if d == 1
    scales = reshape(scales,(n_gammas,))
end
rez = hcat(alpha,scales)
rez = rez[sortperm(-rez[:,1]),:]
display(rez)

m = 21
coefs = ThorinDistributions.get_coefficients(alpha,scales,m)
E = ThorinDistributions.empirical_coefs(sample,(m,))
x = BigFloat.(0:0.01:7.5)

model_name = "N$(N)_m$(m[1])_Tpso$(Time_ps)_Tpolish$(Time_lbfgs)"
true_density = (x) -> convert(Float64,Distributions.pdf(dist,x+shift))
moschdist = ThorinDistributions.UnivariateGammaConvolution(alpha,scales)
fMosch = (x) -> convert(Float64,Distributions.pdf(moschdist,x))
y = KernelDensity.kde(convert.(Float64,sample[1,:]))
kern = x -> KernelDensity.pdf(y,convert(Float64,x))
fE = (x)->convert(Float64,ThorinDistributions.laguerre_density(x, E))
f = (x)->convert(Float64,ThorinDistributions.laguerre_density(x, coefs))


plotMat = hcat(true_density.(x),
                kern.(x),
                fE.(x),
                f.(x))
                #mosh)
x = Float64.(x)
# Let's do some KS testing : 
new_sample = sample = Array{BigFloat}(undef, 1,Int(N//10))
Random.rand!(moschdist,new_sample)
N_ks_tests = 100
new_samples = [new_sample for i in 1:N_ks_tests]

p_values = zeros(N_ks_tests)

println("Computing KS...")
Threads.@threads for i in 1:N_ks_tests
    Random.rand!(moschdist,new_samples[i])
    p_values[i] = pvalue(ExactOneSampleKSTest(vec(new_samples[i]) .+ shift,dist))
    print("KS : $i","\n")
end

diffMat = deepcopy(plotMat)
for i in 3:size(diffMat,2)
    diffMat[:,i] -= diffMat[:,1]
end
diffMat = diffMat[:,3:end]

density_label = ["Theoretical $dist_name" "Gaussian kernel of input data" "L_$m estimation" "L_$m projection of the estimated GC"]
diff_label = ["L_$m estimation" "L_$m projection of the estimated GC"]

println("Plotting...")
p0 = Plots.plot(x,plotMat,title = "Densities",label =density_label)
p1 = Plots.plot(x,diffMat,title = "Difference to the true density",label =diff_label)
y = ones(3)
title = Plots.scatter(y,
                        marker=0,
                        markeralpha=0,
                        annotations=(2,
                                        y[2],
                                        Plots.text("Univariate $dist_name, $N samples, $n_gammas gammas, m = $(m[1])")),
                                        #Plots.text("Univariate $dist_name, $N samples, $n_gammas gammas, m = $(m[1]), $(Total_time) seconds")),
                        axis=nothing,
                        legend=false,
                        border=:none,
                        size=(200,100))

new_sample = deepcopy(sample)
Random.rand!(moschdist,new_sample)
p2 = StatsPlots.qqplot(Float64.(vec(log.(sample))), Float64.(vec(log.(new_sample))), qqline = :fit, title="Empirical QQplot")

p3 = StatsPlots.histogram(p_values,bins=20,legend=nothing,title="KS test: histogram of p-values of $(N_ks_tests) resamples", yaxis=nothing)
                        

p = Plots.plot(
    title,
    Plots.plot(p0, p2, p1, p3,layout = (2,2)),
    layout=Plots.grid(2,1,heights=[0.01,0.99]),
    size=[1920,1080]
)
Plots.savefig(p,"ecgf_univ.png")