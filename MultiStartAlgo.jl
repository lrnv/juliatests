include("MGC.jl")
setprecision(MGCLaguerre.P.PREC)
import Optim, Plots, Random, Distributions, LineSearches, KernelDensity

# The multivariate lognormal test case:
Random.seed!(123)
N = 1000
dist = Distributions.LogNormal(0,1)
sample = Array{Float64}(undef, 1,N)
Random.rand!(dist,sample)
sample = convert.(BigFloat,sample)
m = (20,)
println("Computing empirical coefs of standard Log normal, $N samples (may take some time..)")
E = MGCLaguerre.empirical_coefs(sample,m)
println("Done")
d = 1
n_gammas = 40

obj = x -> MGCLaguerre.L2Objective(x,E)

function uniform(dim::Int, lb::Array{Float64, 1}, ub::Array{Float64, 1})
    arr = rand(Float64, dim)
    @inbounds for i in 1:dim; arr[i] = arr[i] * (ub[i] - lb[i]) + lb[i] end
    return arr
end

function initialize_particles(problem, population)
    dim = problem.dim
    lb = problem.lb
    ub = problem.ub
    cost_func = problem.cost_func

    gbest_position = uniform(dim, lb, ub)
    gbest = Gbest(gbest_position, cost_func(gbest_position))

    particles = []
    for i in 1:population
        position = uniform(dim, lb, ub)
        velocity = zeros(dim)
        cost = cost_func(position)
        best_position = copy(position)
        best_cost = copy(cost)
        push!(particles, Particle(position, velocity, cost, best_position, best_cost))

        if best_cost < gbest.cost
            gbest.position = copy(best_position)
            gbest.cost = copy(best_cost)
        end
    end
    return gbest, particles
end

function minby(itr; by=identity, init=nothing)
     init = isnothing(init) ? pop!(itr) : init
     mapreduce(x->(by(x)=>x), (x,y)->(first(x)>first(y)) ? y : x, itr; init=by(init)=>init).second
end

function PSO(problem; max_iter=100,population=100,c1=1.4962,c2=1.4962,w=0.7298,wdamp=1.0)
    dim = problem.dim
    lb = problem.lb
    ub = problem.ub
    cost_func = problem.cost_func

    gbest, particles = initialize_particles(problem, population)

    # main loop
    for iter in 1:max_iter
        gbests = fill((gbest.cost, 0), Threads.nthreads())
        @threads for i in 1:population
            particles[i].velocity .= w .* particles[i].velocity .+
                c1 .* rand(dim) .* (particles[i].best_position .- particles[i].position) .+
                c2 .* rand(dim) .* (gbest.position .- particles[i].position)

            particles[i].position .= particles[i].position .+ particles[i].velocity
            particles[i].position .= max.(particles[i].position, lb)
            particles[i].position .= min.(particles[i].position, ub)

            particles[i].cost = cost_func(particles[i].position)

            if particles[i].cost < particles[i].best_cost
                particles[i].best_position = copy(particles[i].position)
                particles[i].best_cost = copy(particles[i].cost)

                if particles[i].best_cost < gbests[Threads.threadid()][1]
                    gbests[Threads.threadid()] = (particles[i].best_cost, i)
                end
            end
        end
        gbest1 = minby(gbests, by = x -> x[1])
        if gbest1[2] != 0
            idx = gbest1[2]
            gbest.position = copy(particles[idx].best_position)
            gbest.cost = copy(particles[idx].best_cost)
        end
        w = w * wdamp
        if iter % 50 == 1
            println("Iteration " * string(iter) * ": Best Cost = " * string(gbest.cost))
            println("Best Position = " * string(gbest.position))
            println()
        end
    end
    gbest, particles
end
