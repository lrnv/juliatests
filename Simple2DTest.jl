import Random, Distributions, ThorinDistributions, Optim, LineSearches, Plots

#
# A simple test case:
alpha = [10, 10, 1, 5]
rates = [0.5 0.1; 0.1 0.5; 1 1; 2 2]
m = (25,25)
E = ThorinDistributions.get_coefficients(alpha, rates,m)

println("Launching optimisation...")
n_gammas = 4
d = 2
println("Minimum m : $(((d+1)n_gammas)^(1/d))")
par = Random.rand(3n_gammas+1) .- 1/2
obj = x -> ThorinDistributions.L2ObjectiveWithPenalty(x,E)
opt = Optim.Options(g_tol=1e-22,
                x_tol=1e-22,
                f_tol=1e-22,
                time_limit=1800.0, # 30min
                show_trace = true,
                allow_f_increases = true,
                iterations = 100000)
algo = Optim.ParticleSwarm()
program = Optim.optimize(obj, par, algo, opt; autodiff = :forward)
print(program)
par = Optim.minimizer(program)

println("Polishing with LBFGS...")
opt2 = Optim.Options(g_tol=1e-22,
                x_tol=1e-22,
                f_tol=1e-22,
                time_limit=1800.0, # 30min
                show_trace = true,
                allow_f_increases = false,
                iterations = 100000)
algo2 = Optim.LBFGS(linesearch = LineSearches.MoreThuente())
program2 = Optim.optimize(obj, par, algo2, opt2)
print(program2)
par = Optim.minimizer(program2)



alpha = par[1:n_gammas] .^2 #make them positives
rates = reshape(par[(n_gammas+1):3n_gammas],(n_gammas,2)) .^ 2 # make them positives
lambda = par[3n_gammas+1]
rez = hcat(alpha,rates)
rez = rez[sortperm(-rez[:,1]),:]
println("lambda = $lambda")
display(rez)

coefs = ThorinDistributions.get_coefficients(alpha,rates,m)
x = y = 0:0.5:40
tpl = (2, n_gammas)
fE = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([x, y], E))
g = (x,y)->convert(Float64,ThorinDistributions.laguerre_density([x, y], coefs))
p1 = Plots.plot(x, y, fE, legend=false, title = "Projection on L_$m", seriestype=:wireframe)
p2 = Plots.plot(x, y, g, legend=false, title = "Estimation in G_$tpl", seriestype=:wireframe)
p = Plots.plot(p1,p2, layout = (1,2), size=[1920,1024])
Plots.display(p)
Plots.savefig(p,"Simple2DTest_$(m)_$(n_gammas).png")

import Serialization
Serialization.serialize("Simple2DTest_$(m)_$(n_gammas).model",(alpha,rates))
