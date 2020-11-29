## References
# - [1] Zhan, Zhang, and Chung. Adaptive particle swarm optimization, IEEE Transactions on Systems, Man, and Cybernetics, Part B: CyberneticsVolume 39, Issue 6 (2009): 1362-1381

function mpso(f, initial_x, lower, upper, n_particles; maxiter=1000)
    iteration = 0

    T = eltype(initial_x)

    n = length(initial_x)
    # TODO do we even need a lower((n) upper    ) that is different from method.lower(upper)
    # do some checks on input parameters
    limit_search_space = false

    c1 = T(2)
    c2 = T(2)
    w = T(1)

    X = [initial_x*T(0) for i = 1:n_particles]
    V = [initial_x*T(0) for i = 1:n_particles]
    X_best = [initial_x*T(0) for i = 1:n_particles]
    dx = initial_x*T(0)
    score = zeros(T, n_particles)
    x = copy(initial_x)
    best_score = zeros(T, n_particles)
    x_learn = copy(initial_x)
    current_state = 0

    # spread the initial population uniformly over the whole search space
    for i in 1:n_particles
        for j in 1:n
            ww = upper[j] - lower[j]
            X[i][j] = lower[j] + ww * rand(T)
            X_best[i][j] = X[i][j]
            V[i][j] = ww * (rand(T) * T(2) - T(1)) / 10
        end
    end

    X[1] .= initial_x
    X_best[1] .= initial_x
    best_f= T(0.0)
    for i = 1:maxiter
        limit_X!(X, lower, upper, n_particles, n)
        f(score, X)

        if iteration == 0
            copyto!(best_score, score)
            best_f = Base.minimum(score)
        end
        best_f = housekeeping!(score,
                                  best_score,
                                  X,
                                  X_best,
                                  x,
                                  best_f,
                                  n_particles)
        # Elitist Learning:
        # find a new solution named 'x_learn' which is the current best
        # solution with one randomly picked variable being modified.
        # Replace the current worst solution in X with x_learn
        # if x_learn presents the new best solution.
        # In all other cases discard x_learn.
        # This helps jumping out of local minima.
        worst_score, i_worst = findmax(score)
        for k in 1:n
            x_learn[k] = x[k]
        end
        random_index = rand(1:n)
        random_value = randn()
        sigma_learn = 1 - (1 - 0.1) * iteration / maxiter

        r3 = randn() * sigma_learn

        x_learn[random_index] = x_learn[random_index] + (upper[random_index] - lower[random_index]) / 3.0 * r3

        if x_learn[random_index] < lower[random_index]
            x_learn[random_index] = lower[random_index]
        end

        score_learn = f, x_learn
        if score_learn[1] < best_f
            best_f = score_learn * 1.0
            for j in 1:n
                X_best[j, i_worst] = x_learn[j]
                X[j, i_worst] = x_learn[j]
                x[j] = x_learn[j]
            end
            score[i_worst] = score_learn
            best_score[i_worst] = score_learn
        end

        # TODO find a better name for _f (look inthe paper, it might be called f there)
        current_state, _f = get_swarm_state(X, score, x, current_state)
        w, c1, c2 = update_swarm_params!(c1, c2, w, current_state, _f)
        update_swarm!(X, X_best, x, n, n_particles, V, w, c1, c2)
        iteration += 1

    end
    best_f, x
end

function update_swarm!(X, X_best, best_point, n, n_particles, V,
                       w, c1, c2)

  Tx = eltype(first(X))
  # compute new positions for the swarm particles
  for i in 1:n_particles
      for j in 1:n
          r1 = rand(Tx)
          r2 = rand(Tx)
          vx = X_best[i][j] - X[i][j]
          vg = best_point[j] - X[i][j]
          V[i][j] = V[i][j]*w + c1*r1*vx + c2*r2*vg
          X[i][j] = X[i][j] + V[i][j]
      end
    end
end

function get_mu_1(f::Tx) where Tx
    if Tx(0) <= f <= Tx(4)/10
        return Tx(0)
    elseif Tx(4)/10 < f <= Tx(6)/10
        return Tx(5) * f - Tx(2)
    elseif Tx(6)/10 < f <= Tx(7)/10
        return Tx(1)
    elseif Tx(7)/10 < f <= Tx(8)/10
        return -Tx(10) * f + Tx(8)
    else
        return Tx(0)
    end
end

function get_mu_2(f::Tx) where Tx
    if Tx(0) <= f <= Tx(2)/10
        return Tx(0)
    elseif Tx(2)/10 < f <= Tx(3)/10
        return Tx(10) * f - Tx(2)
    elseif Tx(3)/10 < f <= Tx(4)/10
        return Tx(1)
    elseif Tx(4)/10 < f <= Tx(6)/10
        return -Tx(5) * f + Tx(3)
    else
        return Tx(0)
    end
end

function get_mu_3(f::Tx) where Tx
    if Tx(0) <= f <= Tx(1)/10
        return Tx(1)
    elseif Tx(1)/10 < f <= Tx(3)/10
        return -Tx(5) * f + Tx(3)/2
    else
        return Tx(0)
    end
end

function get_mu_4(f::Tx) where Tx
    if Tx(0) <= f <= Tx(7)/10
        return Tx(0)
    elseif Tx(7)/10 < f <= Tx(9)/10
        return Tx(5) * f - Tx(7)/2
    else
        return Tx(1)
    end
end

function get_swarm_state(X, score, best_point, previous_state)
    # swarm can be in 4 different states, depending on which
    # the weighing factors c1 and c2 are adapted.
    # New state is not only depending on the current swarm state,
    # but also from the previous
    n_particles = length(X)
    n = length(first(X))
    Tx = eltype(first(X))
    f_best, i_best = findmin(score)
    d = zeros(Tx, n_particles)
    for i in 1:n_particles
        dd = Tx(0)
        for k in 1:n_particles
            for dim in 1:n
                @inbounds ddd = (X[i][dim] - X[k][dim])
                dd += ddd * ddd
            end
        end
        d[i] = sqrt(dd)
    end
    dg = d[i_best]
    dmin = Base.minimum(d)
    dmax = Base.maximum(d)

    f = (dg - dmin) / max(dmax - dmin, sqrt(eps(Tx)))

    mu = zeros(Tx, 4)
    mu[1] = get_mu_1(f)
    mu[2] = get_mu_2(f)
    mu[3] = get_mu_3(f)
    mu[4] = get_mu_4(f)
    best_mu, i_best_mu = findmax(mu)
    current_state = 0

    if previous_state == 0
        current_state = i_best_mu
    elseif previous_state == 1
        if mu[1] > 0
            current_state = 1
        else
          if mu[2] > 0
              current_state = 2
          elseif mu[4] > 0
              current_state = 4
          else
              current_state = 3
          end
        end
    elseif previous_state == 2
        if mu[2] > 0
            current_state = 2
        else
          if mu[3] > 0
              current_state = 3
          elseif mu[1] > 0
              current_state = 1
          else
              current_state = 4
          end
        end
    elseif previous_state == 3
        if mu[3] > 0
            current_state = 3
        else
          if mu[4] > 0
              current_state = 4
          elseif mu[2] > 0
              current_state = 2
          else
              current_state = 1
          end
        end
    elseif previous_state == 4
        if mu[4] > 0
            current_state = 4
        else
            if mu[1] > 0
                current_state = 1
            elseif mu[2] > 0
                current_state = 2
            else
                current_state = 3
            end
        end
    end
    return current_state, f
end

function update_swarm_params!(c1, c2, w, current_state, f::T) where T

    delta_c1 = T(5)/100 + rand(T) / T(20)
    delta_c2 = T(5)/100 + rand(T) / T(20)

    if current_state == 1
        c1 += delta_c1
        c2 -= delta_c2
    elseif current_state == 2
        c1 += delta_c1 / 2
        c2 -= delta_c2 / 2
    elseif current_state == 3
        c1 += delta_c1 / 2
        c2 += delta_c2 / 2
    elseif current_state == 4
        c1 -= delta_c1
        c2 -= delta_c2
    end

    if c1 < T(3)/2
        c1 = T(3)/2
    elseif c1 > T(5)/2
        c1 = T(5)/2
    end

    if c2 < T(3)/2
        c2 = T(5)/2
    elseif c2 > T(5)/2
        c2 = T(5)/2
    end

    if c1 + c2 > T(4)
        c_total = c1 + c2
        c1 = c1 / c_total * 4
        c2 = c2 / c_total * 4
    end

    w = 1 / (1 + T(3)/2 * exp(-T(26)/10 * f))
    return w, c1, c2
end

function housekeeping!(score, best_score, X, X_best, best_point,
                       F, n_particles)
    n = length(X)
    for i in 1:n_particles
        if score[i] <= best_score[i]
            best_score[i] = score[i]
            X_best[i] .= X[i]

            if score[i] <= F
              	best_point .= X[i]
              	F = score[i]
            end
        end
    end
    return F
end

function limit_X!(X, lower, upper, n_particles, n)
    # limit X values to boundaries
    for i in 1:n_particles
        X[i] .= min.(max.(X[i], lower), upper)
    end
    nothing
end


rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
function f(F, X)
    i = 1
    for x in X
        F[i] = rosenbrock(x)
        i+=1
    end
end
mpso(f, zeros(2), fill(-1.0, 2), fill(2.0, 2), 5)
