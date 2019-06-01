using LinearAlgebra
using StatsBase
using Distributions
using BenchmarkTools
using SparseArrays
using Random
using Revise


N = 200                # Number of
K = 400                # Number of
σ = 0.01                # density
θ = 0.001               # direction ?
Sample = 1              # number of simulation
max_iter_j = 30         # number of maximal iterations in j loop
max_iter_m = 10         # number of maximal iterations in m loop
max_iter_g = 400        # number of maximal iterations in g loop

# initialzing 0 vectors for storing values of each loop
val_j = zeros(Float64,Sample, max_iter_j+1)
val_g = zeros(Float64,Sample, max_iter_g+1)
val_m = zeros(Float64,Sample, max_iter_m+1)

# initializing 0 vector for storing calculation error
error_j = zeros(Float64, Sample, max_iter_j)
error_g = zeros(Float64, Sample, max_iter_g)
θ_vec = θ*ones(Float64, K, 1)

# set random seed for reprocucability
rng = MersenneTwister(42)

# function to generate parameters A, b, mu and x
function parameter(M::Int64,N::Int64,σ::Float64)
    x_orig = sprandn(N,1,σ)
    σ_2 = 0.0001
    A = randn(M,N)
    for n in 1:M
        A[n,:] = A[n,:] / norm(A[n,:])
    end
    b = A*x_orig + sqrt(σ_2) * randn(M,1)
    μ = 0.1 * maximum(abs.(A'*b))
    return A, b, μ, x_orig

end

function testloop(N,K,σ)
    σ = 0.01                # density
    θ = 0.001
    θ_vec = θ*ones(Float64, K, 1)
    for s in 1:Sample
        A, b, μ, x0 = parameter(N, K, σ)
        μ_vec = μ * ones(K,1)
        x_g = zeros(K,1)
        residual_g = A * x_g - b
        ∇_g = (residual_g' * A)'
        val_g[s,1] = (0.5*residual_g'*residual_g + μ_vec' * minimum!(abs.(x_g)[:], θ_vec[:]))[1]
        error_g[s,1] = norm(x_g - x0) / norm(x0)
        println("Proximal MM Algorithmus: iteration 0 mit Wert ", val_g[s,1])

        for t in 1:max_iter_g
            # line search for lipschitz constant
            global c = 1.0
            α = 0.5
            β = 2.0
            global flag = true
            while flag
                u_g = x_g - ∇_g./c
                x1 = sign.(u_g).*maximum!(θ_vec, abs.(u_g))
                h1 = 0.5 * (x1 - u_g).*(x1 - u_g) + μ * minimum!(abs.(x1), θ_vec)
                x2 = sign.(u_g).*minimum!(θ_vec, maximum!(zeros(K,1), abs.(u_g) - μ_vec./c ))
                h2 = 0.5 * (x2 - u_g).*(x2 - u_g) + μ * minimum!(abs.(x2), θ_vec)
                x_new = x1.*(h1 .<= h2) + x2.*(ones(K,1) - (h1 .<= h2))
                residual_new = A*x_new - b
                val_g_new = 0.5 * (residual_new'*residual_new) + μ_vec' * minimum!(abs.(x_new), θ_vec)
                println("Wert g_neu ist ",val_g_new, " letzter Teil ", val_g[s,t] - (α * c/2.0* ((x_new - x_g)' * (x_new - x_g)))[1], " momentaner Wert C ", c)
                if val_g_new[1]  <= (val_g[s,t+1] - (α * c/2.0* ((x_new - x_g)' * (x_new - x_g)))[1]) #|| isnan(val_g_new[1])
                    println("If Bedingung")
                    #println("x_g_alt -- ",x_g,"x_new --", x_new)
                    println("val_g_alt lautet ",val_g[s,t]," val_g_new lautet ", val_g_new)
                    x_g = x_new
                    val_g[s,t+1] = val_g_new[1]
                    #println(val_g)
                    residual_g = residual_new
                    ∇_g = (residual_g'*A)'
                    print(" If-Wert c pre: ", c)
                    c = 1.0
                    print(" If Wert c post: ", c)
                    println()
                    flag = false
                    break
                else
                    c = c * β
                    println("c is ", c)
                end
                println("Bin in der while schleife")
            end
            #println("Iteration ", t )
            println("Proximal MM Algorithmus: iteration ",t, " mit Wert ", val_g[s,t+1])
        end

    end
end


testloop(200,400,0.01)
