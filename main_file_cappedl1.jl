using LinearAlgebra
using StatsBase
using Distributions
using BenchmarkTools
using SparseArrays
using Random


N = 2000                # Number of 
K = 4000                # Number of 
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
rng = MersenneTwister(1234)

for s in 1:Sample
    A, b, μ, x0 = parameter(N, K, σ)
    μ_vec = μ * ones(K,1)
    x_g = zeros(K,1)
    residual_g = A * x_g - b
    ∇_g = residual_g' * A
    val_g[1] = 4.0
    println(float(residual_g'*residual_g))
    val_g[1] = Matrix(0.5 * residual_g'*residual_g)#+ μ_vec' * minimum(abs(x_g), θ_vec)
    error_g[s,1] = norm(x_g - x0) / norm(x0)
    println("Proximal MM Algorithmus: iteration 0 mit Wert", val_g[s,1])

end


A = sprandn(100,100,0.1)
x = 5*ones(100,)
c = A*x


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

