#=
test:
- Julia version: 1.0.2
- Author: Von Mir
- Date: 2019-03-16
=#
using Pkg
using LinearAlgebra
using Formatting
using StatsBase
using Distributions
using Plots
using BenchmarkTools


function soft_threshholding(q,t,K)
    x = max.(q-t, zeros(K)) - max.(-q - t, zeros(K))
    return x
end

function descent_dir(𝔹x, x, γ)
    return (𝔹x - x)*γ
end

function stepsize(ϵ, ∇Ax, 𝔹x, x, μ_vec)
    return max.(min.(- (ϵ' * ∇Ax + (abs.(𝔹x) - abs.(x))' * µ_vec) / ∇Ax' * ∇Ax, 1), 0)
end

function stepnum(ϵ, ∇Ax, 𝔹x, x, μ_vec)
    return (- (ϵ' * ∇Ax + (abs.(𝔹x) - abs.(x))' * µ_vec))
end

stepdom(∇Ax) = ∇Ax'*∇Ax


function own_stela!(x, ∇f, μ_vec_norm, K, A, ϵ, μ_vec, ν, objval, err, Maxiter)
    for t = 1:Maxiter
        𝔹x = soft_threshholding((x' - ))
    end
end

function kernel_stela!(x, ∇f, AtA_diag, µ_vec_norm, K, A, ϵ, µ_vec, objval, error, Maxiter)
    for t = 1:Maxiter
        𝔹x = soft_threshholding((x'- ∇f ./ AtA_diag)', µ_vec_norm', K)
        @inbounds δx = descent_dir(𝔹x, x, 1)#𝔹x - x
        @inbounds ∇Ax = A * δx
        #bla = descent_dir(𝔹x, x, 1)

        @inbounds step_num = stepnum(ϵ, ∇Ax, 𝔹x, x, μ_vec) #- (ϵ' * ∇Ax + (abs.(𝔹x) - abs.(x))' * µ_vec)
        @inbounds step_denom = stepdom(∇Ax) #∇Ax' * ∇Ax
        step_size =  max.(min.(step_num/ step_denom, 1), 0) #stepsize(ϵ, ∇Ax, 𝔹x, x, μ_vec)

        @inbounds x[:] +=  δx[:] * step_size #descent_dir(𝔹x, x, step_size)
        @inbounds ϵ[:] += ∇Ax * step_size

        @inbounds ∇f[:] = ϵ' * A
        @inbounds f = 0.5 * ϵ' * ϵ
        @inbounds g = µ * norm(x,1)
        @inbounds objval[t+1] = f[1] + g
        #setindex!(objval[:],f[1]+g,t+1)
        @inbounds error[t+1] = norm(abs.(∇f' - min.( max.((∇f - x')', -µ*ones(K) ), µ*ones(K))), Inf)
        #printfmtln(IterationOut, t+1, "N/A", format(objval[t+1], width=7), format(error[t+1], width=7), format(CPU_Time[t+1], precision=7))
        #printfmtln(IterationOut, t+1, format(step_size[1],width = 4), format(objval[t+1], width=7), format(error[t+1], width=7))

        if error[t+1] < 1e-6
            println("Succesfull")
            break
        elseif t == Maxiter
            println("Optimization not possibles with given amount iterations")
        end

    end
end


function stela_lasso(A::Array{Float64,2}, y::Vector{Float64}, µ::Float64, Maxiter::Int64)
    if µ <= 0
        println("must be positive")
        return
    elseif size(A)[1] != size(y)[1]
        println("Number of rows in A must be equal to dimension of y")
        return
    end

    K = size(A)[2]
    @inbounds AtA_diag = sum(A.*A,dims=1)
    µ_vec = µ*ones(K)
    @inbounds µ_vec_norm = µ_vec'./AtA_diag
    x = zeros(K)
    objval = Array{Float64}(zeros(Maxiter+1))
    error = zeros(Maxiter+1)
    @inbounds ϵ = A*x - y
    @inbounds ∇f = ϵ' * A
    @inbounds f = 0.5 * ϵ' * ϵ
    @inbounds g = µ * norm(x,1)
    setindex!(objval[:],f[1],1)
    @inbounds error[1] = norm(abs.(∇f' - min.( max.((∇f - x')', -µ*ones(K) ), µ*ones(K))),Inf)

    #IterationOut = "{1:9}|{2:10}|{3:15}|{4:15}"
    #printfmtln(IterationOut,"Iteration", "stepsize", "objval", "error")
    #printfmtln(IterationOut, 1, "N/A", format(objval[1], width=7), format(error[1], width=7))
    kernel_stela!(x, ∇f, AtA_diag, µ_vec_norm, K, A, ϵ, µ_vec, objval, error, Maxiter)
    # for t = 1:Maxiter
    #     𝔹x = soft_threshholding((x'- ∇f ./ AtA_diag)', µ_vec_norm', K)
    #     δx = 𝔹x - x
    #     ∇Ax = A * δx
    #
    #     step_num = - (ϵ' * ∇Ax + (abs.(𝔹x) - abs.(x))' * µ_vec)
    #     step_denom = ∇Ax' * ∇Ax
    #     step_size = max.(min.(step_num / step_denom, 1), 0)
    #
    #     x += δx * step_size
    #     ϵ += ∇Ax * step_size
    #
    #     ∇f = ϵ' * A
    #     f = 0.5 * ϵ' * ϵ
    #     g = µ * norm(x,1)
    #     objval[t+1] = f[1] + g
    #     error[t+1] = norm(abs.(∇f' - min.( max.((∇f - x')', -µ*ones(K) ), µ*ones(K))), Inf)
    #     #printfmtln(IterationOut, t+1, "N/A", format(objval[t+1], width=7), format(error[t+1], width=7), format(CPU_Time[t+1], precision=7))
    #     #printfmtln(IterationOut, t+1, format(step_size[1],width = 4), format(objval[t+1], width=7), format(error[t+1], width=7))
    #
    #     if error[t+1] < 1e-6
    #         println("Succesfull")
    #         break
    #     elseif t == Maxiter
    #         println("Optimization not possibles with given amount iterations")
    #     end
    #
    # end
    return objval, x, error
end

N = 40000
K = 3000
A = rand(Normal(0,0.1),N, K)

dens = 0.01
x0 = zeros(K)
x0_pos = sample(collect(1:K), round(Int,K*dens))

for t = 1:round(Int,K*dens)
    x0[x0_pos[t]] = rand(Normal(0,1))
end

sigma = 0.01
v = rand(Normal(0,sigma), N)'
y = (A*x0 + v')

µ = 0.01*norm((y'*A),Inf)

@benchmark objval , x, err = stela_lasso(A, y, µ, 100)
println("fertig")
