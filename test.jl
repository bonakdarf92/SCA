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
using Plots, PyCall
using BenchmarkTools
using NPZ
using Convex
using Gurobi
using LightGraphs
using LightXML

function findPos(att)
    counter = 1
    x_pos, y_pos = zeros(1,1), zeros(1,1)
    for m in junc
        for k in attributes(m)
            if (name(k) == "type") && (value(k) == att)
                counter += 1
                println("x_pos: ",attribute(m,"x"), " y_pos: ", attribute(m,"y"))                x_pos = hcat(x_pos, parse(Float64,attribute(m,"x")))
                y_pos = hcat(y_pos, parse(Float64,attribute(m,"y")))
            end
        end
    end
    println(counter)
    return x_pos, y_pos
end

function findNodes(xroot, childNode, att)
   counter = 1
   for m in xroot[childNode]
       for k in attributes(m)
           if (name(k) == att)
               counter += 1
           end
       end
   end
   println(counter)
end
function compare(c::Number)
    plot(A*x, label="Stela")
    plot!(A*x_cv.value, label="Gurobi")
    plot!(x0, label="Original")
    plot!(y, label="Measurement")
end

performance() = plot(plot(objval), plot(err), layout=(1,2))

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


function plot_spectorgram(s, fs, hw, overlap)
    S = spectrogram(s[:,1], convert(Int, hw * fs), round(Int, overlap * fs); window=DSP.hanning)
    heatmap(S.time, S.freq, DSP.pow2db(S.power))
    return S
end


#function own_stela!(x, ∇f, μ_vec_norm, K, A, ϵ, μ_vec, ν, objval, err, Maxiter)
#    for t = 1:Maxiter
#        �x = soft_threshholding((x' - ))
#    end
#end

function kernel_stela!(x, ∇f, AtA_diag, µ_vec_norm, K, A, ϵ, µ_vec, objval, error, Maxiter)
    for t = 1:Maxiter
        𝔹x = soft_threshholding((x'- ∇f ./ AtA_diag)' , µ_vec_norm', K)
        @inbounds δx = descent_dir(𝔹x, x, 1)#𝔹x - x
        @inbounds ∇Ax = A * δx
        #bla = descent_dir(𝔹x, x, 1)

        @inbounds step_num = stepnum(ϵ, ∇Ax, 𝔹x, x, μ_vec) #- (ϵ' * ∇Ax + (abs.(�x) - abs.(x))' * µ_vec)
        @inbounds step_denom = stepdom(∇Ax) #∇Ax' * ∇Ax
        step_size =  max.(min.(step_num/ step_denom, 1), 0) #stepsize(ϵ, ∇Ax, �x, x, μ_vec)

        @inbounds x[:] +=  δx[:] * step_size #descent_dir(𝔹x, x, step_size)
        @inbounds ϵ[:] += ∇Ax * step_size

        @inbounds ∇f[:] = ϵ' * A
        @inbounds f = 0.5 * ϵ' * ϵ
        @inbounds g = µ * norm(x,1)
        @inbounds objval[t+1] = f[1] + g
        #setindex!(objval[:],f[1]+g,t+1)
        IterationOut = "{1:9}|{2:10}|{3:15}|{4:15}"
        @inbounds error[t+1] = norm(abs.(∇f' - min.( max.((∇f - x')', -µ*ones(K) ), µ*ones(K))), Inf)
        #printfmtln(IterationOut, t+1, "N/A", format(objval[t+1], width=7), format(error[t+1], width=7), format(CPU_Time[t+1], precision=7))
        printfmtln(IterationOut, t+1, format(step_size[1],width = 4), format(objval[t+1], width=7), format(error[t+1], width=7))

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

    IterationOut = "{1:9}|{2:10}|{3:15}|{4:15}"
    printfmtln(IterationOut,"Iteration", "stepsize", "objval", "error")
    printfmtln(IterationOut, 1, "N/A", format(objval[1], width=7), format(error[1], width=7))
    kernel_stela!(x, ∇f, AtA_diag, µ_vec_norm, K, A, ϵ, µ_vec, objval, error, Maxiter)
    return objval, x, error
end

# N = 4000
# K = 3000
# A = rand(Normal(0,0.1),N, K)
A = npzread("PathDic_20_7.npz")["arr_0"];
Graph = npzread("DarmstadtJulia.npz")["arr_0"]
GG = DiGraph(Graph)
L = laplacian_matrix(GG)
dens = 0.01
x0 = zeros(75)
#x0_pos = sample(collect(1:K), round(Int, K*dens))
x0_pos = [20, 41, 74, 6, 16, 45, 68, 57, 15, 30, 11, 23, 43, 24]

for t = 1:14#round(Int,K*dens)
    x0[x0_pos[t]] = rand(20:40)#rand(Normal(0,1))
end

sigma = 0.01
v = rand(Normal(0,sigma), 75)'
y = x0 + rand(-2:5,75)#v #(A*x0 + v')
y[y.<0] .= 0

µ = 0.0017*norm((y'*A),Inf)

#@benchmark
objval , x, err = stela_lasso(A, y, µ, 500)

x_cv = Variable(size(A)[2]);
problem = minimize(0.5*sumsquares(y - A*x_cv) + μ * norm_1(x_cv), x_cv >=0)
solve!(problem, Gurobi.Optimizer)
x_prob = A*x ./ maximum(A*x)

println("fertig")
