#=
test:
- Julia version: 1.0.2
- Author: Von Mir
- Date: 2019-03-16
=#
using Pkg
using PyCall
using LinearAlgebra
using Formatting
using StatsBase
using Distributions
#@pyimport matplotlib.pyplot as plt


function soft_threshholding(q,t,K)
    x = max.(q-t, zeros(K)) - max.(-q - t, zeros(K))
    return x
end


function stela_lasso(A, y, µ, Maxiter)
    """
    STELA algorithm solves the following optimization problem:
        min_x 0.5*||y - A * x||^2 + mu * ||x||_1

    Reference:
        Sec. IV-C of [Y. Yang, and M. Pesavento, "A unified successive pseudoconvex approximation framework", IEEE Transactions on Signal Processing, 2017]
    Input Parameters:
        A :      N * K matrix,  dictionary
        y :      N * 1 vector,  noisy observation
        mu:      positive scalar, regularization gain
        MaxIter: (optional) maximum number of iterations, default = 1000

    Definitions:
        N : the number of measurements
        K : the number of features
        f(x) = 0.5 * ||y - A * x||^2
        g(x) = mu * ||x||_1

    Output Parameters:
        x:      K * 1 vector, the optimal variable that minimizes {f(x) + g(x)}
        objval: objective function value = f + g
        error:  specifies the solution precision (a smaller error implies a better solution), defined in (53) of the reference

    """
    if µ <= 0
        println("must be positive")
        return
    elseif size(A)[1] != size(y)[1]
        println("Number of rows in A must be equal to dimension of y")
        return
    end

    K = size(A)[1]
    t = A.*A
    AtA_diag = sum(t,dims=1)
    #println(size(AtA_diag))
    µ_vec = µ*ones(K)
    µ_vec_norm = µ_vec'./AtA_diag
    #println(µ_vec)
    #println(size(µ_vec_norm))
    x = zeros(K)
    objval = Array{Float64}(zeros(Maxiter+1))
    error = zeros(Maxiter+1)
    #CPU_Time = zeros(Maxiter+1)
    #print(size(A), size(x), size(y))
    #CPU_Time[1] = time_ns()
    #resid = (x'*A) - y
    resid = A*x - y
    #f_grad = resid*A
    f_grad = resid'*A
    #print(size(f_grad), size(AtA_diag), size(x))
    #CPU_Time[1] = (time_ns() - CPU_Time[1])/1.0e9

    f = 0.5 * resid*resid'
    g = µ * norm(x,1)
    setindex!(objval[:],f[1],1)
    #error[1] = norm(abs.(f_grad' - min.(max.( (f_grad - x')', -µ*ones(K)),µ*ones(K)),Inf)
    error[1] = norm(abs.(f_grad' - min.( max.((f_grad - x')', -µ*ones(K) ), µ*ones(K))),Inf)

    IterationOut = "{1:9}|{2:10}|{3:15}|{4:15}"
    printfmtln(IterationOut,"Iteration", "stepsize", "objval", "error")
    printfmtln(IterationOut, 1, "N/A", format(objval[1], width=7), format(error[1], width=7))
    for t = 1:Maxiter
        #CPU_Time[t+1] = time_ns()
        #println(size( (x'-f_grad./AtA_diag)'), size(µ_vec_norm'))
        #println(size(x), size(f_grad), size(AtA_diag), size(µ_vec_norm))
        #Bx = soft_threshholding((x - f_grad./AtA_diag)', µ_vec_norm', K)
        Bx = soft_threshholding((x'-f_grad./AtA_diag)', µ_vec_norm', K)
        #Bx = soft_threshholding( ( x - (f_grad./AtA_diag)' )', µ_vec_norm', K)
        x_diff = Bx - x
        # überprüfen
        Ax_diff = A * x_diff

        step_num = - (resid' * Ax_diff + (abs.(Bx) - abs.(x))'*µ_vec)
        #step_num = - (resid * Ax_diff + (abs.(Bx) - abs.(x))'*µ_vec)
        step_denom = Ax_diff' * Ax_diff
        step_size = max.(min.(step_num / step_denom, 1), 0)

        x = x + Ax_diff*step_size
        #println(size(step_size), size(Ax_diff), size(resid))
        resid = resid + (Ax_diff*step_size)

        # checken
        #println(size(A), size(resid))
        f_grad = (A*resid)'
        #f_grad = A * resid'
        #CPU_Time[t + 1] = (time_ns() - CPU_Time[t + 1] + CPU_Time[t])/1.0e9
        f = 0.5 * resid * resid'
        #f = 0.5 * resid'*resid
        g = µ * norm(x,1)
        objval[t+1] = f[1] + g
        #error[t+1] = norm(abs.(f_grad - min.(max.(f_grad - x, -µ*ones(K)),µ*ones(K))),Inf)
        error[t+1] = norm(abs.(f_grad' - min.( max.((f_grad - x')', -µ*ones(K) ), µ*ones(K))),Inf)
        #printfmtln(IterationOut, t+1, "N/A", format(objval[t+1], width=7), format(error[t+1], width=7), format(CPU_Time[t+1], precision=7))
        printfmtln(IterationOut, t+1, format(step_size[1],width = 4), format(objval[t+1], width=7), format(error[t+1], width=7))

        if error[t+1] < 1e-6
            objval = objval[1:t+2]
            #CPU_Time = CPU_Time[0:t+2]
            error = error[0:t+2]
            println("Succesfull")
            break
        elseif t == Maxiter
            println("Optimization not possibles with given amount iterations")
        end

    end
    return objval, error
end

N = 1000
K = 1000
A = rand(Normal(0,0.1),N, K)
#print(size(A))
dens = 0.01
x0 = zeros(K)
x0_pos = sample(collect(1:K), round(Int,K*dens))

for t = 1:round(Int,K*dens)
x0[x0_pos[t]] = rand(Normal(0,1))
end

sigma = 0.01
v = rand(Normal(0,sigma), N)'
y = (A*x0 + v')
#y = y'
#print(size(y))
µ = 0.01*norm((y'*A),Inf)

#y = rand(100)
#µ = 0.01
objval, error = stela_lasso(A, y, µ, 100)
println("fertig")
