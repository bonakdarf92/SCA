using LinearAlgebra, Statistics, Compat, Plots
gr(fmt=:png)

n = 100
ϵ = randn(n)
 
ϵ_sum = 0.0
m = 5

for ϵ_val in e[1:m]
    global ϵ_sum = ϵ_sum + ϵ_val
end


ϵ_mean = mean(ϵ[1:m])
ϵ_mean = sum(ϵ[1:m]) / m