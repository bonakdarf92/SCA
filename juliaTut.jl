using Printf
using BenchmarkTools

function sphere_vol(r)
    return 4/3*pi*r^3
end

quadratic(a,sqr_term, b) = (-b+sqr_term)/2a

function quadratic2(a::Float64, b::Float64, c::Float64)
    sqr_term = sqrt(b^2-4*a*c)
    r1 = quadratic(a,sqr_term,b)
    r2 = quadratic(a,-sqr_term,b)
    r1, r2
end

@btime vol = sphere_vol(3)

@printf "Volumen = %0.3f\n" vol 

@btime quad1, quad2 = quadratic2(2.0,-2.0,-12.0)
println("Ergebnis 1: ", quad1)
println("Ergebnis 2: ", quad2)

function printsum(a)
    println(summary(a), ":", repr(a))
end

printsum([1,2,3])
