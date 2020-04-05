using OhMyREPL, Plots, Gurobi
using LinearAlgebra, GraphRecipes
using Convex, JLD2, FileIO, NPZ
using SparseArrays, LightGraphs, Revise
using Distributions, SymEngine
using PyCall, ColorSchemes


Plots.pyplot()
ps = pyimport("pygsp")
np = pyimport("numpy")

children(m::Module) = filter((x) ->typeof(eval(x)) <: Module && x!= :Main,
                names(Main, imported=true))



#pyData15 = np.load("Darmstadt_verkehr/SensorData_Sensor_Small_15_01_2020_Counts.npz", allow_pickle=true)
#sigs15 = get(pyData15,"arr_0");
#DD = convert(Dict{String,Any}, sigs15[]);
#A3 = get(DD["A003"],"signals");
#A7 = get(DD["A007"],"signals");
#zuw = zeros(1440,30);
#pos = convert.(Array{Float64},pos);
@load "A3A7_matrices.jld2"

function nansum(x::AbstractArray{T}) where T<:AbstractFloat
    if length(x) == 0
        result = zero(eltype(x))
    else
        result = convert(eltype(x), NaN)
        for i in x
            if !isnan(i)
                if isnan(result)
                    result = i
                else
                    result += i
                end
            end
        end
    end
    if isnan(result)
        @warn "All elements of Array are NaN!"
    end
    return result    
end

function nanmaximum(x::AbstractArray{T}) where T<:AbstractFloat
    result = convert(eltype(x), NaN)
    for i in x
        if !isnan(i)
            if (isnan(result) || i > result)
                result = i
            end
        end
    end
    return result
end

G = ps.graphs.Graph(A_con);
G.set_coordinates(pos)

Deg = Diagonal(sum(A_con, dims=2)[:]);
Deg_inv = Diagonal(inv.(sqrt.(diag(Deg)))[:]);
L_init = Deg - A_con;
L_norm = Diagonal(ones(30)[:]) - Deg_inv * A_con * Deg_inv;
Λ, V = eigen(L_init)

x_pos = [pos[k][1] for k in 1:length(pos)]
y_pos = [pos[k][2] for k in 1:length(pos)]

cM = hcat(1:25,ColorSchemes.hot[1:4:100])
cS = [cM[Int.(x)+1,2] for x in zuw[1,:]]

graphplot(A_con, x=x_pos,y=y_pos,markersize=3,names=1:30,fontsize=10,
        nodeshape=:circle,curves=true,linecolor=:black,nodecolor=cS)

function crossPlot(i::Int)
    cS = [cM[Int.(x)+1,2] for x in zuw[i,:]]
    graphplot(A_con, x=x_pos,y=y_pos,markersize=3,names=1:30,fontsize=10,
        nodeshape=:circle,curves=true,linecolor=:black,nodecolor=cS)
end

function plot_signal(y, direc)
    cS = [cM[round(Int, x) + 1, 2] for x in direc*y.value]
    graphplot(A_con, x=x_pos,y=y_pos,markersize=3,names=1:30,fontsize=10,
        nodeshape=:circle,curves=true,linecolor=:black,nodecolor=cS)
end
# TODO
function simulate_network(beginn::Int, ende::Int, name::String,fps=30)
    anim = @animate for i ∈ beginn:ende
        crossPlot(i)
    end
    mp4(anim,string(name,".mp4"), fps);
end


μ = 0.01
x = Variable(51,Positive())
s = Variable(51,Positive())
p = Variable(51,Negative())

Dp = max.(D, 0);
Dm = min.(D, 0);

problem = minimize(sumsquares(-Dm*(x+s) - zuw[2,:]) +
                    sumsquares(Dp*(x-p) - zuw[1,:]) + 
                    μ * (norm_1(x) + norm_1(s) + norm_1(p)))

blind = zeros(30,1);
blindSpots = [6,7,11,12,16,17,18,23,24,25];
blind[blindSpots] .= 1;

x_val = zeros(51,1440);
s_val = zeros(51,1440);
p_val = zeros(51,1440);
zuwNoNan = zuw;
zuwNoNan[isnan.(zuw)] .= 0;
function optim_day()
    for k in 1:1439
        fix!(y1,zuwNoNan[k,:]);
        fix!(y2,zuwNoNan[k+1,:]);
        problem2 = minimize(norm_1(-Dm*(x+s+p) - y2) + 
                        norm_1(Dp*(x) - y1) + 
                        μ * (norm_1(x) + 5*norm_1(s + p)))
        solve!(problem2, Gurobi.Optimizer(),verbose=false,warmstart=true)
        x_val[:,k] = x.value;
        s_val[:,k] = s.value;
        p_val[:,k] = p.value;
        println("Time step %", k)
    end
end

rec_p = Dp*(x_val + s_val);
rec_m = Dm*p_val;
U_norm, Λ_norm, V_norm = svd(L_norm);
Λ_nn, U_nn = eigen(L_norm);


function save_ws(wsname::string)
    packages = [string(k) for k in children(Main)];
    save(string(wsname,".jld2"),
            Dict("A_con"=>A_con,"D"=>D, "pos_Sen"=>pos_Sen,
                 "A3"=>A3,"A7"=>A7,"senID"=>senID,"sigs_A"=>sigs_A,
                 "pos"=>pos,"zuw"=>zuw, "zuwNoNan"=>zuwNoNan,
                 "packages"=>packages))
end

function load_ws(pack::Array{String,1})
   return 0
end

