using OhMyREPL, Plots, Gurobi
using LinearAlgebra, GraphRecipes
using Convex, JLD2, FileIO, NPZ
using SparseArrays, LightGraphs, Revise
using Distributions, SymEngine
using PyCall, ColorSchemes


#Plots.pyplot()
#ps = pyimport("pygsp")
#np = pyimport("numpy")

children(m::Module) = filter(
    (x) -> typeof(eval(x)) <: Module && x != :Main,
    names(Main, imported = true),
)



#pyData15 = np.load("Darmstadt_verkehr/SensorData_Sensor_Small_15_01_2020_Counts.npz", allow_pickle=true)
#sigs15 = get(pyData15,"arr_0");
#DD = convert(Dict{String,Any}, sigs15[]);
#A3 = get(DD["A003"],"signals");
#A7 = get(DD["A007"],"signals");
#zuw = zeros(1440,30);
#pos = convert.(Array{Float64},pos);
@load "A3A7_matrices.jld2"

function nansum(x::AbstractArray{T}) where {T<:AbstractFloat}
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

function nanmaximum(x::AbstractArray{T}) where {T<:AbstractFloat}
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

#G = ps.graphs.Graph(A_con);
#G.set_coordinates(pos)

Deg = Diagonal(sum(A_con, dims = 2)[:]);
Deg_inv = Diagonal(inv.(sqrt.(diag(Deg)))[:]);
L_init = Deg - A_con;
L_norm = Diagonal(ones(30)[:]) - Deg_inv * A_con * Deg_inv;
Λ, V = eigen(L_init)

x_pos = [pos[k][1] for k = 1:length(pos)]
y_pos = [pos[k][2] for k = 1:length(pos)]

cM = hcat(1:25, ColorSchemes.hot[1:4:100])
cS = [cM[Int.(x)+1, 2] for x in zuw[1, :]]

graphplot(
    A_con,
    x = x_pos,
    y = y_pos,
    markersize = 3,
    names = 1:30,
    fontsize = 10,
    nodeshape = :circle,
    curves = true,
    linecolor = :black,
    nodecolor = cS,
)

function crossPlot(i::Int)
    cS = [cM[Int.(x)+1, 2] for x in zuw[i, :]]
    graphplot(
        A_con,
        x = x_pos,
        y = y_pos,
        markersize = 3,
        names = 1:30,
        fontsize = 10,
        nodeshape = :circle,
        curves = true,
        linecolor = :black,
        nodecolor = cS,
    )
end

function plot_signal(y, direc)
    cS = [cM[round(Int, x)+1, 2] for x in direc * y.value]
    graphplot(
        A_con,
        x = x_pos,
        y = y_pos,
        markersize = 3,
        names = 1:30,
        fontsize = 10,
        nodeshape = :circle,
        curves = true,
        linecolor = :black,
        nodecolor = cS,
    )
end

function plot_signal(y)
    cS = [cM[round(Int, x)+1, 2] for x in y]
    graphplot(
        A_con,
        x = x_pos,
        y = y_pos,
        markersize = 3,
        names = 1:30,
        fontsize = 10,
        nodeshape = :circle,
        curves = true,
        linecolor = :black,
        nodecolor = cS,
    )
end
# TODO
function simulate_network(beginn::Int, ende::Int, name::String, fps = 30)
    anim = @animate for i ∈ beginn:ende
        crossPlot(i)
    end
    mp4(anim, string(name, ".mp4"), fps)
end


μ = 0.01
x = Variable(51, Positive())
s = Variable(51, Positive())
p = Variable(51, Negative())

Dp = max.(D, 0);
Dm = min.(D, 0);

problem = minimize(
    sumsquares(-Dm * (x + s) - zuw[2, :]) +
    sumsquares(Dp * (x - p) - zuw[1, :]) +
    μ * (norm_1(x) + norm_1(s) + norm_1(p)),
)

blind = zeros(30, 1);
blindSpots = [6, 7, 11, 12, 16, 17, 18, 23, 24, 25];
blind[blindSpots] .= 1;

x_val = zeros(51, 1440);
s_val = zeros(51, 1440);
p_val = zeros(51, 1440);
zuwNoNan = zuw;
zuwNoNan[isnan.(zuw)] .= 0;
function optim_day()
    for k = 1:1439
        fix!(y1, zuwNoNan[k, :])
        fix!(y2, zuwNoNan[k+1, :])
        problem2 = minimize(
            norm_1(-Dm * (x + s + p) - y2) +
            norm_1(Dp * (x) - y1) +
            μ * (norm_1(x) + 5 * norm_1(s + p)),
        )
        solve!(problem2, Gurobi.Optimizer(), verbose = false, warmstart = true)
        x_val[:, k] = x.value
        s_val[:, k] = s.value
        p_val[:, k] = p.value
        println("Time step %", k)
    end
end

rec_p = Dp * (x_val + s_val);
rec_m = Dm * p_val;
U_norm, Λ_norm, V_norm = svd(L_norm);
Λ_nn, U_nn = eigen(L_norm);

"""
function save_ws(wsname::string)
    packages = [string(k) for k in children(Main)]
    save(
        string(wsname, ".jld2"),
        Dict(
            "A_con" => A_con,
            "D" => D,
            "pos_Sen" => pos_Sen,
            "A3" => A3,
            "A7" => A7,
            "senID" => senID,
            "sigs_A" => sigs_A,
            "pos" => pos,
            "zuw" => zuw,
            "zuwNoNan" => zuwNoNan,
            "packages" => packages,
        ),
    )
end
"""
#function load_ws(pack::Array{String,1})
#    return 0
#end
A_pap = [0 1 1 0 0; 0 0 0 1 1; 0 0 0 0 1; 1 0 0 0 0; 1 0 0 1 0];
Ap = DiGraph(A_pap);
Dpp = incidence_matrix(Ap);
Dplus = Matrix(max.(-Dpp, 0));
Dminus = Matrix(min.(-Dpp, 0));
x_star = Variable(8, Positive());
R = [
    1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
    0.0 0.0 0.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0
    0.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0
    0.0 0.0 0.0 0.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0
    0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0
    0.0 0.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0
    0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
]
c1 = x_star[1] + x_star[2] - x_star[7] - x_star[8] - 1;
c2 = x_star[3] + x_star[4] - x_star[1];
c3 = x_star[5] - x_star[2];
c4 = x_star[7] - x_star[3] - x_star[6] + 1;
c5 = x_star[6] + x_star[8] - x_star[4] - x_star[5];
Dplus[:,6], Dplus[:,7], Dplus[:,8] = Dplus[:,8], Dplus[:,6], Dplus[:,7];
Dminus[:,6], Dminus[:,7], Dminus[:,8] = Dminus[:,8], Dminus[:,6], Dminus[:,7];
Dpp[:,6], Dpp[:,7], Dpp[:,8] = Dpp[:,8], Dpp[:,6], Dpp[:,7];

function find_branches(cc::Array{Int64,2},expo::Int64,debug=false,rc=true)
    #m = dp' * -dm                       # Hashimoto / non-backtracking matrix
    k = size(cc)                         # dimension of edges
    σ = findall(x -> x == 1, vec(cc))    # no. of succeding edges in Hashimoto
    I = Matrix(Diagonal(ones(k[1])))    # Identity Matrix
    out = zeros(k[1], sum(vec(cc)[σ]))
    rows = σ .% k[1]
    cols = round.(Int, σ / k[1], RoundUp)
    rows[rows.==0] .= 8
    if rc println("Cols: ",cols); println("Rows :", rows) end
    flag = 0
    counter = 0
    for k = 1:length(rows)
        if debug println("k = ", k, " flag = ", flag, " cols = ", cols[k]) end
        if cols[k] == flag
            if debug
                println(
                    "Selbe Spalte ... Setze Zeile ",
                    rows[k],
                    " Spalte ",
                    cols[k] + 1 + counter,
                )
            end
            out[rows[k], cols[k]+1+counter] = 1
            counter += 1
        else
            if debug
                println(
                    "Unterschiedlich  Setze Zeile ",
                    rows[k],
                    " Spalte ",
                    cols[k] + counter,
                )
            end
            out[rows[k], cols[k]+counter] = 1
        end
        if debug println("cols k: ", cols[k], "counter : ", counter) end
        flag = cols[k]
    end
    if expo == 1
        count = 1
        for t in cols
            out[:, count] += I[:, t]
            count += 1
        end
    else
        temp1, eR, eC = find_branches(cc^(expo-1), expo);
        for jj in 1:length(cols)
            sC, sR = cols[jj], rows[jj];
            preselect = findall(x-> x==1, cc'[:,sC]);

    if rc display(out) end
    return out, rows, cols
end

function find_routing(incidence::SparseMatrixCSC{Int64,Int64})
    n,m = size(incidence)
    Dp = Matrix(max.(-incidence,0));
    Dm = Matrix(min.(-incidence,0));
    c = Dp' * -Dm;
    d = copy(c);
    memory = zeros(n-1,1);
    for j in 1:n-1
        temp = findall(x -> x == 1, vec(c^j))
        if isempty(temp)
            break;
        end
        memory[j] = sum(vec(c^j)[temp])
    end
    R_width = sum(memory)+m;
    println(R_width)
    R = zeros(m,Int(R_width));
    R[1:m,1:m] = Matrix(Diagonal(ones(m)));
    memory = Int.(filter(!iszero,memory))
    start = Int(m);
    for j in 1:length(memory)
        println("Step :", j)
        temp = find_branches(c^j,j);
        println("Startp: ", start+1, "MemL: ", memory[j], "Index: ", start+memory[j], "Dim T: ", size(temp))
        #display(temp)
        R[:,start+1:start+memory[j]] = temp;
        start += memory[j]
        #display(temp)
    end
    return R
end
