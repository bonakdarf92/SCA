using OhMyREPL, Plots, Gurobi
using LinearAlgebra, GraphRecipes
using Convex, JLD2, FileIO, NPZ
using SparseArrays, LightGraphs, Revise
using Distributions, SymEngine
using PyCall, ColorSchemes
using Random, Formatting

#Plots.pyplot()
#ps = pyimport("pygsp")
np = pyimport("numpy")

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

function a3a7(filepath, sensorID, selectSensors)
    pyData = np.load(filepath, allow_pickle=true)
    sigs = get(pyData,"arr_0");
    tDic = convert(Dict{String,Any},sigs[]);
    tA3 = get(tDic["A003"],"signals");
    tA7 = get(tDic["A003"],"signals");
    tSigs = hcat(tA7,tA3);
    tzuw = zeros(1440,30);
    tzuw[:,sensorID] .= tSigs[:,Bool.(selectSensors)];
    return tzuw
end

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

A_con2 = copy(A_con)
A_con2[6:7,1] .= 0;
A_con2[11:12,13] .= 0;
A_con2[17:18,19] .= 0;
A_con2[26:27,30] .= 0;
DCon2 = incidence_matrix(DiGraph(A_con2));
Dp = max.(-DCon2, 0);
Dm = -min.(-DCon2, 0);

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
        A_con2,
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
        A_con2,
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
x = Variable(43)#, Positive())
s = Variable(43)
p = Variable(43, Negative())

problem = minimize(
    sumsquares(-Dm * (x + s) - zuw[2, :]) +
    sumsquares(Dp * (x - p) - zuw[1, :]) +
    μ * (norm_1(x) + norm_1(s) + norm_1(p)),
)

blind = zeros(30, 1);
blindSpots = [6, 7, 11, 12, 16, 17, 18, 23, 24, 25];
blind[blindSpots] .= 1;

x_val = zeros(43, 1440);
s_val = zeros(43, 1440);
p_val = zeros(43, 1440);
zuwNoNan = zuw;
zuwNoNan[isnan.(zuw)] .= 0;
Dx = [Dm; Dp; Dp; -Dp; -Dm];
Ds = [zeros(90,43); Dp; -Dm];
function optim_day(measurements)
    for k = 1:1439
        fix!(y1, measurements[k, :])
        fix!(y2, measurements[k+1, :])
        ym1 = copy(y1.value);
        ym2 = copy(y2.value);
        ym1[[6,7,11,12,17,18,26,27]] .= 0;
        ym2[[1,13,19,30]] .= 0;
        y_bar = vcat(y2,y1,(y2-y1),-ym1,-ym2)
        """
        problem2 = minimize(
            sum(huber(Dm * x - y2,1)) +
            sum(huber(Dp * x - y1,1)) +
            sum(huber(Dp * x - (y2-y1),1)) +
            sum(huber(-Dp * (x-s) + ym1,1)) +
            sum(huber(-Dm * (x+s) + ym2,1)) +
            μ * (norm_1(x) +5*norm_1(s))# + 50*norm_1(p))
        )
        """
        problem2 = minimize(norm_1(Dx*x + Ds*s - y_bar) + μ *(norm_1(x) + 5* norm_1(s) ))
        
        solve!(problem2, Gurobi.Optimizer(),verbose=0,warmstart=true)
        x_val[:, k] = x.value
        s_val[:, k] = s.value
        p_val[:, k] = zeros(43)#p.value
        println("Time step %", k)
    end
end

function stela_network(Y::Array{Float64,2}, ρ::Int64, maxiter::Int,R::Array{Float64,2} = 0)
    II = size(R)[2];
    N = size(R)[1];
    K = size(Y)[2];
    ρ_r = 10;
    rng = MersenneTwister(460); #

    #if R == 0
    #    println("Kein R übergeben")
    #    R = zeros(N,II);
    #    for k in 1:II
    #        R[rand(1:N), k] = 1;
    #        R[rand(1:N), k] = 1;
    #    end
    #end
    val_j = zeros(maxiter+1,1);
    time_j = zeros(maxiter+1,1);
    err_j = zeros(maxiter+1,1);
    #S0 = sprand(II,K,0.01); # density
    #tt = range(0,stop=π,length=K);
    #P0    = sqrt(100/II) * randn(N, ρ_r) + 5*ones(N,ρ_r);
    #Q0    = sqrt(100/K) * randn(ρ_r, K) .* sin.(tt)' + 5*ones(ρ_r, K);
    #P0    = randn(N, ρ_r) + 5*ones(N, ρ_r);
    #Q0    = randn(ρ_r, K) + 5*ones(ρ_r, K);

    #X0    = P0 * Q0; # perfect X

    #σ = 0.01;
    #V     = σ * randn(N,K); # noise
    #Y = X0 + R * S0 + V; # observation
    λ = 100 * 10^-3 * norm(Y); #spectral norm  former 2.5
    μ = 2 * 10^-1 * norm(R' * (Y), Inf); #    former 2

    #initial point (common for all algorithms)
    initial_P = randn(rng,N,ρ);# sqrt(100/II) * randn(N,ρ);
    initial_Q = randn(ρ,K); #sqrt(100/K) * randn(ρ,K);
    initial_S = zeros(II,K);
    val0 = 0.5 * norm(Y - initial_P * initial_Q - R * initial_S)^2 + 0.5 * λ * (norm(initial_P)^2 + norm(initial_Q)^2) + μ * norm(vec(initial_S), 1);
    #initial value
    #val0 = 0.5 * norm(Y - initial_P * initial_Q - R * initial_S)^2
    #     + 0.5 * λ * (norm(initial_P)^2 + norm(initial_Q)^2)
    #     + μ * norm(vec(initial_S), 1);

    # STELA algorithm: Initialization
    P           = initial_P;
    Q           = initial_Q;
    S           = initial_S;
    val_j[1,1]  = val0;
    time_j[1,1] = 0;
    d_DtD       = Diagonal(diag(R' * R));
    ϵ = P*Q + R*S - Y
    err_j[1,1] = abs(tr(P'*(ϵ*Q' + λ*P)) + tr(Q'*(P'*ϵ + λ*Q) ) + tr(S'*R'*ϵ));

    println("STELA, iteration ",  0 , ", time ", 0, ", value ", val_j[1,1] );

    for t in 1: maxiter

        Y_DS  = Y - R * S;
        P_new = Y_DS * Q' * (Q * Q' + λ * I(ρ))^-1;
        cP    = P_new - P;

        Q_new = (P' * P + λ * I(ρ))^-1 * P' * Y_DS;
        cQ    = Q_new - Q;

        G     = d_DtD * S - R' * (P * Q - Y_DS); # clear Y_DS
        S_new = d_DtD^-1 * (max.(G - μ * ones(II,K), zeros(II,K)) - max.(-G - μ * ones(II,K), zeros(II,K))); #clear G
        cS    = S_new - S;

        #-------------------- to calculate the stepsize by exact line search----------------
        A = cP * cQ;
        B = P * cQ + cP * Q + R * cS;
        C = P * Q + R * S - Y;

        a = 2 * sum(sum(A.^2,dims=1));
        b = 3 * sum(sum(A.*B,dims=1));
        c = sum(sum(B.^2,dims=1)) + 2 * sum(sum(A.*C,dims=1)) + λ * sum(sum(cP.^2,dims=1)) + λ * sum(sum(cQ.^2,dims=1));
        d = sum(sum(B.*C,dims=1)) + λ * sum(sum(cP.*P,dims=1)) + λ * sum(sum(cQ.*Q,dims=1)) + μ * (norm(vec(S_new),1) - norm(vec(S),1));

        #%calculating the stepsize by closed-form expression
        Σ_1      = (-(b/3/a)^3 + b*c/6/(a^2) - d/2/a);
        Σ_2      = c/3/a - (b/3/a)^2;
        Σ_3      = Σ_1^2 + Σ_2^3;
        Σ_3_sqrt = sqrt(Σ_3);
        #C1, C2 = zeros(4,1), zeros(4,1)

        if Σ_3 >= 0
            γ = Float64(cbrt(big(Σ_1 + Σ_3_sqrt)) + cbrt(big(Σ_1 - Σ_3_sqrt)) - b/3/a);
            #gamma = cbrt(Σ_1 + Σ_3_sqrt) + cbrt(Σ_1 - Σ_3_sqrt) - b/3/a;
        else
            C1 = 1;
            C1[4] =  -(Σ_1 + Σ_3_sqrt);
            C2 = 1;
            C2[4] = -(Σ_1 - Σ_3_sqrt);
            R = real(roots(Poly(C1)) + roots(Poly(C2))) - b/3/a * ones(3,1);
            γ = min(R(R>0));

        end

        γ = max(0,min(γ,1));
        ϵ += γ * B + γ^2 * A;
        err_j[t+1,1] = abs(tr(cP'*(ϵ*Q' + λ*P)) + tr(cQ'*(P'*ϵ + λ*Q)) + tr(cS'*R'*ϵ) + μ * (norm(vec(cS),1) - norm(vec(S),1)));

        #variable update
        P = P + γ * cP; 
        Q = Q + γ * cQ; 
        S = S + γ * cS; 

        #time_j(s,t+1) = toc + time_j(s,t);

        val_j[t+1,1]  = 0.5 * norm(Y - P * Q - R * S)^2 + 0.5 * λ * (norm(P)^2 + norm(Q)^2) + μ * norm(vec(S),1);
        println("Stela iteration ", t, ", time ", time_j[t+1,1], ", value ", val_j[t+1,1], ", stepsize ",γ, " Convergence ", err_j[t+1,1]);
        if err_j[t+1,1] <= 1e-2
            println("Optimum reached")
            println("check the optimality of solution: ", norm(Y - X - R * S), " Lambda: ", λ);
            return P, Q, S, val_j, err_j
        end
    end
    X = P * Q;
    println("check the optimality of solution: ", norm(Y - X - R * S), " Lambda: ", λ);
    #plot(val_j)
    #graphplot(R)
    #return X0, P0, Q0, P, Q, S0, S, val_j
    #println("optimality reached ", norm(Y - P * Q - R * S), " ", λ)
    return P, Q, S, val_j, err_j
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

function find_branches!(cin::Array{Int64,2},expo::Int64,debug=false,rc=true)
    if expo == 0 return 1; end
    cc = copy(cin);
    if expo > 1
        cc[diagind(cc)] .= 0
    end
    #m = dp' * -dm                       # Hashimoto / non-backtracking matrix
    k = size(cc)                         # dimension of edges
    σ = findall(x -> x == 1, vec(cc^expo))    # no. of succeding edges in Hashimoto
    I = Matrix(Diagonal(ones(k[1])))    # Identity Matrix
    out = zeros(k[1], sum(vec(cc^expo)[σ]))
    rows = σ .% k[1]
    cols = round.(Int, σ / k[1], RoundUp)
    rows[rows.==0] .= k[2]
    if rc println("Cols: ",cols); println("Rows :", rows) end
    flag = 0
    counter = 0
    for k = 1:length(rows)
        #if debug println("k = ", k, " flag = ", flag, " cols = ", cols[k]) end
        if cols[k] == flag
            if debug
                println("k = ",
                    k,
                    " flag = ",
                    flag,
                    " cols = ",
                    cols[k],
                    " Selbe Spalte ... Setze Zeile ",
                    rows[k],
                    " Spalte ",
                    cols[k] + 1 + counter,
                    " cols k: ",
                    cols[k],
                    " counter : ",
                    counter
                )
            end
            #out[rows[k], cols[k]+1+counter] = 1
            out[rows[k], k] = 1
            counter += 1
        else
            if debug
                println("k = ",
                    k,
                    " flag = ",
                    flag,
                    " cols = ",
                    cols[k],
                    " Unterschiedlich  Setze Zeile ",
                    rows[k],
                    " Spalte ",
                    cols[k] + counter,
                    " cols k: ",
                    cols[k],
                    " counter : ",
                    counter
                )
            end
            #out[rows[k], cols[k]+counter] = 1
            out[rows[k], k] = 1
        end
        #if debug print("cols k: ", cols[k], " counter : ", counter) end
        flag = cols[k]
    end
    if expo == 1
        count = 1
        for t in cols
            out[:, count] += I[:, t]
            count += 1
        end
    else
        o1,r1,c1 = find_branches!(cc,expo-1, false, false);
        for jj in 1:length(cols)
            sC, sR = cols[jj], rows[jj];
            preselect = findall(x-> x==1, cc'[:,sR]);
            #println("Gefundene Zeile ", sR, " Spalte ", sC, " preselect ", preselect);
            hits = reduce(vcat,[findall(x->x==preselect[k],r1) for k in 1:length(preselect)]);
            for k in hits
                if c1[k] == sC
                    out[:,jj] += o1[:,k];
                else
                    continue
                end
            end
        end
    end
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
    R = zeros(m,Int(R_width));
    R[1:m,1:m] = Matrix(Diagonal(ones(m)));
    memory = Int.(filter(!iszero,memory))
    start = Int(m);
    for j in 1:length(memory)
        temp,_,_ = find_branches!(c,j,false,false);
        R[:,start+1:start+memory[j]] .= temp;
        start += memory[j]
    end
    redundant = sum((Dp*R) .> 1, dims=1) + sum((-Dm*R) .> 1,dims=1)
    R_nonred = zeros(m, sum(redundant.==0))
    counter = 1;
    for k in 1:R_width
        if redundant[Int(k)] == 0
            R_nonred[:,counter] = R[:,Int(k)];
            counter += 1;
        end
    end
    return R_nonred;
end


function mse(x,y,val=false,normal=false)
    n = size(x)[1]
    if normal
        x = x ./ maximum(x)
        y = y ./ maximum(y)
    end
    if val
        return norm2(x-y) / n
    end
    return (x .- y) .^2
end

function mae(x,y,val=false,normal=false)
    n = size(x)[1]
    if normal
        x = x ./ maximum(x)
        y = y ./ maximum(y)
    end
    if val
        return norm(x-y,1) / n
    end
    return norm.(x-y,1)
end

function parseSignals(pyFrame::String)
    pyData = np.load(pyFrame,allow_pickle=true);
    sigs = convert(Dict{String,Any},get(pyData,"arr_0")[]);
    A3_ = get(sigs["A003"],"signals");
    A7_ = get(sigs["A007"],"signals");
    zuw = hcat(A7_,A3_);
    dayframe = zeros(1440,30);
    y_mask = zeros(46);
    y_mask[1:4] .= 1;
    y_mask[16:27] .= 1;
    y_mask[46] = 1;
    senID = [26;27;28;29;3;4;5;8;9;10;14;15;16;20;21;22;2];
    dayframe[:,senID] .= zuw[:,Bool.(y_mask)];
    return dayframe;
end

function parseDay(day::String)
    tDay = parseSignals(day)
    tDay[isnan.(tDay)] .= 0;
    optim_day(tDay);
    return copy(x_val), copy(s_val), copy(p_val)
end

function st(k::Int,x,s,p,sig)
    p1 = plot(x[:,k],marker=:x,title="Edge")
    p2 = plot!(s[:,k],marker=:s)
    p3 = plot!(p[:,k],marker=:p)
    p4 = plot(Dp*x[:,k],marker=:x,title="Dp")
    p5 = plot!(Dp*s[:,k],marker=:s)
    p6 = plot!(Dp*(x[:,k]-s[:,k]),marker=:p)#   -Dp*p[:,k],marker=:p)
    p10 = plot!(sig[k,:],marker=:circ,markersize=2)
    p7 = plot(Dm*x[:,k],marker=:x,title="Dm")
    p8 = plot!(Dm*s[:,k],marker=:s)
    p9 = plot!(-Dm*p[:,k],marker=:p)
    p11 = plot!(sig[k,:],marker=:circ,markersize=2)
    plot(p3,p10,p11,layout=(1,3),legend=:false)
end


function soft_threshholding(y,λ,δ=1)
    return δ*δ * (y/δ + λ * max(abs(y/δ) - λ - 1, 0)) / (1 + λ)
end

function descent_dir(𝔹x, x, γ)
    return (𝔹x - x)*γ
end

function stepsize(ϵ, ∇Ax, 𝔹x, x, μ_v)
    return max.(min.(- (ϵ' * ∇Ax + (abs.(𝔹x) - abs.(x))' * µ_v) / ∇Ax' * ∇Ax, 1), 0)
end

function stepnum(ϵ, ∇Ax, 𝔹x, x, μ_vec)
    return (-(ϵ' * ∇Ax + (norm(𝔹x,1) - norm(x,1)) * µ_vec[1]))
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

function kernel_stela!(x, ∇f, AtA_diag, µ_vec_norm, K, A, ϵ, δ, µ_vec, objval, error, Maxiter)
    for t = 1:Maxiter
        𝔹x = soft_threshholding.(ϵ,μ_vec[1],δ);#(x'- ∇f ./ AtA_diag)' , µ_vec_norm', K)
        @inbounds δx = descent_dir(𝔹x, x, 1)#𝔹x - x
        @inbounds ∇Ax = A * δx
        #bla = descent_dir(𝔹x, x, 1)

        @inbounds step_num = stepnum(ϵ, ∇Ax, 𝔹x, x, μ_vec)
        @inbounds step_denom = stepdom(∇Ax) #∇Ax' * ∇Ax
        step_size =  max.(min.(step_num/ step_denom, 1), 0)

        @inbounds x[:] +=  δx[:] * step_size 
        @inbounds ϵ[:] += ∇Ax * step_size

        @inbounds ∇f[:] = ϵ' * A
        @inbounds f = 0.5*sum(huber_loss1.(ϵ))#0.5 * ϵ' * ϵ
        @inbounds g = µ * norm(x,1)
        @inbounds objval[t+1] = f[1] + g
        #setindex!(objval[:],f[1]+g,t+1)
        IterationOut = "{1:9}|{2:10}|{3:15}|{4:15}"
        @inbounds error[t+1] = norm(abs.(∇f' - min.( max.( (∇f - x')', -µ*ones(K) ), µ*ones(K) ) ), Inf);
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


function stela_lasso(A::Array{Float64,2}, y::Vector{Float64}, µ::Float64, δ::Float64, Maxiter::Int64)
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
    @inbounds ϵ = huber_loss1.((A*x - y),δ)
    @inbounds ∇f = ϵ' * A
    @inbounds f = 0.5 * sum(huber_loss1.(ϵ,δ))#ϵ' * ϵ
    @inbounds g = µ * norm(x,1)
    setindex!(objval[:],f[1],1)
    @inbounds error[1] = norm(abs.(∇f' - min.( max.((∇f - x')', -µ*ones(K) ), µ*ones(K))),Inf)

    IterationOut = "{1:9}|{2:10}|{3:15}|{4:15}"
    printfmtln(IterationOut,"Iteration", "stepsize", "objval", "error")
    printfmtln(IterationOut, 1, "N/A", format(objval[1], width=7), format(error[1], width=7))
    kernel_stela!(x, ∇f, AtA_diag, µ_vec_norm, K, A, ϵ, δ, µ_vec, objval, error, Maxiter)
    return objval, x, error
end

function huber_loss1(x,M=1)
    if abs(x) <= M
        return 0.5*x*x
    else
        return M*(abs(x) - 0.5* M)
    end
end


function threshold(x,level=1,norm=false)
    if norm
        return x ./ maximum(x) + max.(-x ./ maximum(x), - level*ones(size(x)))#,zeros(size(x)))
    else
        return x - min.(x, level*ones(size(x)))
    end
end

function soft_thresholding_recons(q,t,k)
    return max.(q-t,zeros(k)) - max.(-q - t, zeros(k))
end

function stepnumerator(ϵ, ∇Dx_x, ∇Ds_s, Bx, Bs, xx, ss, μx, μs)
    return ϵ' * (∇Dx_x + ∇Ds_s) + μx * (norm(Bx,1) - norm(xx,1)) + μs * (norm(Bs,1) - norm(ss, 1))
    
end

function stepdenominator(∇Dxxss)
    return @inbounds ∇Dxxss'*∇Dxxss
end

function stepsize(num,den)
    return max(min(num/den, 1), 0)
end

function stela_rec!(xx, ss, ϵ, ϵ_s, ϵ_x, Dx, Ds, Dx_diag, Ds_diag, ∇f, ∇fs, ∇fx, μx, μs, objval, err, Maxiter, μ_vx_n, μ_vs_n, K)
    for t = 1:Maxiter 
        Bx = soft_thresholding_recons(xx - ∇fs./Dx_diag, μ_vx_n,K)
        Bs = soft_thresholding_recons(ss - ∇fx./Ds_diag, μ_vs_n,K)
        
        @inbounds δx = descent_dir(Bx,xx,1)
        @inbounds δs = descent_dir(Bs,ss,1)
        @inbounds ∇Dx_x = Dx * δx
        @inbounds ∇Ds_s = Ds * δs
        
        @inbounds step_num = -stepnumerator(ϵ, ∇Dx_x, ∇Ds_s, Bx, Bs, xx, ss, μx, μs)
        @inbounds step_dom = stepdenominator(∇Dx_x + ∇Ds_s)
        @inbounds γ = stepsize(step_num, step_dom)
        
        @inbounds xx[:] += γ*δx;
        @inbounds ss[:] += γ*δs;
        
        @inbounds ϵ_x[:] += γ * ∇Dx_x
        @inbounds ϵ_s[:] += γ * ∇Ds_s
        @inbounds ϵ[:] += γ * (∇Dx_x + ∇Ds_s)
        
        @inbounds ∇f = (Dx+Ds)' * ϵ;
        @inbounds ∇fx = Ds' * ϵ_x  
        @inbounds ∇fs = Dx' * ϵ_s
        
        #println(size(∇fx), size((∇f)), typeof(ϵ))#, typeof(δs))
        @inbounds f = 0.5* ϵ' * ϵ
        @inbounds g = μx * norm(xx,1) + μs* norm(ss,1)
        objval[t+1] = f + g;
        #setindex!(objval[:], f[1]+g, t)
        
        @inbounds err[t+1] =  abs.(((δx+δs)' * ∇f) - μx*norm(Bx,1) +μs *norm(Bs,1) - g) ;
        logOpt = "{1:9}|{2:10}|{3:15}|{4:15}";
        printfmtln(logOpt, t+1, format(γ,width=6), format(objval[t+1],width=7), format(err[t+1],width=7));
        if err[t+1] <= 1e-6
            println("Succesfull")
            #return 
            #return objval, err, γ, x, s
        elseif t == Maxiter
            println("Optimization needs more iterations")
            #return 
        end
    end
end

function stela_reconstruct(Dx::SparseMatrixCSC, Ds::SparseMatrixCSC, y::Vector{Float64},μx::Float64,μs::Float64, Maxiter::Int)
    if μ <= 0
        println("μ must be positive")
        return 
    elseif size(Dx)[1] != size(y)[1] || size(Ds)[1] != size(y)[1]
        println("Number of rows in A must be equal to dimension of y")
        return 
    end
    K = size(Dx)[2]
    @inbounds Dx_diag = diag(Dx'*Dx);
    @inbounds Ds_diag = diag(Ds'*Ds);
    @inbounds μ_vx = μx*ones(K);
    @inbounds μ_vs = μs*ones(K);
    @inbounds μ_vx_n = μ_vx ./ Dx_diag;
    @inbounds μ_vs_n = μ_vs ./ Ds_diag;
    xx = zeros(K);
    ss = zeros(K);
    objval = zeros(Maxiter+1)# Array{Float64}(zeros(Maxiter+1));
    err = zeros(Maxiter+1);
    @inbounds ϵ = Dx*xx + Ds*ss- y;
    @inbounds ϵ_s = Ds*ss - y;
    @inbounds ϵ_x = Dx*xx - y; 
    @inbounds ∇f = (Dx+Ds)'* ϵ;
    @inbounds ∇fs = Dx'* ϵ_s;
    @inbounds ∇fx = Ds'* ϵ_x;
    @inbounds f = 0.5 * ϵ'* ϵ;
    @inbounds g = μx * norm(xx,1) + μs* norm(ss,1)
    objval[1] = f;
    #setindex!(objval[:],f[1],1);
    @inbounds err[1] = norm(∇f,2) + g;
    logOpt = "{1:9}|{2:10}|{3:15}|{4:15}";
    printfmtln(logOpt, "Iteration", "stepsize", "objval", "error");
    printfmtln(logOpt, "1", "N/A", format(objval[1],width=7), "0");
    stela_rec!(xx, ss, ϵ, ϵ_s, ϵ_x, Dx, Ds, Dx_diag, Ds_diag, ∇f, ∇fs, ∇fx, μx, μs, objval, err, Maxiter, μ_vx_n, μ_vs_n, K);
    return objval, xx, ss, err
end
