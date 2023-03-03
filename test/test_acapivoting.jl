using FastBEAST
using LinearAlgebra
using Base.Threads
using StaticArrays

##
# random distribution
function greensfunction(
    src::Vector{SVector{3, F}},
    trg::Vector{SVector{3, F}}
) where {F <: Real}

    G = zeros(F, size(trg)[1], size(src)[1])

    @threads for row = 1:size(trg)[1]
        @threads for col = 1:size(src)[1]
        
            if src[row, :] != trg[col, :]
                r = norm(src[row, :] - trg[col, :])
                G[row, col] = 1/(4*pi*r)
            end
        end
    end

    return G
end

nsrc = 1000
ntrg = 1000
s = [@SVector rand(3) for i = 1:nsrc]
t = [@SVector rand(3) for i = 1:ntrg] + 1.5 .* [@SVector ones(3) for i = 1:ntrg]

src = s
trg = t

A = greensfunction(src, trg)

function fct(C, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            C[i,j] = A[x[i],y[j]]
        end
    end
end

rowindices = Array(1:nsrc)
colindices = Array(1:ntrg)

piv(roworcolumn, acausedindices, totalindices) = FastBEAST.minimalfilldistance(trg, roworcolumn, acausedindices, totalindices)
lm = LazyMatrix(fct, rowindices, colindices, Float64);
U, V = aca(lm, maxrank=100, pivoting=piv, tol=1e-12)
##
# Example Corner
using Test
using FastBEAST
using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using IterativeSolvers

c = 3e8
f = 1e8
λ = c/f
k = 2*π/λ


a = 1.0
src1 = CompScienceMeshes.meshrectangle(1.0, 1.0, 0.5, 3)
src2 = CompScienceMeshes.rotate(src1, SVector(-pi/2, 0, 0))
src2 = CompScienceMeshes.translate(src2, SVector(0, 0,-1))

Γsrc = CompScienceMeshes.weld(src1, src2)
Γtrg = CompScienceMeshes.translate(Γsrc, SVector(2.0, 0.0, 0.0))

MS = Maxwell3D.singlelayer(wavenumber=k)
MD = Maxwell3D.doublelayer(wavenumber=k)

Xsrc = raviartthomas(Γsrc)
Ysrc = buffachristiansen(Γtrg)
Ytrg = buffachristiansen(Γtrg)
Xtrg = raviartthomas(Γtrg)

@views farblkasm = BEAST.blockassembler(
    MD,
    Ytrg,
    Xsrc,
    quadstrat=BEAST.defaultquadstrat(MD, Ytrg, Xsrc)
)

@views function farassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    farblkasm(tdata,sdata,store)
end

test_tree = create_tree(Ytrg.pos, FastBEAST.BoxTreeOptions(nmin=10))
trial_tree = create_tree(Xsrc.pos, FastBEAST.BoxTreeOptions(nmin=10))

piv(roworcolumn, acausedindices, totalindices) = FastBEAST.minimalfilldistance(Ytrg, roworcolumn, acausedindices, totalindices)#
#piv(roworcolumn, acausedindices, totalindices) = FastBEAST.smartmaxlocal(roworcolumn, acausedindices, totalindices)
am = allocate_aca_memory(scalartype(MS), length(Xsrc.fns), length(Xtrg.fns), maxrank=100)
block = FastBEAST.getcompressedmatrix(
    farassembler,
    test_tree,
    trial_tree,
    Int64,
    scalartype(MD),
    am,
    piv,
    compressor=:aca,
    tol=10^-5,
    maxrank=50,
    svdrecompress=false
)

@show size(block.M.U)
norm(block.M.U*block.M.V - assemble(MD, Ytrg, Xsrc))

##
# Example Plates
using Test
using FastBEAST
using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using IterativeSolvers

c = 3e8
f = 1e8
λ = c/f
k = 2*π/λ


a = 1.0

##
src = CompScienceMeshes.meshrectangle(1.0, 1.0, 0.05, 3)
trg = translate(src, SVector(2.0, 0.0, 0.0))

##
fn = joinpath(@__DIR__, "mesh/ihplate.msh")
src = CompScienceMeshes.read_gmsh_mesh(fn)
trg = CompScienceMeshes.translate(src, SVector(-2.5, -1.0, 0.0))
trg = CompScienceMeshes.rotate(trg, SVector(0.0,0.0,pi))
##
using Plotly

plot([wireframe(src), wireframe(trg)])
##

MS = Maxwell3D.singlelayer(wavenumber=k)
XS = raviartthomas(src)
XT = raviartthomas(trg)

@views farblkasm = BEAST.blockassembler(
    MS,
    XT,
    XS,
    quadstrat=BEAST.defaultquadstrat(MS, XT, XS)
)

@views function farassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    farblkasm(tdata,sdata,store)
end

test_tree = create_tree(XT.pos, FastBEAST.BoxTreeOptions(nmin=1000))
trial_tree = create_tree(XS.pos, FastBEAST.BoxTreeOptions(nmin=1000))

piv(roworcolumn, acausedindices, totalindices) = FastBEAST.minimalfilldistance(XT, roworcolumn, acausedindices, totalindices)#
#piv(roworcolumn, acausedindices, totalindices) = FastBEAST.smartmaxlocal(roworcolumn, acausedindices, totalindices)
A = assemble(MS, XT, XS)
err = []
for i = 28#1:5
    am = allocate_aca_memory(scalartype(MS), length(XS.fns), length(XT.fns), maxrank=201)
    block = FastBEAST.getcompressedmatrix(
        farassembler,
        test_tree,
        trial_tree,
        Int64,
        scalartype(MS),
        am,
        piv,
        firstindex=i;
        compressor=:aca,
        tol=10^-3,
        maxrank=200,
        svdrecompress=false
    )
    println(i, "/", size(A)[1])
    push!(err, norm(block.M.U*block.M.V - A)/norm(A))
end
println(err)
##
using JLD2
save("hc_fd.jld2", "sc_fd", err)

##
using Test
using FastBEAST
using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using IterativeSolvers

fn = joinpath(@__DIR__, "mesh/ogive_coarse.msh")
ogive = CompScienceMeshes.read_gmsh_mesh(fn)

c = 3e8
f = 1e8
λ = c/f
k = 2*π/λ

MS = Maxwell3D.singlelayer(wavenumber=k)
MD = Maxwell3D.doublelayer(wavenumber=k)
S = Helmholtz3D.singlelayer((wavenumber=k))

Xd0 = lagrangecxd0(ogive)
X = raviartthomas(ogive)
Y = buffachristiansen(ogive)
##

hmat = hassemble(
    MS,
    X,
    X,
    pivoting=:max,
    treeoptions=BoxTreeOptions(nmin=10),
    threading=:multi,
    #quadstrat=BEAST.DoubleNumQStrat(1, 1),
    maxrank=500,
    verbose=true,
    svdrecompress=false
)

@time kmat = assemble(MS, X, X);
##
println("Accuracy test: ", estimate_reldifference(hmat, kmat))
println("Compression rate: ", compressionrate(hmat)*100)
##
err = []
ind = []
for i in eachindex(hmat.lowrankblocks)
    push!(err, norm(hmat.lowrankblocks[i].M.U*hmat.lowrankblocks[i].M.V - kmat[hmat.lowrankblocks[i].τ, hmat.lowrankblocks[i].σ]))
end

err
##
src = X.pos[hmat.lowrankblocks[4].τ]
trg = X.pos[hmat.lowrankblocks[4].σ]
##
using Plots
plotlyjs()
scatter()
for i in eachindex(X.pos)
    scatter!([X.pos[i][1]], [X.pos[i][2]], [X.pos[i][3]], markercolor=:black, markeralpha=0.1, legend=false)
end
for i = eachindex(src)
    scatter!([src[i][1]], [src[i][2]], [src[i][3]], markercolor=:blue, legend=false)
end
for i = eachindex(trg)
    scatter!([trg[i][1]], [trg[i][2]], [trg[i][3]], markercolor=:orange, legend=false)
end
scatter!()

##
a = [0.7656646390359855, 0.6786852626989625, 0.2291744946588669, 0.3348627637623181, 0.12213402082413286, 0.09840940895656955, 0.02666511523991335, 0.008994969341509968, 0.007549720926749921, 0.009045868792815475, 0.003243657957926657, 0.004366597584463416, 0.0042307553333333055, 0.003689804106029018, 0.001420853550324385, 0.0004209957194041176, 0.0005764729534128723, 0.0005690637673082653, 0.0013458233868866088, 0.0002903070222305007, 0.00030832894196292044, 0.000323038121503603, 0.00011410213199095744, 5.327421300936894e-5, 4.942568328630221e-5, 4.3820624623645054e-5, 4.1237419930945294e-5, 1.2972753356660643e-5, 2.9553587889375893e-5, 4.155864683018181e-5, 3.1160110286076075e-5, 1.728287434266383e-5, 1.3460566584916453e-5, 3.206573950787234e-5, 9.34670177021742e-6, 7.599668830756626e-6, 3.7451322799228808e-6, 5.09372486657107e-6, 3.3200937177939004e-6, 3.623922653546478e-6, 9.84005524770563e-7, 1.7891048164671982e-6, 9.987830001883973e-7, 1.6422200901635476e-6, 8.44453873205838e-7, 2.751821428657973e-6, 1.1376374489146753e-6, 1.383996630954555e-6, 5.688805661253813e-7, 7.905383215461937e-7, 3.1328977742059633e-7, 2.0970020056816812e-7, 1.3818432017259862e-7, 1.403270335330593e-7, 2.645279240650214e-7, 2.510383609363621e-7, 1.481280484068458e-7]

plot(Array(1:length(a)), a, yaxis=:log)