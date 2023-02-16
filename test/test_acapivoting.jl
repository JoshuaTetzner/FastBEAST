using FastBEAST
using LinearAlgebra
using Base.Threads

function greensfunction(
    src::Matrix{F},
    trg::Matrix{F}
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
s = rand(nsrc, 3)
t = rand(ntrg, 3)

s[:, 3] += 5 .* ones(nsrc)
src = s
trg = t

A = greensfunction(src, trg)

M2 = FastBEAST.FillDistance(zeros(Bool, nsrc), zeros(Bool, ntrg), src, trg)

function fct(C, x, y)
    for i in eachindex(x)
        for j in eachindex(y)
            C[i,j] = A[x[i],y[j]]
        end
    end
end

rowindices = Array(1:nsrc)
colindices = Array(1:ntrg)

lm = LazyMatrix(fct, rowindices, colindices, Float64);

##
U, V = aca(lm, tol=1e-15)
size(U)
norm(A-U*V)/norm(A)

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
src1 = CompScienceMeshes.meshrectangle(1.0, 1.0, 0.3, 3)
src2 = CompScienceMeshes.rotate(src1, SVector(-pi/2, 0, 0))
src2 = CompScienceMeshes.translate(src2, SVector(0, 0,-1))

Γsrc = CompScienceMeshes.weld(src1, src2)
Γtrg = translate(Γsrc, SVector(2.0, 0.0, 0.0))

MS = Maxwell3D.singlelayer(wavenumber=k)
MD = Maxwell3D.doublelayer(wavenumber=k)

Xsrc = raviartthomas(Γsrc)
Ysrc = buffachristiansen(Γtrg)
Ytrg = buffachristiansen(Γtrg)
Xtrg = raviartthomas(Γtrg)


println("Number of RWG functions: ", numfunctions(Xsrc))
T = hassemble(
    MS,
    Ytrg,
    Xsrc,
    pivoting=:filldistance,
    treeoptions=KMeansTreeOptions(nmin=30),
    threading=:single,
    quadstrat=BEAST.DoubleNumQStrat(1, 1),
    verbose=true,
    svdrecompress=false
)

##
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

test_tree = create_tree(Ytrg.pos, FastBEAST.BoxTreeOptions(nmin=1000))
trial_tree = create_tree(Xsrc.pos, FastBEAST.BoxTreeOptions(nmin=1000))

#piv(roworcolumn, acausedindices, totalindices) = FastBEAST.minimalfilldistance(Ytrg, roworcolumn, acausedindices, totalindices)#
piv(roworcolumn, acausedindices, totalindices) = FastBEAST.smartmaxlocal(roworcolumn, acausedindices, totalindices)
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


##
norm(block.M.U*block.M.V - assemble(MD, Ytrg, Xsrc))
##

# Example Corner
using Test
using FastBEAST
using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using Plotly

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
trg = translate(src, SVector(-2.5, -1.0, 0.0))
trg = rotate(trg, SVector(0.0,0.0,pi))
##
#plot([CompScienceMeshes.wireframe(src),CompScienceMeshes.wireframe(trg)])
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

#piv(roworcolumn, acausedindices, totalindices) = FastBEAST.minimalfilldistance(Ytrg, roworcolumn, acausedindices, totalindices)#
piv(roworcolumn, acausedindices, totalindices) = FastBEAST.smartmaxlocal(roworcolumn, acausedindices, totalindices)

am = allocate_aca_memory(scalartype(MS), length(XS.fns), length(XT.fns), maxrank=201)
block = FastBEAST.getcompressedmatrix(
    farassembler,
    test_tree,
    trial_tree,
    Int64,
    scalartype(MS),
    am,
    piv,
    compressor=:aca,
    tol=10^-6,
    maxrank=200,
    svdrecompress=false
)

norm(block.M.U*block.M.V - assemble(MS, XT, XS))

##
fn = joinpath(@__DIR__, "mesh/ihplate.geo")
Γ = CompScienceMeshes.meshgeo(fn; dim=2)

##
fn = joinpath(@__DIR__, "mesh/ihplate.msh")
src = CompScienceMeshes.read_gmsh_mesh(fn)
trg = translate(src, SVector(-3.0, -1.0, 0.0))
trg = rotate(trg, SVector(0.0,0.0,pi))

plot([wireframe(src), wireframe(trg)])
