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

src = CompScienceMeshes.meshrectangle(1.0, 1.0, 0.4, 3)
src2 = CompScienceMeshes.rotate(src, SVector(0, pi/2, 0))
src2 = CompScienceMeshes.rotate(src2, SVector(0, 0, pi))
src2 = CompScienceMeshes.translate(src2, SVector(0, 1, 0))

trg = CompScienceMeshes.translate(src, SVector(2.0, 0, 0))
Γsrc = CompScienceMeshes.weld(src, src2)
Γtrg = trg

MS = Maxwell3D.singlelayer(wavenumber=k)
MD = Maxwell3D.doublelayer(wavenumber=k)

Xsrc = raviartthomas(Γtrg)
Ysrc = buffachristiansen(Γsrc)
Ytrg = buffachristiansen(Γsrc)
Xtrg = raviartthomas(Γtrg)

A = assemble(MD, Ytrg, Xsrc)

using SparseArrays
sparse(round.(A, digits=10))


##

using Plotly
plot([wireframe(Γsrc), wireframe(Γtrg)])
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

lm = LazyMatrix(farassembler, FastBEAST.indices(test_tree), FastBEAST.indices(trial_tree), scalartype(MD));
fd = FastBEAST.FillDistance(Ytrg.pos, zeros(Float64, length(Ytrg.fns)), 10, false)
slm = FastBEAST.LocalMax(1)

@time U, V = FastBEAST.modifiedaca(lm, fd, maxrank=100, tol=1e-5);

A = assemble(MD, Ytrg, Xsrc)

using SparseArrays
sparse(round.(A, digits=10))
#norm(U*V-A)/norm(A)

##
function fullmatrixview(hmat)
    M = zeros(eltype(hmat), size(hmat)[1], size(hmat)[2])

    for fb in hmat.fullrankblocks
        M[fb.τ, fb.σ] = fb.M
    end

    for lb in hmat.lowrankblocks
        M[lb.τ, lb.σ] = lb.M.U*lb.M.V
    end
    return M
end

function errblocks(hmat, A)

    err = []
    for lb in hmat.lowrankblocks
        push!(err, norm(A[lb.τ, lb.σ] - lb.M.U*lb.M.V)/norm(A[lb.τ, lb.σ]))
    end

    return err
end

##

using CompScienceMeshes
using BEAST
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using FastBEAST
using Test

function farquaddata(op::BEAST.MaxwellOperator3D,
    test_local_space::BEAST.RefSpace, trial_local_space::BEAST.RefSpace,
    test_charts, trial_charts)

    a, b = 0.0, 1.0
    # CommonVertex, CommonEdge, CommonFace rules

    tqd = quadpoints(test_local_space, test_charts, (1,6))
    bqd = quadpoints(trial_local_space, trial_charts, (1,7))
    leg = (BEAST._legendre(3,a,b), BEAST._legendre(4,a,b), BEAST._legendre(5,a,b),)

    # High accuracy rules (use them e.g. in LF MFIE scenarios)
    # tqd = quadpoints(test_local_space, test_charts, (8,8))
    # bqd = quadpoints(trial_local_space, trial_charts, (8,9))
    # leg = (_legendre(8,a,b), _legendre(10,a,b), _legendre(5,a,b),)


    return (tpoints=tqd, bpoints=bqd, gausslegendre=leg)
end


c = 3e8
μ = 4*π*1e-7
ε = 1/(μ*c^2)
f = 1e8
λ = c/f
k = 2*π/λ
ω = k*c
η = sqrt(μ/ε)

a = 1.0
Γ_orig = CompScienceMeshes.meshcuboid(a,a,a,0.2)
Γ = translate(Γ_orig,SVector(-a/2,-a/2,-a/2))

Φ, Θ = [0.0], range(0,stop=π,length=100)
pts = [point(cos(ϕ)*sin(θ), sin(ϕ)*sin(θ), cos(θ)) for ϕ in Φ for θ in Θ]

# This is an electric dipole
# The pre-factor (1/ε) is used to resemble 
# (9.18) in Jackson's Classical Electrodynamics
E = (1/ε) * dipolemw3d(location=SVector(0.4,0.2,0), 
                    orientation=1e-9.*SVector(0.5,0.5,0), 
                    wavenumber=k)

n = BEAST.NormalVector()

𝒆 = (n × E) × n
H = (-1/(im*μ*ω))*curl(E)
𝒉 = (n × H) × n

𝓣 = Maxwell3D.singlelayer(wavenumber=k)
𝓝 = BEAST.NCross()
MD = Maxwell3D.doublelayer(wavenumber=k)

X = raviartthomas(Γ)
Y = buffachristiansen(Γ)

println("Number of RWG functions: ", numfunctions(X))

K_bc = hassemble(
    𝓚,
    Y,
    X,
    pivoting=:filldistance,
    treeoptions=BoxTreeOptions(nmin=50),
    threading=:multi,
    tol=1e-8,
    maxrank=500,
    verbose=true,
    svdrecompress=false
)

##
length(Y.fns)
A = assemble(𝓚, Y, X)
M = fullmatrixview(K_bc)
err = errblocks(K_bc, A)


norm(A .- M)/norm(A)
##
err[2503]
ind = 2503#argmax(err)
rows = K_bc.lowrankblocks[ind].τ
columns = K_bc.lowrankblocks[ind].σ

rp = Y.pos[rows]
cp = X.pos[columns]

rpm = zeros(Float64, length(rp), 3)
cpm = zeros(Float64, length(cp), 3)
for r in eachindex(rp)
    rpm[r, :] = rp[r]
end
for c in eachindex(cp)
    cpm[c, :] = cp[c]
end
##
using SparseArrays
K_bc.lowrankblocks[ind].M.U
K_bc.lowrankblocks[ind].M.V
(round.(A[rows, columns], digits=10))
A[rows, columns]-K_bc.lowrankblocks[ind].M.U * K_bc.lowrankblocks[ind].M.V
##
using Plots
plotlyjs()
Plots.scatter(rpm[:, 1], rpm[:,2], rpm[:,3])
Plots.scatter!(cpm[:, 1], cpm[:,2], cpm[:,3])

##
using Plotly

plot(wireframe(Γ))
##
G_nxbc_rt = Matrix(assemble(𝓝,Y,X))
h_bc = η*Vector(assemble(𝒉,Y))
K_bc_full = assemble(𝓚,Y,X)
M_bc = -0.5*G_nxbc_rt + K_bc

##
h_bc

println("Enter iterative solver")
@time j_BCMFIE, ch = IterativeSolvers.gmres(M_bc, h_bc, log=true, reltol=1e-4, maxiter=500)
##

nf_E_BCMFIE = potential(MWSingleLayerField3D(wavenumber=k), pts, j_BCMFIE, X)
nf_H_BCMFIE = potential(BEAST.MWDoubleLayerField3D(wavenumber=k), pts, j_BCMFIE, X) ./ η
ff_E_BCMFIE = potential(MWFarField3D(wavenumber=k), pts, j_BCMFIE, X)

@test norm(nf_E_BCMFIE - E.(pts))/norm(E.(pts)) ≈ 0 atol=0.01
@test norm(nf_H_BCMFIE - H.(pts))/norm(H.(pts)) ≈ 0 atol=0.01
@test norm(ff_E_BCMFIE - E.(pts, isfarfield=true))/norm(E.(pts, isfarfield=true)) ≈ 0 atol=0.01

