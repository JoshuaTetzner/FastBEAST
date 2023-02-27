using FastBEAST
using LinearAlgebra
using Base.Threads
using StaticArrays
using CompScienceMeshes

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

nsrc = 100
ntrg = 100
s = [@SVector rand(3) for i = 1:nsrc]
t = [@SVector rand(3) for i = 1:ntrg] +  1.5 .* [@SVector ones(3) for i = 1:ntrg]

src = s
trg = t

A=greensfunction(src, trg)

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
fd = FastBEAST.FillDistance(s, zeros(Float64, nsrc), 10, false)
slm = FastBEAST.LocalMax(1)
##
am = allocate_aca_memory(Float64, ntrg, nsrc, maxrank=200)
#@time U1, V1 = FastBEAST.modifiedaca2(lm, am, fd, rank=50, tol=1e-5);
@time U1, V1 = FastBEAST.modifiedaca(lm, am, fd, tol=1e-14);
@show norm(U1*V1-A)/norm(A)
##
am = allocate_aca_memory(Float64, ntrg, nsrc, maxrank=100)
@time U, V = FastBEAST.modifiedaca(lm, fd, maxrank=200, tol=1e-5);
@show norm(U*V-A)/norm(A)
size(U)
##
U1 - U
V1 - V
A
##
## inhomo Plates
using FastBEAST
using BEAST
using LinearAlgebra
using Base.Threads
using StaticArrays
using CompScienceMeshes


c = 3e8
f = 1e8
λ = c/f
k = 2*π/λ

fn = joinpath(@__DIR__, "mesh/ihplate.msh")
src = CompScienceMeshes.read_gmsh_mesh(fn)
trg = CompScienceMeshes.translate(src, SVector(-2.5, -1.0, 0.0))
trg = CompScienceMeshes.rotate(trg, SVector(0.0,0.0,pi))

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

A = assemble(MS, XT, XS)
err = []
XT.pos
am = allocate_aca_memory(scalartype(MS), length(XT.fns), length(XS.fns), maxrank=300)
lm = LazyMatrix(farassembler, FastBEAST.indices(test_tree), FastBEAST.indices(trial_tree), scalartype(MS))
fd = FastBEAST.FillDistance(XT.pos, zeros(Float64, length(XT.pos)), 28, false)
slm = FastBEAST.LocalMax(28)

##
#am = allocate_aca_memory(scalartype(MS), length(XT.pos), length(XS.pos), maxrank=200)
#@time U, V = FastBEAST.modifiedaca2(lm, am, fd, rank=70, tol=1e-12);
#@show norm(U*V-A)/norm(A)
am = allocate_aca_memory(scalartype(MS), length(XT.pos), length(XS.pos), maxrank=600)
@time U, V = FastBEAST.modifiedaca(lm, fd, maxrank=600, tol=1e-10);
@show norm((U*V-A))/(norm(A))
##
using SparseArrays
X = sparse(round.(U*V-A, digits=15))
X.rowval
X[1,:]
(U*V-A)[1,:]
##
using JLD2

err = []
for i = 1:length(XS.fns)
    println(i, "/", length(XS.fns))
    lm = LazyMatrix(farassembler, FastBEAST.indices(test_tree), FastBEAST.indices(trial_tree), scalartype(MS))
    slm = FastBEAST.LocalMax(i)
    fd = FastBEAST.FillDistance(XT.pos, zeros(Float64, length(XT.pos)), i, false)
    am = allocate_aca_memory(scalartype(MS), length(XS.pos), length(XS.pos), maxrank=800)
    U, V = FastBEAST.modifiedaca(lm, slm, maxrank=800, tol=1e-11);
    push!(err, norm(U*V-A)/norm(A))
end

save("cc_slm11.jld2", "1e11", err)



##
 using Plots
err = []
for j = 1:length(XS.fns)
    println(j,"/", nsrc)
    am = allocate_aca_memory(scalartype(MS), length(XT.fns), length(XS.fns), maxrank=300)
    lm = LazyMatrix(farassembler, FastBEAST.indices(test_tree), FastBEAST.indices(trial_tree), scalartype(MS))
    slm = FastBEAST.LocalMax(j)
    U, V = FastBEAST.modifiedaca(lm, am, slm; tol=1e-4)
    push!(err, norm(U*V-A)/norm(A))
end
histogram(err, xaxis=:log, xlim=[1e-5, 1e-3])
##
err = []
for j = 1:length(XS.fns)
    println(j,"/", length(XS.fns))
    am = allocate_aca_memory(scalartype(MS), length(XT.fns), length(XS.fns), maxrank=300)
    lm = LazyMatrix(farassembler, FastBEAST.indices(test_tree), FastBEAST.indices(trial_tree), scalartype(MS))
    fd = FastBEAST.FillDistance(XS.pos, j, false)
    U, V = FastBEAST.modifiedaca(lm, am, fd; tol=1e-4)
    push!(err, norm(U*V-A)/norm(A))
end
histogram(err, xaxis=:log, xlim=[1e-5, 1e-3])
##
a = log.([167.0165485578869, 15.983984539322567, 15.62152600413606, 29.961435420614105, 5.252202607777557])
aim = log.(1e-14)
b = Array(1:length(a))

a_ = sum(a)/length(a)
b_ = sum(b)/length(a)

m = sum((a .- a_).*(b .- b_))/sum((b .- b_).^2)

(aim-a[1])/m


##
using Plots
a = [167.0165485578869, 15.983984539322567, 15.62152600413606, 29.961435420614105, 5.252202607777557, 2.2585510055471913, 6.871296996064547, 5.445733522925249, 2.5443836576669008, 0.8244022716102344, 1.0255724508534263, 0.2552629673567543, 0.6547188058002177, 0.12787812217032754, 0.2878892515908694, 0.321512494187139, 0.21036058397177204, 0.17642697991760836, 0.2986543947501773, 0.08557604233077903, 0.7737375596732374, 0.1588294638302307, 0.7465628948751695, 0.09958270153077506, 0.10513446305765811, 0.08236259064497446, 0.11692904669759716, 0.052958149109265225, 0.12032709908014963, 0.11037288286711618, 0.24536100226846225, 0.2883995738927258, 0.059026528401510556, 0.061364838182633996, 0.7481702428870094, 0.4931929397431158, 0.29311279026595993, 0.3043570403007433, 0.09710070521620254, 0.13648731232782152, 0.08946757801943026, 0.061631513082070585, 0.026905729843592206, 0.03870889207579167, 0.016690004595238887, 0.019949171047249387, 0.05095719195209779, 0.015054699285752949, 0.05192042708385444, 0.043303242354520766, 0.04745117286062586, 0.02576229087822767, 0.04264187167360401, 0.019896617047073812, 0.02643815644755846]
plot(Array(1:length(a)), a, yaxis=:log)