using BEAST
using ClusterTrees
using CompScienceMeshes
using FastBEAST
using LinearAlgebra
using Test

c = 3e8
f = 1e8
λ = c/f
k = 2*π/λ

a = 1.0
Γ = CompScienceMeshes.meshsphere(1.0, 0.08)

SL = Maxwell3D.singlelayer(wavenumber=k)
X = raviartthomas(Γ)
@show length(X.pos)

T = assemble(SL, X, X)

# blockassembler
@views blkasm = BEAST.blockassembler(SL, X, X)
    
@views function assembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    blkasm(tdata,sdata,store)
end

@time tree = create_CT_tree(X.pos, nchildren=2, nmin=200, maxlevel=10);
block_tree = ClusterTrees.BlockTrees.BlockTree(tree, tree)

@time h2mat = FastBEAST.H2Matrix(
    assembler, block_tree, ComplexF64, multithreading=true
);

x = rand(ComplexF64, length(X.pos))

@test norm(h2mat*x - T*x)/norm(T*x) ≈ 0.0 atol=1e-4
@test norm(adjoint(h2mat) * x - adjoint(T) * x)/norm(adjoint(T)*x) ≈ 0.0 atol=1e-4
@test norm(transpose(h2mat) * x - transpose(T) * x)/norm(transpose(T)*x) ≈ 0.0 atol=1e-4
@test estimate_reldifference(h2mat, T; tol=1e-4) ≈ 0.0 atol=1e-4