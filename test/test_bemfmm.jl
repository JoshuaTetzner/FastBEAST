using BEAST
using FastBEAST
using StaticArrays
using Test

r = 10.0
λ = 20 * r
k = 2 * π / λ
npoints = 3

refmesh = meshsphere(r, 0.2*r)
S = Helmholtz3D.singlelayer(; gamma=im*k)

# patch basis functions
X0 = lagrangecxd0(refmesh)

@time B, fmm, BtCB, fullmat = fmmassemble(
    S,
    X0,
    X0,
    nmin=5,
    threading=:multi,
    npoints=npoints,
    fmmncrit=10,
    fmmp=10
);

charges = ComplexF64.(rand(Float64, length(X0.fns)))
GBx, ϵ = fmm(B*charges)

Ax = transpose(B) * conj.(GBx[:,1])

A = Ax - BtCB*charges + fullmat * charges
A_true = assemble(S, X0, X0) * charges

@test norm(A-A_true) / norm(A_true) ≈ 0 atol=1e-4

# pyramid basis functions
X0 = lagrangec0d1(refmesh)

@time B, fmm, BtCB, fullmat = fmmassemble(
    S,
    X0,
    X0,
    nmin=5,
    threading=:multi,
    npoints=npoints,
    fmmncrit=10,
    fmmp=10
);

charges = ComplexF64.(rand(Float64, length(X0.fns)))
GBx, ϵ = fmm(B*charges)

Ax = transpose(B) * conj.(GBx[:,1])

A = Ax - BtCB*charges + fullmat * charges
A_true = assemble(S, X0, X0) * charges

@test norm(A-A_true) / norm(A_true) ≈ 0 atol=1e-4