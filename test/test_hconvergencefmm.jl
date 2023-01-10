using Test
using StaticArrays
using CompScienceMeshes
using BEAST
using LinearAlgebra
using FastBEAST

# Following the example given by Jackson - Classical Electrodynamics Third Edition Chap. 2.2
r = 10.0
λ = 20 * r
k = 0.0#2 * π / λ

sphere = meshsphere(r, 0.1*r)



# Position of point charge
q = 100
ϵ = 1.0
y = [15.0, 0.0, 0.0]

function Φ_inc(x)
    return q / (4 * π * ϵ) * (
        exp(-im * k * norm(x - y)) / (norm(x - y))
    )
end

# Surfacecharge
σ(a, γ) = - q / (4*pi*norm(a)^2) * norm(a) / norm(y) * 
    (1 - norm(a)^2 / norm(y)^2) / (1 + norm(a)^2 / norm(y)^2 - 2 * norm(a) / norm(y) * cos(γ))^(3/2)

##
X0 = lagrangecxd0(sphere)
surfacecharges = [σ(c, acos(dot(c,y) / (norm(c)*norm(y)))) for c in X0.pos]

gD0 = assemble(ScalarTrace(Φ_inc), X0)
S = Helmholtz3D.singlelayer(; gamma=im*k)

@time fmat = fmmassemble(
    S,
    X0,
    X0,
    nmin=50,
    threading=:multi,
    npoints=5,
    fmmoptions=HelmholtzFMMOptions(imag(S.gamma))
)

fmat*surfacecharges
#norm((A*surfacecharges + gD0))/norm((gD0))
