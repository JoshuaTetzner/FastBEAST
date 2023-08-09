using Test
using FastBEAST
using StaticArrays
using LinearAlgebra

# Do a 3D test with Laplace kernel

function OneoverRkernel(testpoint::SVector{3,T}, sourcepoint::SVector{3,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e-4)
        return T(0.0)
    else
        return T(1.0) / (norm(testpoint - sourcepoint))
    end
end

function assembler(kernel, testpoints, sourcepoints)
    kernelmatrix = zeros(promote_type(eltype(testpoints[1]),eltype(sourcepoints[1])), 
                length(testpoints), length(sourcepoints))

    for j = 1:length(sourcepoints)
        for i = 1:length(testpoints)
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
    return kernelmatrix
end


function assembler(kernel, matrix, testpoints, sourcepoints)
    for j = 1:length(sourcepoints)
        for i = 1:length(testpoints)
            matrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end

##

N =  2000

spoints = [@SVector rand(3) for i = 1:N]
tpoints = spoints

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, tpoints[tdata], spoints[sdata])

stree = create_tree(spoints, KMeansTreeOptions(maxlevel=5))
ttree = create_tree(tpoints, KMeansTreeOptions(maxlevel=5))
kmat = assembler(OneoverRkernel, tpoints, spoints)

@time fi, ci = h2matrix(
    OneoverRkernelassembler,
    ttree,
    stree,
    Int64,
    Float64,
    compressor=FastBEAST.ACAOptions(tol=1e-4)
)

I = 1

A = zeros(Float64, N, N)
for f in fi
    A[f.τ, f.σ] .= f.M
end

for c in ci
    A[c.τ, c.σ] .= c.M.U*c.M.V
end

fullmat = assembler(OneoverRkernel, tpoints, spoints)

norm(fullmat .- A)/norm(fullmat)

ci[2].τ
ci[2].σ

ci[2].M.τ
ci[2].M.σ
ci[2].M.U
ci[2].M.V

##
N =  100

t1 = [@SVector rand(3) for i = 1:N]
t2 = [@SVector rand(3) for i = 1:N] + [1.0*SVector(0, 1, 0) for i = 1:N]
s1 = [@SVector rand(3) for i = 1:N] + [1.0*SVector(4, 0, 0) for i = 1:N]
s2 = [@SVector rand(3) for i = 1:N] + [1.0*SVector(5, 0, 0) for i = 1:N]

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, tpoints[tdata], spoints[sdata])

Z1 = assembler(OneoverRkernel, t1, s1)
Z2 = assembler(OneoverRkernel, t1, s2)

Z3 = assembler(OneoverRkernel, t2, s1)
Z4 = assembler(OneoverRkernel, t2, s2)

##
vcat(Z1, Z2)

##
@views function fct_ts1(B, x, y)
    B[:,:] = vcat(Z1, Z3)[x, y]
end

@views function fct_ts2(B, x, y)
    B[:,:] = vcat(Z2, Z4)[x, y]
end

@views function fct_t1s(B, x, y)
    B[:,:] = hcat(Z1, Z2)[x, y]
end

@views function fct_t2s(B, x, y)
    B[:,:] = hcat(Z3, Z4)[x, y]
end

vcat(Z1, Z3)
Z1

lm_ts1 = LazyMatrix(fct_ts1, Vector(1:2N), Vector(1:N), Float64)
lm_ts2 = LazyMatrix(fct_ts2, Vector(1:2N), Vector(1:N), Float64)
lm_t1s = LazyMatrix(fct_t1s, Vector(1:N), Vector(1:2N), Float64)
lm_t2s = LazyMatrix(fct_t2s, Vector(1:N), Vector(1:2N), Float64)


retU_ts1, retV_ts1, U_ts1, V_ts1, r_ts1, c_ts1 = nca(lm_ts1, tol=1e-4, svdrecompress=false)
retU_ts2, retV_ts2, U_ts2, V_ts2, r_ts2, c_ts2 = nca(lm_ts2, tol=1e-4, svdrecompress=false)
retU_t1s, retV_t1s, U_t1s, V_t1s, r_t1s, c_t1s = nca(lm_t1s, tol=1e-4, svdrecompress=false)
retU_t2s, retV_t2s, U_t2s, V_t2s, r_t2s, c_t2s = nca(lm_t2s, tol=1e-4, svdrecompress=false)


Ut1 = U_t1s * (U_t1s[r_t1s,:])^-1
Ut2 = U_t2s * (U_t2s[r_t2s,:])^-1 
Vs1 = (V_ts1[:, c_ts1])^-1 * V_ts1  
Vs2 = (V_ts2[:, c_ts2])^-1 * V_ts2  

Ut1
Vs1
Vs2


norm(Ut1 * Z1[r_t1s, c_ts1]*Vs1 - Z1)/norm(Z1)