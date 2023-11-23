using Test
using FastBEAST
using StaticArrays
using LinearAlgebra
using ClusterTrees
 
##
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

    for j in eachindex(sourcepoints)
        for i in eachindex(testpoints)
            kernelmatrix[i, j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
    return kernelmatrix
end


function assembler(kernel, matrix, testpoints, sourcepoints)
    for j in eachindex(sourcepoints)
        for i in eachindex(testpoints)
            matrix[i, j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end
##


Nt=200
points = [@SVector rand(3) for i = 1:Nt]




##
points2 = [((@SVector rand(3)))*2 + SVector(2, 0, 0) for i = 1:Nt]
for x = [2, 3] 
    for y = [0, 1, 2, 3]
        append!(points2, [(@SVector rand(3)) + SVector(x, y, 0) for i = 1:Nt])
    end
end

for x = [0, 1, 2, 3]
    for y = [2, 3]
        append!(points2, [(@SVector rand(3)) + SVector(x, y, 0) for i = 1:Nt])
    end
end

##far field
Nfar=10
for y = [0, 2, 4] 
    append!(points2, [(@SVector rand(3)) + SVector(4, y, 0) for i = 1:Nfar])
end
for x = [0, 2] 
    append!(points2, [(@SVector rand(3)) + SVector(x, 4, 0) for i = 1:Nfar])
end
##
@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, points[tdata], points[sdata])

A = assembler(OneoverRkernel, points2, points)

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

am = allocate_aca_memory(Float64, size(lm, 1), size(lm, 2); maxrank=30)

filld = FastBEAST.FillDistance(points2)
tfd = FastBEAST.TrueFillDistance(points2)

# classic aca filldistance
@time U, V = aca(lm, rowpivstrat=filld, tol=1e-4);
norm(U*V-A)/norm(A)

##
@time U, V = aca(lm, tol=1e-4);
norm(U*V-A)/norm(A)


##

function partialACA( 
    M::FastBEAST.LazyMatrix{I, K},
    am;
    rowpivstrat=FastBEAST.MaxPivoting(1),
    columnpivstrat=FastBEAST.MaxPivoting(1),
    tol=1e-14
) where {I, K} 

    (maxrows, maxcolumns) = size(M)

    rowpivstrat, nextrow = FastBEAST.firstindex(rowpivstrat, M.τ)
    column = FastBEAST.pivoting(
        rowpivstrat,
        abs.(am.V[am.Ic, 1:maxcolumns]),
        am.used_J[1:maxcolumns],
        M.σ
    )
    am.used_I[nextrow] = true

    for i = 1:10
        nextrow = FastBEAST.pivoting(
            rowpivstrat,
            abs.(am.U[1:maxrows, am.Jc]),
            am.used_I[1:maxrows],
            M.τ
        )
        am.used_I[nextrow] = true
        println(nextrow)
    end
end

partialACA(lm, rowpivstrat=filld, am)