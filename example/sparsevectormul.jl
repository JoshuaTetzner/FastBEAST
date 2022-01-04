using BEAST
using FastBEAST
using Printf
using LinearAlgebra
using StaticArrays
using SparseArrays

##
function logkernel(testpoint::SVector{2,T}, sourcepoint::SVector{2,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e1)
        return 0.0
    else
        return - 2*π*log(norm(testpoint - sourcepoint))
    end
end

function assembler(kernel, testpoints, sourcepoints)
    kernelmatrix = zeros(
        promote_type(eltype(testpoints[1]),eltype(sourcepoints[1])), 
        length(testpoints), 
        length(sourcepoints)
    )

    for i = 1:length(testpoints)
        for j = 1:length(sourcepoints)
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end

    return kernelmatrix
end

function assembler(kernel, matrix, testpoints, sourcepoints)
    for i = 1:length(testpoints)
        for j = 1:length(sourcepoints)
            matrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end

##
N =  5000
NT = 1000
sparsevector = Array(sprand(Float64,N,0.001))

spoints = [@SVector rand(2) for i = 1:N]
tpoints = 0.1*[@SVector rand(2) for i = 1:NT] + [SVector(1.0, 1.0) for i = 1:NT]

logkernelassembler(matrix, tdata, sdata) = assembler(
    logkernel,
    matrix,
    tpoints[tdata],
    spoints[sdata]
)

stree = create_tree(spoints, KMeansTreeOptions(iterations=100, nchildren=2, nmin=100))
ttree = create_tree(tpoints, KMeansTreeOptions(iterations=100, nchildren=2, nmin=100))
@printf("kmat assembly time: \n")
@time kmat = assembler(logkernel, tpoints, spoints)
@printf("kmat sparsmultiplication time: \n")
@time solution_true = kmat*sparsevector;

@printf("hmat assembly time: \n")
@time hmat = HMatrix(logkernelassembler, ttree, stree, T=Float64)
@printf("hmat sparsmultiplication time: \n")
@time solution = sparsevectormul(hmat,sparsevector)

@printf("\n Accuracy test (hmat): %.2e\n", estimate_reldifference(hmat, kmat))
@printf("\n Compression rate (hmat): %.2f %%\n", compressionrate(hmat)*100)

@printf("Accuracy of sparsevectormul:  %.2e\n", norm(solution_true-solution)/norm(solution_true))

##

N = 200
spoints = [@SVector rand(2) for i = 1:N]
sparsevector = Array(sprand(Float64,N,0.001))

logkernelassembler(matrix, tdata, sdata) = assembler(
    logkernel, 
    matrix, 
    spoints[tdata], 
    spoints[sdata]
)

stree = create_tree(spoints, BoxTreeOptions(nmin=200))
@printf("kmat assembly time: \n")
@time kmat = assembler(logkernel, spoints, spoints)
@printf("kmat sparsmultiplication time: \n")
@time solution_true = kmat*sparsevector;

@printf("hmat assembly time: \n")
@time hmat = HMatrix(logkernelassembler, stree, stree, T=Float64)
@printf("hmat sparsmultiplication time: \n")
@time solution = sparsevectormul(hmat,sparsevector)

@printf("\n Accuracy test (hmat): %.2e\n", estimate_reldifference(hmat, kmat))
@printf("\n Compression rate (hmat): %.2f %%\n", compressionrate(hmat)*100)

@printf("Accuracy of sparsevectormul:  %.2e\n", norm(solution_true-solution))
