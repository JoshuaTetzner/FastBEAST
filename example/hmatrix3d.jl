using FastBEAST
using StaticArrays
using LinearAlgebra
using Printf

function OneoverRkernel(testpoint::SVector{3,T}, sourcepoint::SVector{3,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e1)
        return 0.0
    else
        return 1.0 / (norm(testpoint - sourcepoint))
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
N =  1000
NT = N

spoints = [@SVector rand(3) for i = 1:N]
tpoints = 0.1*[@SVector rand(3) for i = 1:NT] + [1.0*SVector(3.5, 3.5, 3.5) for i = 1:NT]

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
    OneoverRkernel,
    matrix,
    tpoints[tdata],
    spoints[sdata]
)

stree = create_tree(spoints, BoxTreeOptions(nmin=50))
ttree = create_tree(tpoints, BoxTreeOptions(nmin=50))
kmat = assembler(OneoverRkernel, tpoints, spoints)
hmat = HMatrix(OneoverRkernelassembler, ttree, stree, compressor=:naive, Int64, Float64)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)

##
N = 3000
NT = N

spoints = [@SVector rand(3) for i = 1:N]

##
@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
    OneoverRkernel,
    matrix,
    spoints[tdata],
    spoints[sdata]
)

stree = create_tree(spoints, BoxTreeOptions(nmin=100))
kmat = assembler(OneoverRkernel, spoints, spoints)
hmat = HMatrix(OneoverRkernelassembler, stree, stree, compressor=:aca, Int64, Float64)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)

##
typeof(stree.children[1].data)
##
totalspoints = 0
for child in stree.children
    totalspoints += length(child.data.indices)
end

@assert totalspoints == N

##
function hmatrix3d_benchmark(N)
    spoints = [@SVector rand(3) for i = 1:N]

    @views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
        OneoverRkernel,
        matrix,
        spoints[tdata],
        spoints[sdata]
    )

    stree = create_tree(spoints, nmin=400)
    @time hmat = HMatrix(
        OneoverRkernelassembler,
        stree,
        stree,
        compressor=:aca,
        tol=1e-4,
        Int64, 
        Float64
    )
    
    @printf("Memory usage: %.2f GiB\n", nnz(hmat)*8/(1024*1024*1024.0))
    @printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)
    return nothing #stree, hmat
end


## Example1 a) Random distribution
N = 5000
spoints = [@SVector rand(3) for i = 1:N];
## Example2 a) Not evenly distributed sphere
spoints = [SVector((sin(i)*cos(j),sin(i)*sin(j),cos(i))) for j=0:0.1:2*pi for i = 0:0.1:pi];

## Example2 b) Not evenly distributed sphere like shape
spoints = [SVector((sin(i)*cos(j),sin(i)*sin(j),i^2*cos(i))) for j=0:0.1:2*pi for i = 0:0.1:2*pi];

##
pivoting(roworcolumn, acausedindices, totalindices) = FastBEAST.minimalfilldistance(
    spoints,
    roworcolumn,
    acausedindices,
    totalindices
)
##
@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
    OneoverRkernel,
    matrix,
    spoints[tdata],
    spoints[sdata]
)
stree = create_tree(spoints, BoxTreeOptions(nmin=100))
kmat = assembler(OneoverRkernel, spoints, spoints)
@time hmat = HMatrix(OneoverRkernelassembler, stree, stree, compressor=:aca, Int64, Float64)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)
##
stree = create_tree(spoints, KMeansTreeOptions(iterations=100,nchildren=2,nmin=40))
kmat = assembler(OneoverRkernel, spoints, spoints)
@time hmat = HMatrix(OneoverRkernelassembler, stree, stree, compressor=:aca, Int64, Float64)

@printf("Accuracy test: %.2e\n", estimate_reldifference(hmat,kmat))
@printf("Compression rate: %.2f %%\n", compressionrate(hmat)*100)


##
using Plots
plotlyjs()
test = zeros(Float64, 10000, 3)
ind = 1
for i = 0.0:0.2:2*pi
    for j = 0.0:0.2:2*pi
        test[ind, :] = [sin(i)*cos(j),sin(i)*sin(j),i^2*cos(i)]
        ind += 1
    end
end
scatter(test[:,1], test[:,2], test[:,3])