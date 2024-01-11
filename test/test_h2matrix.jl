using Test
using FastBEAST
using StaticArrays
using LinearAlgebra
using ClusterTrees
 
##
function fullmat(h2mat::FastBEAST.H2Matrix{I, K}) where {I, K}

    fmat = zeros(K, h2mat.rowdim, h2mat.columndim)

    for frb in h2mat.fullrankblocks
        fmat[frb.τ, frb.σ] = frb.M
    end
    idx = 0
    for lrb in h2mat.lowrankblocks
        idx+=1
        #println(idx)
        fmat[lrb.τ, lrb.σ] = fulltestblock(
            h2mat.nestedtestbases, lrb.row_basis
        ) * lrb.Z.M * fulltrialblock(h2mat.nestedtrialbases, lrb.col_basis)
    end

    return fmat
end

function fulltestblock(blks::Vector{FastBEAST.H2BasisBlock{I, K}}, idx::I) where {I, K}

    if blks[idx].children == []
        return blks[idx].T
    else
        fblk = fulltestblock(blks, blks[idx].children[1])*blks[idx].T[1]
        for childidx in 2:length(blks[idx].children)
            fblk = vcat(fblk, fulltestblock(blks, blks[idx].children[childidx]) * blks[idx].T[childidx])
        end

        return fblk
    end
end

function fulltrialblock(blks::Vector{FastBEAST.H2BasisBlock{I, K}}, idx::I) where {I, K}
    if blks[idx].children == []
        return blks[idx].T
    else
        fblk = blks[idx].T[1]*fulltrialblock(blks, blks[idx].children[1])
        for childidx in 2:length(blks[idx].children)
            fblk = hcat(fblk, blks[idx].T[childidx]*fulltrialblock(blks, blks[idx].children[childidx]))
        end

        return fblk
    end
end

function compression(h2mat::FastBEAST.H2Matrix{I, K}) where {I, K}
    elements = 0
    for frb in h2mat.fullrankblocks
        elements += length(frb.M)
    end

    for lrb in h2mat.lowrankblocks
        elements += length(lrb.Z.M)
    end

    for i in 1:length(h2mat.nestedtrialbases)
        if isdefined(h2mat.nestedtrialbases, i)
            if h2mat.nestedtrialbases[i].T isa Vector
                for trans in h2mat.nestedtrialbases[i].T
                    elemts += length(trans)
                end
            else
                elements += length(h2mat.nestedtrialbases[i].T)
            end
        end
    end

    for i in 1:length(h2mat.nestedtestbases)
        if isdefined(h2mat.nestedtestbases, i)
            if h2mat.nestedtestbases[i].T isa Vector
                for trans in h2mat.nestedtestbases[i].T
                    elemts += length(trans)
                end
            else
                elements += length(h2mat.nestedtestbases[i].T)
            end
        end
    end

    return elements / (h2mat.rowdim * h2mat.columndim)

end

function compression(hmat::HT) where HT <: HMatrix
    fullsize = hmat.rowdim*hmat.columndim
    return nnz(hmat)/fullsize
end

##
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

function c_rate(hmat)
    elements = 0
    for lrb in hmat.lowrankblocks
        elements += size(lrb.M.U, 1)*size(lrb.M.U, 2) + size(lrb.M.V, 1)*size(lrb.M.U, 2)
    end

    for frb in hmat.fullrankblocks
        elements += size(frb.M, 1)*size(frb.M, 2) 
    end

    return elements/(size(hmat, 1)*size(hmat, 2))
end

##

N = 25000
points = [@SVector rand(3) for i = 1:N]

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(OneoverRkernel, matrix, points[tdata], points[sdata])

fmat = assembler(OneoverRkernel, points, points)
##
@time tree = create_CT_tree(points, nchildren=2, nmin=200, maxlevel=8);
block_tree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
#nears, fars = computeinteractions(block_tree)
@time h2mat = FastBEAST.H2Matrix(OneoverRkernelassembler, block_tree, points, points);

compression(h2mat)
fh2mat = fullmat(h2mat);
norm(fh2mat-fmat)/norm(fh2mat)

##
stree = create_tree(points, BoxTreeOptions(nmin=400));
ttree = create_tree(points, BoxTreeOptions(nmin=400));
@time hmat = HMatrix(OneoverRkernelassembler, ttree, stree, Int64, Float64, compressor=FastBEAST.ACAOptions(tol=1e-3));
estimate_reldifference(hmat,fmat)
c_rate(hmat)