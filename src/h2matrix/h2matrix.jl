using BEAST
using FastBEAST
using ClusterTrees
using LinearMaps
using FLoops

#storage for pivoting
struct ClusterMatrix{F} <: LinearMaps.LinearMap{F}
    U::Matrix{F}
    V::Matrix{F}
    τ::Vector{Int}
    σ::Vector{Int}
end

Base.size(lrm::ClusterMatrix) = (size(lrm.U, 1), size(lrm.V, 2))

#storage for pivoting
struct Pivotlrb{F}
    M::ClusterMatrix{F}
    τ::Vector{Int}
    σ::Vector{Int}
    roworcolidcs::Vector{Int}
    roworcolmap::Vector{UnitRange{Int}}
    children::Vector{Tuple{Int,UnitRange{Int}}}
end

struct H2MatrixBlock{I,K}
    Z::MatrixBlock{I,K,Matrix{K}}
    τ::Vector{I}
    σ::Vector{I}
    row_basis::I#Vector{I}
    col_basis::I#Vector{I}
end

struct H2BasisBlock{I,K}
    T::Union{Vector{Matrix{K}},Matrix{K}}
    children::Vector{I}
end

struct H2Matrix{I,K} <: LinearMaps.LinearMap{K}
    fullrankblocks::Vector{MatrixBlock{I,K,Matrix{K}}}
    lowrankblocks::Vector{H2MatrixBlock{I,K}}
    nestedtestbases::Vector{H2BasisBlock{I,K}}
    nestedtrialbases::Vector{H2BasisBlock{I,K}}
    fars
    rowdim::I
    columndim::I
    ismultithreaded::Bool
end

function fulltestblock(blks::Vector{FastBEAST.H2BasisBlock{I,K}}, idx::I) where {I,K}
    if blks[idx].children == []
        return blks[idx].T
    else
        fblk = fulltestblock(blks, blks[idx].children[1]) * blks[idx].T[1]
        for childidx in 2:length(blks[idx].children)
            fblk = vcat(
                fblk,
                fulltestblock(blks, blks[idx].children[childidx]) * blks[idx].T[childidx],
            )
        end

        return fblk
    end
end

function fulltrialblock(blks::Vector{FastBEAST.H2BasisBlock{I,K}}, idx::I) where {I,K}
    if blks[idx].children == []
        return blks[idx].T
    else
        fblk = blks[idx].T[1] * fulltrialblock(blks, blks[idx].children[1])
        for childidx in 2:length(blks[idx].children)
            fblk = hcat(
                fblk,
                blks[idx].T[childidx] * fulltrialblock(blks, blks[idx].children[childidx]),
            )
        end

        return fblk
    end
end

function ismultithreaded(h2mat::HT) where {HT<:H2Matrix}
    return h2mat.ismultithreaded
end

function Base.size(h2mat::H2Matrix, dim=nothing)
    if dim === nothing
        return (h2mat.rowdim, h2mat.columndim)
    elseif dim == 1
        return h2mat.rowdim
    elseif dim == 2
        return h2mat.columndim
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(h2mat::Adjoint{T}, dim=nothing) where {T<:H2Matrix}
    if dim === nothing
        return reverse(size(adjoint(h2mat)))
    elseif dim == 1
        return size(adjoint(h2mat), 2)
    elseif dim == 2
        return size(adjoint(h2mat), 1)
    else
        error("dim must be either 1 or 2")
    end
end

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::H2Matrix, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    if !ismultithreaded(A)
        for mb in A.fullrankblocks
            y[mb.τ] .+= mb.M * x[mb.σ]
        end

        for mb in A.lowrankblocks
            y[mb.τ] .+=
                fulltestblock(A.nestedtestbases, mb.row_basis) *
                mb.Z.M *
                (fulltrialblock(A.nestedtrialbases, mb.col_basis) * x[mb.σ])
        end
    else
        yy = zeros(eltype(y), size(A, 1), Threads.nthreads())

        @floop for mb in A.fullrankblocks
            yy[mb.τ, Threads.threadid()] .+= mb.M * x[mb.σ]
        end

        @floop for mb in A.lowrankblocks
            yy[mb.τ, Threads.threadid()] .+=
                fulltestblock(A.nestedtestbases, mb.row_basis) *
                mb.Z.M *
                (fulltrialblock(A.nestedtrialbases, mb.col_basis) * x[mb.σ])
        end
        y[:] = sum(yy; dims=2)
    end

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, transA::LinearMaps.TransposeMap{<:Any,<:H2Matrix}, x::AbstractVector
)
    LinearMaps.check_dim_mul(y, transA.lmap, x)

    fill!(y, zero(eltype(y)))

    if !ismultithreaded(transA.lmap)
        for mb in transA.lmap.fullrankblocks
            y[mb.σ] .+= transpose(mb.M) * x[mb.τ]
        end

        for mb in transA.lmap.lowrankblocks
            y[mb.σ] .+=
                transpose(fulltrialblock(transA.lmap.nestedtrialbases, mb.col_basis)) *
                transpose(mb.Z.M) *
                (transpose(fulltestblock(transA.lmap.nestedtestbases, mb.row_basis)) * x[mb.τ])
        end
    else
        yy = zeros(eltype(y), size(transA.lmap, 1), Threads.nthreads())

        @floop for mb in transA.lmap.fullrankblocks
            yy[mb.σ, Threads.threadid()] .+= transpose(mb.M) * x[mb.τ]
        end

        @floop for mb in transA.lmap.lowrankblocks
            yy[mb.σ, Threads.threadid()] .+=
                transpose(fulltrialblock(transA.lmap.nestedtrialbases, mb.col_basis)) *
                transpose(mb.Z.M) *
                (transpose(fulltestblock(transA.lmap.nestedtestbases, mb.row_basis)) * x[mb.τ])
        end
        y[:] = sum(yy; dims=2)
    end
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, transA::LinearMaps.AdjointMap{<:Any,<:H2Matrix}, x::AbstractVector
)
    LinearMaps.check_dim_mul(y, transA.lmap, x)

    fill!(y, zero(eltype(y)))

    if !ismultithreaded(transA.lmap)
        for mb in transA.lmap.fullrankblocks
            y[mb.σ] .+= adjoint(mb.M) * x[mb.τ]
        end

        for mb in transA.lmap.lowrankblocks
            y[mb.σ] .+=
                adjoint(fulltrialblock(transA.lmap.nestedtrialbases, mb.col_basis)) *
                adjoint(mb.Z.M) *
                (adjoint(fulltestblock(transA.lmap.nestedtestbases, mb.row_basis)) * x[mb.τ])
        end
    else
        yy = zeros(eltype(y), size(transA.lmap, 1), Threads.nthreads())

        @floop for mb in transA.lmap.fullrankblocks
            yy[mb.σ, Threads.threadid()] .+= adjoint(mb.M) * x[mb.τ]
        end

        @floop for mb in transA.lmap.lowrankblocks
            yy[mb.σ, Threads.threadid()] .+=
                adjoint(fulltrialblock(transA.lmap.nestedtrialbases, mb.col_basis)) *
                adjoint(mb.Z.M) *
                (adjoint(fulltestblock(transA.lmap.nestedtestbases, mb.row_basis)) * x[mb.τ])
        end
        y[:] = sum(yy; dims=2)
    end

    return y
end

function H2Matrix(
    matrixassembler::Function,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    ::Type{K};
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    multithreading=true,
) where {T,K}
    nears, fars = computeinteractions(tree)
    println("Computing near interactions")
    @time fullrankblocks = getfullrankblocks(
        tree, nears, matrixassembler, K; multithreading=multithreading
    )

    println("Pivot selection")
    @time test_fars = test_top_down_pivots(
        tree, fars, matrixassembler, K; compressor=compressor, multithreading=multithreading
    )
    trial_fars = trial_top_down_pivots(
        tree, fars, matrixassembler, K; compressor=compressor, multithreading=multithreading
    )

    println("Compute blocks")
    @time lowrankblocks = FastBEAST.getlowrankblocks(
        matrixassembler, fars, test_fars, trial_fars; multithreading=multithreading
    )

    println("Build Basis")
    test_basis = build_test_bases(tree.test_cluster, test_fars, K)
    trial_basis = build_trial_bases(tree.trial_cluster, trial_fars, K)
    rowdim = length(value(tree.test_cluster, 1))
    coldim = length(value(tree.trial_cluster, 1))

    return H2Matrix(
        fullrankblocks,
        lowrankblocks,
        test_basis,
        trial_basis,
        fars,
        rowdim,
        coldim,
        multithreading,
    )
end

function build_test_bases(
    tree::KMeansTree{T}, test_fars, ::Type{K}; multithreading=true
) where {T,K}
    test_basis = Vector{H2BasisBlock{Int,K}}(undef, length(tree.nodes))

    if multithreading
        @floop for idx in eachindex(test_fars)
            if isassigned(test_fars, idx)
                testfar = test_fars[idx]
                if testfar.children == []
                    test_basis[idx] = H2BasisBlock(
                        test_fars[idx].M.U * test_fars[idx].M.U[test_fars[idx].M.τ, :]^-1,
                        Int[],
                    )
                else
                    transfer = Vector{Matrix{K}}(undef, length(testfar.children))
                    children = zeros(Int, length(testfar.children))

                    for (ind, (child, range)) in enumerate(testfar.children)
                        transfer[ind] =
                            testfar.M.U[range, :][test_fars[child].M.τ, :] *
                            testfar.M.U[testfar.M.τ, :]^-1
                        children[ind] = child
                    end

                    test_basis[idx] = H2BasisBlock(transfer, children)
                end
            end
        end
    else
        for idx in eachindex(test_fars)
            if isassigned(test_fars, idx)
                testfar = test_fars[idx]
                if testfar.children == []
                    test_basis[idx] = H2BasisBlock(
                        test_fars[idx].M.U * test_fars[idx].M.U[test_fars[idx].M.τ, :]^-1,
                        Int[],
                    )
                else
                    transfer = Vector{Matrix{K}}(undef, length(testfar.children))
                    children = zeros(Int, length(testfar.children))

                    for (ind, (child, range)) in enumerate(testfar.children)
                        transfer[ind] =
                            testfar.M.U[range, :][test_fars[child].M.τ, :] *
                            testfar.M.U[testfar.M.τ, :]^-1
                        children[ind] = child
                    end

                    test_basis[idx] = H2BasisBlock(transfer, children)
                end
            end
        end
    end

    return test_basis
end

function build_trial_bases(
    tree::KMeansTree{T}, trial_fars, ::Type{K}; multithreading=true
) where {T,K}
    trial_basis = Vector{H2BasisBlock{Int,K}}(undef, length(tree.nodes))

    if multithreading
        @floop for idx in eachindex(trial_fars)
            if isassigned(trial_fars, idx)
                trialfar = trial_fars[idx]
                if trialfar.children == []
                    trial_basis[idx] = H2BasisBlock(
                        trial_fars[idx].M.V[:, trial_fars[idx].M.σ]^-1 *
                        trial_fars[idx].M.V,
                        Int[],
                    )
                else
                    transfer = Vector{Matrix{K}}(undef, length(trialfar.children))
                    children = zeros(Int, length(trialfar.children))

                    for (ind, (child, range)) in enumerate(trialfar.children)
                        transfer[ind] =
                            trialfar.M.V[:, trialfar.M.σ]^-1 *
                            trialfar.M.V[:, range][:, trial_fars[child].M.σ]

                        children[ind] = child
                    end
                    trial_basis[idx] = H2BasisBlock(transfer, children)
                end
            end
        end
    else
        for idx in eachindex(trial_fars)
            if isassigned(trial_fars, idx)
                trialfar = trial_fars[idx]
                if trialfar.children == []
                    trial_basis[idx] = H2BasisBlock(
                        trial_fars[idx].M.V[:, trial_fars[idx].M.σ]^-1 *
                        trial_fars[idx].M.V,
                        Int[],
                    )
                else
                    transfer = Vector{Matrix{K}}(undef, length(trialfar.children))
                    children = zeros(Int, length(trialfar.children))

                    for (ind, (child, range)) in enumerate(trialfar.children)
                        transfer[ind] =
                            trialfar.M.V[:, trialfar.M.σ]^-1 *
                            trialfar.M.V[:, range][:, trial_fars[child].M.σ]

                        children[ind] = child
                    end
                    trial_basis[idx] = H2BasisBlock(transfer, children)
                end
            end
        end
    end

    return trial_basis
end

function test_top_down_pivots(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    fars,
    matrixassembler,
    ::Type{K};
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    multithreading=true,
) where {T,K}
    test_lowrankblocks = Vector{Pivotlrb{K}}(undef, length(block_tree.test_cluster.nodes))

    interactionlist = sort_interactions_tree(
        length(block_tree.test_cluster.nodes), fars; testortrial=1
    )

    if multithreading
        levelednodes = Vector{Int}[]
        for level in eachindex(block_tree.test_cluster.levels)
            nodes = Int[]
            for node in FastBEAST.clusterlink(block_tree.test_cluster; target=level)
                push!(nodes, node)
            end
            push!(levelednodes, nodes)
        end

        for level in levelednodes
            @floop for nodeidx in level
                childidcs = FastBEAST.childidcs(block_tree.test_cluster, nodeidx)
                childrange = Tuple{Int,UnitRange{Int}}[]
                if ClusterTrees.haschildren(block_tree.test_cluster, nodeidx)
                    childidcs = FastBEAST.childidcs(block_tree.test_cluster, nodeidx)

                    idxcounter = 1
                    for childidx in childidcs
                        push!(
                            childrange,
                            (
                                childidx,
                                idxcounter:(idxcounter + length(value(block_tree.test_cluster, childidx)) - 1),
                            ),
                        )
                        idxcounter += length(value(block_tree.test_cluster, childidx))
                    end
                end

                col_set = Int[]
                colmaps = UnitRange{Int}[]
                idx = 1
                if interactionlist[nodeidx] != []
                    for adm_blk in interactionlist[nodeidx]
                        append!(col_set, value(block_tree.trial_cluster, adm_blk))
                        push!(
                            colmaps,
                            idx:(idx + length(value(block_tree.trial_cluster, adm_blk)) - 1),
                        )
                        idx += length(value(block_tree.trial_cluster, adm_blk))
                    end
                end
                if ClusterTrees.parent(block_tree.test_cluster, nodeidx) != 0 && isassigned(
                    test_lowrankblocks,
                    ClusterTrees.parent(block_tree.test_cluster, nodeidx),
                )
                    parent = ClusterTrees.parent(block_tree.test_cluster, nodeidx)
                    append!(
                        col_set,
                        test_lowrankblocks[parent].σ[test_lowrankblocks[parent].M.σ],
                    )
                end
                if col_set != []
                    row_set = value(block_tree.test_cluster, nodeidx)

                    test_lowrankblocks[nodeidx] = getcompressedmatrix(
                        row_set,
                        col_set,
                        [nodeidx],
                        colmaps,
                        childrange,
                        matrixassembler,
                        K;
                        compressor=compressor,
                    )
                end
            end
        end
    else
        for level in eachindex(block_tree.test_cluster.levels)
            for nodeidx in FastBEAST.clusterlink(block_tree.test_cluster; target=level)
                childidcs = FastBEAST.childidcs(block_tree.test_cluster, nodeidx)
                childrange = Tuple{Int,UnitRange{Int}}[]
                if ClusterTrees.haschildren(block_tree.test_cluster, nodeidx)
                    childidcs = FastBEAST.childidcs(block_tree.test_cluster, nodeidx)

                    idxcounter = 1
                    for childidx in childidcs
                        push!(
                            childrange,
                            (
                                childidx,
                                idxcounter:(idxcounter + length(value(block_tree.test_cluster, childidx)) - 1),
                            ),
                        )
                        idxcounter += length(value(block_tree.test_cluster, childidx))
                    end
                end

                col_set = Int[]
                colmaps = UnitRange{Int}[]
                idx = 1
                if interactionlist[nodeidx] != []
                    for adm_blk in interactionlist[nodeidx]
                        append!(col_set, value(block_tree.trial_cluster, adm_blk))
                        push!(
                            colmaps,
                            idx:(idx + length(value(block_tree.trial_cluster, adm_blk)) - 1),
                        )
                        idx += length(value(block_tree.trial_cluster, adm_blk))
                    end
                end
                if ClusterTrees.parent(block_tree.test_cluster, nodeidx) != 0 && isassigned(
                    test_lowrankblocks,
                    ClusterTrees.parent(block_tree.test_cluster, nodeidx),
                )
                    parent = ClusterTrees.parent(block_tree.test_cluster, nodeidx)
                    append!(
                        col_set,
                        test_lowrankblocks[parent].σ[test_lowrankblocks[parent].M.σ],
                    )
                end
                if col_set != []
                    row_set = value(block_tree.test_cluster, nodeidx)

                    test_lowrankblocks[nodeidx] = getcompressedmatrix(
                        row_set,
                        col_set,
                        [nodeidx],
                        colmaps,
                        childrange,
                        matrixassembler,
                        K;
                        compressor=compressor,
                    )
                end
            end
        end
    end

    return test_lowrankblocks
end

function trial_top_down_pivots(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    fars,
    matrixassembler,
    ::Type{K};
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    multithreading=true,
) where {T,K}
    trial_lowrankblocks = Vector{Pivotlrb{K}}(undef, length(block_tree.trial_cluster.nodes))

    interactionlist = sort_interactions_tree(
        length(block_tree.trial_cluster.nodes), fars; testortrial=2
    )

    if multithreading
        levelednodes = Vector{Int}[]
        for level in eachindex(block_tree.trial_cluster.levels)
            nodes = Int[]
            for node in FastBEAST.clusterlink(block_tree.trial_cluster; target=level)
                push!(nodes, node)
            end
            push!(levelednodes, nodes)
        end

        for level in levelednodes
            @floop for nodeidx in level
                childidcs = FastBEAST.childidcs(block_tree.trial_cluster, nodeidx)
                childrange = Tuple{Int,UnitRange{Int}}[]
                if ClusterTrees.haschildren(block_tree.trial_cluster, nodeidx)
                    childidcs = FastBEAST.childidcs(block_tree.trial_cluster, nodeidx)

                    idxcounter = 1
                    for childidx in childidcs
                        push!(
                            childrange,
                            (
                                childidx,
                                idxcounter:(idxcounter + length(value(block_tree.trial_cluster, childidx)) - 1),
                            ),
                        )
                        idxcounter += length(value(block_tree.trial_cluster, childidx))
                    end
                end

                row_set = Int[]
                rowmaps = UnitRange{Int}[]
                idx = 1
                if interactionlist[nodeidx] != []
                    for adm_blk in interactionlist[nodeidx]
                        append!(row_set, value(block_tree.test_cluster, adm_blk))
                        push!(
                            rowmaps,
                            idx:(idx + length(value(block_tree.test_cluster, adm_blk)) - 1),
                        )
                        idx += length(value(block_tree.test_cluster, adm_blk))
                    end
                end
                if ClusterTrees.parent(block_tree.trial_cluster, nodeidx) != 0 &&
                    isassigned(
                    trial_lowrankblocks,
                    ClusterTrees.parent(block_tree.trial_cluster, nodeidx),
                )
                    parent = ClusterTrees.parent(block_tree.trial_cluster, nodeidx)
                    append!(
                        row_set,
                        trial_lowrankblocks[parent].τ[trial_lowrankblocks[parent].M.τ],
                    )
                end

                if row_set != []
                    col_set = value(block_tree.trial_cluster, nodeidx)
                    trial_lowrankblocks[nodeidx] = getcompressedmatrix(
                        row_set,
                        col_set,
                        [nodeidx],
                        rowmaps,
                        childrange,
                        matrixassembler,
                        K;
                        compressor=compressor,
                    )
                end
            end
        end
    else
        for level in eachindex(block_tree.trial_cluster.levels)
            for nodeidx in FastBEAST.clusterlink(block_tree.trial_cluster; target=level)
                childidcs = FastBEAST.childidcs(block_tree.trial_cluster, nodeidx)
                childrange = Tuple{Int,UnitRange{Int}}[]
                if ClusterTrees.haschildren(block_tree.trial_cluster, nodeidx)
                    childidcs = FastBEAST.childidcs(block_tree.trial_cluster, nodeidx)

                    idxcounter = 1
                    for childidx in childidcs
                        push!(
                            childrange,
                            (
                                childidx,
                                idxcounter:(idxcounter + length(value(block_tree.trial_cluster, childidx)) - 1),
                            ),
                        )
                        idxcounter += length(value(block_tree.trial_cluster, childidx))
                    end
                end

                row_set = Int[]
                rowmaps = UnitRange{Int}[]
                idx = 1
                if interactionlist[nodeidx] != []
                    for adm_blk in interactionlist[nodeidx]
                        append!(row_set, value(block_tree.test_cluster, adm_blk))
                        push!(
                            rowmaps,
                            idx:(idx + length(value(block_tree.test_cluster, adm_blk)) - 1),
                        )
                        idx += length(value(block_tree.test_cluster, adm_blk))
                    end
                end
                if ClusterTrees.parent(block_tree.trial_cluster, nodeidx) != 0 &&
                    isassigned(
                    trial_lowrankblocks,
                    ClusterTrees.parent(block_tree.trial_cluster, nodeidx),
                )
                    parent = ClusterTrees.parent(block_tree.trial_cluster, nodeidx)
                    append!(
                        row_set,
                        trial_lowrankblocks[parent].τ[trial_lowrankblocks[parent].M.τ],
                    )
                end

                if row_set != []
                    col_set = value(block_tree.trial_cluster, nodeidx)
                    trial_lowrankblocks[nodeidx] = getcompressedmatrix(
                        row_set,
                        col_set,
                        [nodeidx],
                        rowmaps,
                        childrange,
                        matrixassembler,
                        K;
                        compressor=compressor,
                    )
                end
            end
        end
    end

    return trial_lowrankblocks
end
