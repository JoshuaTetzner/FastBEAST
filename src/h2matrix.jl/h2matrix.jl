using LinearAlgebra
using LinearMaps
using ProgressMeter
using SparseArrays
using FastBEAST
using ClusterTrees


struct NestedMatrix{K}
    M::Matrix{K}
    cluster::Tuple{Int, Int}
    τ::Vector{Int}
    σ::Vector{Int}
end


struct NestedBasis{K}
    M::Matrix{K}
    children::Vector{Int}
end


isbasis(node::NestedMatrix) = (node.children == [])


struct H2Matrix{I, K} <: LinearMaps.LinearMap{K}
    fullrankblocks::Vector{MatrixBlock{I, K, Matrix{K}}}
    lowrankblocks::Vector{MatrixBlock{I, K, NestedMatrix{K}}}
    nestedrows::Vector{NestedBasis{K}}
    nestedcolumns::Vector{NestedBasis{K}}
    rowdim::I
    columndim::I
    nnz::I
end


function nnz(hmat::H2T) where H2T <: H2Matrix
    return hmat.nnz
end


function H2Matrix(
    matrixassembler::Function,
    #h2assembler::Function,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    ::Type{I},
    ::Type{K};
    farmatrixassembler=matrixassembler,
    compressor=FastBEAST.ACAOptions(),
    multithreading=false,
    verbose=false
) where {I, K, T}
    
    nears, fars = computeinteractions(tree)
    
    @time topdown_row_pivots(
        tree,
        fars,
        matrixassembler,    
        I,
        K
    )

    @time topdown_column_pivots(
        tree,
        fars,
        matrixassembler,    
        I,
        K
    )

    #h2mat = h2setup( 
    #    testmatrix, trialmatrix, testpivots, trialpivots, tree, nears, fars
    #)
end


function h2setup(
    testmatrix::SparseMatrixCSC{K, Int},
    trialmatrix::SparseMatrixCSC{K, Int},
    testpivots::Vector,
    trialpivots::Vector,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    nears::Vector{Tuple{Int, Int}},
    fars::Vector{Vector{Tuple{Int, Int}}},
    
) where {T, K}
   


end


struct H2Pivots
    τᵥ::Vector{Int}
    σᵤ::Vector{Int}
    σ::Vector{Int}
    τ::Vector{Int}
end


function topdown_row_pivots(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    fars,
    matrixassembler,    
    ::Type{I},
    ::Type{K};
    compressor=FastBEAST.ACAOptions(tol=1e-4)
) where {I, K, T}

    test_pivots = Vector{H2Pivots}(undef, length(block_tree.test_cluster.nodes))
    test_matrix = spzeros(
        K, block_tree.test_cluster.num_elements, block_tree.test_cluster.num_elements
    )

    for level_fars in fars
        if level_fars != []

            sorted_fars = FastBEAST.sort_interactions(level_fars, testortrial=1)
            for sfar in sorted_fars
                c = sfar[1][1]
                row_idcs = FastBEAST.value(block_tree.test_cluster, c)
                col_idcs = Int[]

                for adm_blk in sfar[2]
                    append!(col_idcs, FastBEAST.value(block_tree.trial_cluster, adm_blk))
                end
                
                if ClusterTrees.parent(block_tree.test_cluster, c) != 0 &&
                    isassigned(
                        test_pivots, ClusterTrees.parent(block_tree.test_cluster, c)
                    )
                    parent = ClusterTrees.parent(block_tree.test_cluster, c)
                    append!(col_idcs, test_pivots[parent].σᵤ)
                end
                

                compress_r_blk!(
                    c,
                    row_idcs,
                    col_idcs,
                    test_pivots,
                    test_matrix,
                    matrixassembler,
                    compressor=compressor
                ) 
            end
        end
    end

    return test_pivots, test_matrix
end


function topdown_column_pivots(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    fars,
    matrixassembler,    
    ::Type{I},
    ::Type{K};
    compressor=FastBEAST.ACAOptions(tol=1e-4)
) where {I, K, T}

    trial_pivots = Vector{H2Pivots}(undef, length(block_tree.test_cluster.nodes))
    trial_matrix = spzeros(
        K, block_tree.test_cluster.num_elements, block_tree.test_cluster.num_elements
    )

    for level_fars in fars
        if level_fars != []

            sorted_fars = sort_interactions(level_fars, testortrial=2)
            for sfar in sorted_fars
                c = sfar[2][1]
                col_idcs = value(block_tree.trial_cluster, c)
                row_idcs = Int[]
                

                for adm_blk in sfar[1]
                    append!(row_idcs, value(block_tree.test_cluster, adm_blk))
                end
                
                if ClusterTrees.parent(block_tree.trial_cluster, c) != 0 &&
                    isassigned(
                        trial_pivots, ClusterTrees.parent(block_tree.trial_cluster, c)
                    )
                    
                    append!(row_idcs, trial_pivots[
                        ClusterTrees.parent(block_tree.trial_cluster, c)
                    ].τᵥ)
                end

                compress_c_blk!(
                    c,
                    row_idcs,
                    col_idcs,
                    trial_pivots,
                    trial_matrix,
                    matrixassembler,
                    compressor=compressor
                ) 
            end
        end
    end

    return trial_pivots, trial_matrix
end

function compress_c_blk!(
    block_idx::Int,
    row_idcs::Vector{Int},
    col_idcs::Vector{Int},
    pivots::Vector{H2Pivots},
    matrix::SparseMatrixCSC{K, Int64},
    matrixassembler;
    compressor=FastBEAST.ACAOptions(tol=1e-4)
) where K

    lm = FastBEAST.LazyMatrix(
        matrixassembler,
        row_idcs,
        col_idcs,
        Float64
    )

    maxrank = Int(round(
        length(row_idcs) * length(col_idcs)/(length(row_idcs) + length(col_idcs))))

    am = allocate_aca_memory(
        K,
        length(row_idcs),
        length(col_idcs),
        maxrank=maxrank
    )

    retU, retV, U, V, τᵥ, σᵤ = nca(
        lm,
        am;
        rowpivstrat=compressor.rowpivstrat,
        columnpivstrat=compressor.columnpivstrat,
        tol=compressor.tol,
        svdrecompress=compressor.svdrecompress
    )

    pivots[block_idx] = H2Pivots(τᵥ, σᵤ, row_idcs, col_idcs)
    matrix[τᵥ, col_idcs] = V
end


function compress_r_blk!(
    block_idx::Int,
    row_idcs::Vector{Int},
    col_idcs::Vector{Int},
    pivots::Vector{H2Pivots},
    matrix::SparseMatrixCSC{K, Int64},
    matrixassembler;
    compressor=FastBEAST.ACAOptions(tol=1e-4)
) where K

    lm = FastBEAST.LazyMatrix(
        matrixassembler,
        row_idcs,
        col_idcs,
        Float64
    )

    maxrank = Int(round(
        length(row_idcs) * length(col_idcs)/(length(row_idcs) + length(col_idcs))))

    am = allocate_aca_memory(
        K,
        length(row_idcs),
        length(col_idcs),
        maxrank=maxrank
    )

    retU, retV, U, V, τᵥ, σᵤ = nca(
        lm,
        am;
        rowpivstrat=compressor.rowpivstrat,
        columnpivstrat=compressor.columnpivstrat,
        tol=compressor.tol,
        svdrecompress=compressor.svdrecompress
    )

    pivots[block_idx] = H2Pivots(τᵥ, σᵤ, row_idcs, col_idcs)
    matrix[row_idcs, σᵤ] = U
end
