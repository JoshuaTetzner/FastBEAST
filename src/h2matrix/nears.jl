using FLoops

function getfullrankblocks(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    nears,
    assembler,
    ::Type{K};
    multithreading=true
) where {K,T}

    nearblocks = Vector{MatrixBlock{Int,K,Matrix{K}}}(undef, length(nears))
    
    if multithreading
        @floop for (idx,near) in enumerate(nears)
            nearblocks[idx] = fullmatrix(
                assembler,
                value(block_tree.test_cluster, near[1]),
                value(block_tree.trial_cluster, near[2]),
                Int,
                K,
            )
        end
    else
        for (idx,near) in enumerate(nears)
            nearblocks[idx] = fullmatrix(
                assembler,
                value(block_tree.test_cluster, near[1]),
                value(block_tree.trial_cluster, near[2]),
                Int,
                K,
            )
        end
    end

    return nearblocks
end

function fullmatrix(
    matrixassembler::Function,
    testnode::Vector{I},
    sourcenode::Vector{I},
    ::Type{I},
    ::Type{K};
) where {I,K}
    matrix = zeros(K, length(testnode), length(sourcenode))
    matrixassembler(matrix, testnode, sourcenode)

    return MatrixBlock{I,K,Matrix{K}}(matrix, testnode, sourcenode)
end
