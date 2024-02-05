function getfullrankblocks(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T}, nears, assembler, ::Type{K}
) where {K,T}
    MBF = MatrixBlock{Int,K,Matrix{K}}
    nearblocks = MBF[]
    nonzeros = 0

    for near in nears
        nonzeros +=
            length(value(block_tree.test_cluster, near[1])) *
            length(value(block_tree.trial_cluster, near[2]))
        push!(
            nearblocks,
            fullmatrix(
                assembler,
                value(block_tree.test_cluster, near[1]),
                value(block_tree.trial_cluster, near[2]),
                Int,
                K,
            ),
        )
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
