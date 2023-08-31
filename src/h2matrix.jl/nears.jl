using LinearAlgebra
using LinearMaps
using ProgressMeter
using Base.Threads

function compute_nears(
    block_tree, 
    nears,
    assembler;
    ismultithreaded=true,
    verbose=false
)

    MBF = MatrixBlock{Int, Float64, Matrix{Float64}}
    nearblocks_perthread = Vector{MBF}[]
    nearblocks = MBF[]
    nonzeros = 0

    if ismultithreaded
        for near in nears
            nonzeros += length(value(block_tree.test_cluster, near[1])) * 
                length(value(block_tree.trial_cluster, near[2]))
            push!(
                nearblocks,
                fullmatrix(
                    assembler,
                    value(block_tree.test_cluster, near[1]),
                    value(block_tree.trial_cluster, near[2]),
                    Int,
                    Float64
                )
            )
            verbose && next!(p)
        end
    elseif multithreading
        for i in 1:Threads.nthreads()
            push!(nearblocks_perthread, MBF[])
            push!(nonzeros_perthread, 0)
        end

        Threads.@threads for near in nears
            nonzeros_perthread[Threads.threadid()] += 
                length(value(block_tree.test_cluster, near[1])) * 
                length(value(block_tree.trial_cluster, near[2]))

            push!(
                nearblocks_perthread[Threads.threadid()],
                fullmatrix(
                    assembler,
                    value(block_tree.test_cluster, near[1]),
                    value(block_tree.trial_cluster, near[2]),
                    Int,
                    Float64
                )
            )
            verbose && next!(p)
        end

        for i in eachindex(fnearblocks_perthread)
            append!(nearblocks, nearblocks_perthread[i])
        end
    end
    
    return nearblocks
end

function fullmatrix(
    matrixassembler,
    testnode,
    sourcenode,
    ::Type{I},
    ::Type{K};
) where {I, K}
    matrix = zeros(K, length(testnode), length(sourcenode))
    matrixassembler(matrix, testnode, sourcenode)

    return MatrixBlock{I, K, Matrix{K}}(
        matrix,
        testnode,
        sourcenode
    )
end