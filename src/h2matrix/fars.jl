using FLoops

rows(lbk::MatrixBlock{I, F, LowRankMatrix{F}}) where F = lbk.τ
cols(lbk::MatrixBlock{I, F, LowRankMatrix{F}}) where F = lbk.σ

localrows(lbk::MatrixBlock{I, F, LowRankMatrix{F}}) where F = lbk.M.τ
localcols(lbk::MatrixBlock{I, F, LowRankMatrix{F}}) where F= lbk.M.σ

globalrows(lbk::MatrixBlock{I, F, LowRankMatrix{F}}) where F = lbk.τ[lbk.M.τ]
globalcols(lbk::MatrixBlock{I, F, LowRankMatrix{F}}) where F = lbg.σ[lbk.M.σ]

function getlowrankblocks(
    assembler::Function,
    fars::Vector{Vector{Tuple{I,I}}},
    test_fars::Vector{Pivotlrb{K}},
    trial_fars::Vector{Pivotlrb{K}};
    multithreading=true
) where {I, K}
    lowrankblocks = Vector{H2MatrixBlock{Int,K}}(undef, sum([length(far) for far in fars]))
    fars = reduce(vcat, fars)

    if multithreading
        @floop for (idx, far) in enumerate(fars)
            blk = zeros(
                K,
                length(test_fars[far[1]].τ[test_fars[far[1]].M.τ]),
                length(trial_fars[far[2]].σ[trial_fars[far[2]].M.σ]),
            )
            assembler(
                blk,
                test_fars[far[1]].τ[test_fars[far[1]].M.τ],
                trial_fars[far[2]].σ[trial_fars[far[2]].M.σ],
            )
            lowrankblocks[idx] = H2MatrixBlock(
                MatrixBlock(
                    blk,
                    test_fars[far[1]].τ[test_fars[far[1]].M.τ],
                    trial_fars[far[2]].σ[trial_fars[far[2]].M.σ],
                ),
                test_fars[far[1]].τ,
                trial_fars[far[2]].σ,
                far[1],
                far[2],
            )
        end
    else
        for (idx, far) in enumerate(fars)
            blk = zeros(
                K,
                length(test_fars[far[1]].τ[test_fars[far[1]].M.τ]),
                length(trial_fars[far[2]].σ[trial_fars[far[2]].M.σ]),
            )
            assembler(
                blk,
                test_fars[far[1]].τ[test_fars[far[1]].M.τ],
                trial_fars[far[2]].σ[trial_fars[far[2]].M.σ],
            )
            lowrankblocks[idx] = H2MatrixBlock(
                MatrixBlock(
                    blk,
                    test_fars[far[1]].τ[test_fars[far[1]].M.τ],
                    trial_fars[far[2]].σ[trial_fars[far[2]].M.σ],
                ),
                test_fars[far[1]].τ,
                trial_fars[far[2]].σ,
                far[1],
                far[2],
            )
        end
    end

    return lowrankblocks
end

function getcompressedmatrix(
    test_idcs::Vector{I},
    trial_idcs::Vector{I},
    roworcolidcs::Vector{I},
    roworcolmaps::Vector{UnitRange{I}},
    childrange::Vector{Tuple{I, UnitRange{I}}},
    matrixassembler::Function,
    ::Type{K};
    compressor=FastBEAST.ACAOptions(tol=1e-4)
) where {I, K}

    lm = FastBEAST.LazyMatrix(
        matrixassembler,
        test_idcs,
        trial_idcs,
        K
    )

    maxrank = max(Int(round(
        length(test_idcs) * length(trial_idcs)/(length(test_idcs) + length(trial_idcs)))),
        1
    )

    am = allocate_aca_memory(
        K,
        length(test_idcs),
        length(trial_idcs),
        maxrank=maxrank
    )

    U, V, rowindices, colindices = aca(
        lm,
        am;
        rowpivstrat=compressor.rowpivstrat,
        columnpivstrat=compressor.columnpivstrat,
        tol=compressor.tol,
        maxrank=maxrank,
        svdrecompress=compressor.svdrecompress,
        rcindices=true
    )

    return FastBEAST.Pivotlrb{K}(
        ClusterMatrix(U, V, rowindices, colindices),    
        test_idcs,
        trial_idcs,
        roworcolidcs,
        roworcolmaps,
        childrange
    )
end