rows(lbk::MatrixBlock{I, F, LowRankMatrix{F}}) where F = lbk.τ
cols(lbk::MatrixBlock{I, F, LowRankMatrix{F}}) where F = lbk.σ

localrows(lbk::MatrixBlock{I, F, LowRankMatrix{F}}) where F = lbk.M.τ
localcols(lbk::MatrixBlock{I, F, LowRankMatrix{F}}) where F= lbk.M.σ

globalrows(lbk::MatrixBlock{I, F, LowRankMatrix{F}}) where F = lbk.τ[lbk.M.τ]
globalcols(lbk::MatrixBlock{I, F, LowRankMatrix{F}}) where F = lbg.σ[lbk.M.σ]

function sort_interactions(
    fars::Vector{Tuple{Int, Int}};
    testortrial = 1
)
    
    testortrial == 1 ? trialortest = 2 : trialortest = 1
    
    sortedfars = Tuple{Vector{Int}, Vector{Int}}[]
    sfars = sort(fars, by = x -> x[testortrial])

    push!(sortedfars, ([sfars[1][1]],[sfars[1][2]]))
    for n ∈ 2:length(fars)
        if sfars[n][testortrial] == sfars[n-1][testortrial]
            push!(sortedfars[end][trialortest], sfars[n][trialortest])
        else
            push!(sortedfars, ([sfars[n][1]], [sfars[n][2]]))
        end
    end

    return sortedfars
end

function getcompressedmatrix(
    test_idcs::Vector{Int},
    trial_idcs::Vector{Int},
    roworcolidcs::Vector{Int},
    roworcolmaps::Vector{UnitRange{Int}},
    childrange::Vector{Tuple{Int, UnitRange{Int}}},
    matrixassembler;
    compressor=FastBEAST.ACAOptions(tol=1e-4)
)
    lm = FastBEAST.LazyMatrix(
        matrixassembler,
        test_idcs,
        trial_idcs,
        Float64
    )
    maxrank = Int(round(
        length(test_idcs) * length(trial_idcs)/(length(test_idcs) + length(trial_idcs))))

    am = allocate_aca_memory(
        Float64,
        length(test_idcs),
        length(trial_idcs),
        maxrank=maxrank
    )

    retU, retV, U, V, rowindices, colindices= nca(
        lm,
        am;
        rowpivstrat=compressor.rowpivstrat,
        columnpivstrat=compressor.columnpivstrat,
        tol=compressor.tol,
        svdrecompress=compressor.svdrecompress
    )

    return FastBEAST.Pivotlrb{Float64}(
        ClusterMatrix(U, V, rowindices, colindices),    
        test_idcs,
        trial_idcs,
        roworcolidcs,
        roworcolmaps,
        childrange
    )
end
