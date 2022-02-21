using SparseArrays

struct SparseHMatrix{T,I}
    hmat::HMatrix{T,I}
    colstofmv
    colstolmv
    ncolstofmv
    ncolstolmv
end

function colstomv(hmat::HMatrix)
    colstofmv = zeros(Int, 100, size(hmat)[2])
    colstolmv = zeros(Int, 100, size(hmat)[2])
    ncolstofmv = zeros(Int, size(hmat)[2])
    ncolstolmv = zeros(Int, size(hmat)[2])

    for (fmvindex, fmv) in enumerate(hmat.fullrankblocks)
        for colindex in fmv.σ  
            if ncolstofmv == size(colstofmv)[1]
                colstofmvnew = zeros(Int, size(colstofmv)[1]+100, size(colstofmv)[2])
                colstofmvnew[1:size(colstofmv)[1], :] = colstofmv
                colstofmv = colstofmvnew
                colstofmv[ncolstofmv[colindex]+1, colindex]=fmvindex
                ncolstofmv[colindex] += 1
            else
                colstofmv[ncolstofmv[colindex]+1, colindex]=fmvindex
                ncolstofmv[colindex] += 1
            end
        end
    end
    
    for (lmvindex, lmv) in enumerate(hmat.lowrankblocks)
        for colindex in lmv.σ
            if ncolstolmv[colindex] == size(colstolmv)[1]
                colstolmvnew = zeros(Int, size(colstolmv)[1]+100, size(colstolmv)[2])
                colstolmvnew[1:size(colstolmv)[1], :] = colstolmv
                colstolmv = colstolmvnew
                colstolmv[ncolstolmv[colindex]+1, colindex]=lmvindex
                ncolstolmv[colindex] += 1
            else
                colstolmv[ncolstolmv[colindex]+1, colindex]=lmvindex
                ncolstolmv[colindex] += 1
            end
        end
    end

    return SparseHMatrix(hmat, colstofmv, colstolmv, ncolstofmv, ncolstolmv)
end

function sparsevectormul(
    shmat::SparseHMatrix,
    vector::SparseVector
)
    solution = zeros(size(shmat.hmat)[1])
    for (nnzelement, nnzvalue) in zip(vector.nzind, vector.nzval)
        for j = 1:shmat.ncolstofmv[nnzelement]
            for fmvindex in shmat.colstofmv[j,nnzelement]
                colindex = findfirst(
                    isequal(nnzelement),
                    shmat.hmat.fullrankblocks[fmvindex].σ
                )
                @views solution[shmat.hmat.fullrankblocks[fmvindex].τ] += 
                    shmat.hmat.fullrankblocks[fmvindex].M[:,colindex]*
                    nnzvalue
            end
        end 
    end

    for (nnzelement, nnzvalue) in zip(vector.nzind, vector.nzval)
        for j = 1:shmat.ncolstolmv[nnzelement]
            for lmvindex in shmat.colstolmv[j,nnzelement]
                colindex = findfirst(
                    isequal(nnzelement),
                    shmat.hmat.lowrankblocks[lmvindex].σ
                )
                @views solution[shmat.hmat.lowrankblocks[lmvindex].τ] += 
                    shmat.hmat.lowrankblocks[lmvindex].M.U*
                    shmat.hmat.lowrankblocks[lmvindex].M.V[:,colindex]*
                    nnzvalue
            end
        end 
    end
    # über fullvec loopen und zwar nur über die einträge, die wir geändert hatten

    return solution
end

function sparsevectormul2(
    shmat::SparseHMatrix,
    vector::SparseVector
)
    solution = zeros(size(shmat.hmat)[1])
    usedfmv = zeros(Bool,length(shmat.hmat.fullrankblocks)) 
    for nzind in vector.nzind
        for fmvindex in shmat.colstofmv[1:shmat.ncolstofmv[nzind], nzind]
            if !usedfmv[fmvindex]
                usedfmv[fmvindex] = true
                @views solution[shmat.hmat.fullrankblocks[fmvindex].τ] +=
                    shmat.hmat.fullrankblocks[fmvindex].M*
                    vector[shmat.hmat.fullrankblocks[fmvindex].σ]
            end
        end
    end 

    usedlmv = zeros(Bool,length(shmat.hmat.lowrankblocks)) 
    for nzind in vector.nzind
        for mvindex in shmat.colstolmv[1:shmat.ncolstolmv[nzind], nzind]
            if !usedlmv[mvindex]
                usedlmv[mvindex] = true
                @views solution[shmat.hmat.lowrankblocks[mvindex].τ] +=
                    shmat.hmat.lowrankblocks[mvindex].M.U*
                    shmat.hmat.lowrankblocks[mvindex].M.V*
                    vector[shmat.hmat.lowrankblocks[mvindex].σ]
                    
            end
        end
    end 

    return solution
end

function hmatmatrixmul(
    hmat::HMatrix,
    smat::SparseMatrixCSC
)
    solution = zeros(size(hmat)[1],size(smat)[2])

    for frb in hmat.fullrankblocks
        solution[frb.τ, :] += frb.M * smat[frb.σ, :] 
    end

    for lrb in hmat.lowrankblocks
        solution[lrb.τ, :] += lrb.M.U * (lrb.M.V * smat[lrb.σ, :])
    end

    return solution
end