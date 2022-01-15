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

    for (fmvindex, fmv) in enumerate(hmat.fullmatrixviews)
        for colindex in fmv.rightindices
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
    
    for (lmvindex, lmv) in enumerate(hmat.matrixviews)
        for colindex in lmv.rightindices
            if ncolstolmv == size(colstolmv)[1]
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
                    shmat.hmat.fullmatrixviews[fmvindex].rightindices
                )
                @views solution[shmat.hmat.fullmatrixviews[fmvindex].leftindices] += 
                    shmat.hmat.fullmatrixviews[fmvindex].matrix[:,colindex]*
                    nnzvalue
            end
        end 
    end

    for (nnzelement, nnzvalue) in zip(vector.nzind, vector.nzval)
        for j = 1:shmat.ncolstolmv[nnzelement]
            for lmvindex in shmat.colstolmv[j,nnzelement]
                colindex = findfirst(
                    isequal(nnzelement),
                    shmat.hmat.matrixviews[lmvindex].rightindices
                )
                @views solution[shmat.hmat.matrixviews[lmvindex].leftindices] += 
                    shmat.hmat.matrixviews[lmvindex].leftmatrix*
                    shmat.hmat.matrixviews[lmvindex].rightmatrix[:,colindex]*
                    nnzvalue
            end
        end 
    end

    return solution
end
