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

# schreibe umbrella function die sparsevectormul N mal aufruft und die fullvec = zeros(N) anlegt (einmal!)
# vergleichen mit umbrella fuction ohne fullvec unter verwendung von vector[shmat.hmat.fullmatrixviews[fmvindex].rightindices] 

function sparsevectormul(
    shmat::SparseHMatrix,
    vector::SparseVector
)
    solution = zeros(size(shmat.hmat)[1])
    #fullvec = Vector(vector) # Eher über loop um allozation zu vermeiden
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
    # über fullvec loopen und zwar nur über die einträge, die wir geändert hatten

    return solution
end

function sparsevectormul10(
    shmat::SparseHMatrix,
    vector::SparseVector
)
    solution = zeros(size(shmat.hmat)[1])
    usedfmv = zeros(Bool,length(shmat.hmat.fullmatrixviews)) 
    for nzind in vector.nzind
        for fmvindex in shmat.colstofmv[1:shmat.ncolstofmv[nzind], nzind]
            if !usedfmv[fmvindex]
                usedfmv[fmvindex] = true
                @views solution[shmat.hmat.fullmatrixviews[fmvindex].leftindices] +=
                    shmat.hmat.fullmatrixviews[fmvindex].matrix*
                    vector[shmat.hmat.fullmatrixviews[fmvindex].rightindices]
            end
        end
    end 

    usedlmv = zeros(Bool,length(shmat.hmat.matrixviews)) 
    for nzind in vector.nzind
        for mvindex in shmat.colstolmv[1:shmat.ncolstolmv[nzind], nzind]
            if !usedlmv[mvindex]
                usedlmv[mvindex] = true
                @views solution[shmat.hmat.matrixviews[mvindex].leftindices] +=
                    shmat.hmat.matrixviews[mvindex].leftmatrix*
                    shmat.hmat.matrixviews[mvindex].rightmatrix*
                    vector[shmat.hmat.matrixviews[mvindex].rightindices]
                    
            end
        end
    end 
    # über fullvec loopen und zwar nur über die einträge, die wir geändert hatten

    return solution
end

function sparsevectormul3(
    shmat::SparseHMatrix,
    vector::SparseVector,
)

    solution = zeros(size(shmat.hmat)[1])
    usedfmv = zeros(Bool,length(shmat.hmat.fullmatrixviews)) 
    for nzind in vector.nzind
        for fmvindex in shmat.colstofmv[1:shmat.ncolstofmv[nzind], nzind]
            if !usedfmv[fmvindex]
                usedfmv[fmvindex] = true
                @views solution[shmat.hmat.fullmatrixviews[fmvindex].leftindices] +=
                    shmat.hmat.fullmatrixviews[fmvindex].matrix*
                    vector[shmat.hmat.fullmatrixviews[fmvindex].rightindices]
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

#

function sparsevectormul2(
    shmat::SparseHMatrix,
    vector::SparseVector,
    fullvec
)
    for (nnzelement, nnzvalue) in zip(vector.nzind, vector.nzval)
        fullvec[nnzelement]=nnzvalue
    end

    solution = zeros(size(shmat.hmat)[1])
    usedfmv = zeros(Bool,length(shmat.hmat.fullmatrixviews)) 
    for nzind in vector.nzind
        for fmvindex in shmat.colstofmv[1:shmat.ncolstofmv[nzind], nzind]
            if !usedfmv[fmvindex]
                usedfmv[fmvindex] = true
                solution[shmat.hmat.fullmatrixviews[fmvindex].leftindices] +=
                    shmat.hmat.fullmatrixviews[fmvindex].matrix*
                    fullvec[shmat.hmat.fullmatrixviews[fmvindex].rightindices]
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
    # über fullvec loopen und zwar nur über die einträge, die wir geändert hatten
    for (nnzelement, nnzvalue) in zip(vector.nzind, vector.nzval)
        fullvec[nnzelement]=0
    end
    
    return solution
end
