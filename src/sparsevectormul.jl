function sparsevectormul(
    hmat::HMatrix,
    vector::Vector
)
    solution = zeros(size(hmat)[1])

    function nonzeroelements(vector::Vector)
        nnzelements = Int64[]
        nnzvalues = Float64[]
        for i = 1:length(vector)
            if vector[i] != 0
                push!(nnzelements, i)
                push!(nnzvalues, vector[i]) 
            end
        end

        return nnzelements, nnzvalues
    end

    function hmatvalues(hmat::HMatrix, nnzelements, nnzvalues, solution)
        for fullmatrixview in hmat.fullmatrixviews
            for i = 1:length(nnzelements)
                index = findfirst(fullmatrixview.rightindices.==nnzelements[i])
                if index != nothing
                    for j = 1:length(fullmatrixview.leftindices)
                        solution[fullmatrixview.leftindices[j]] +=
                            fullmatrixview.matrix[j,index]*nnzvalues[i]
                    end
                end
            end
        end

        for matrixview in hmat.matrixviews
            for i = 1:length(nnzelements)
                index = findfirst(matrixview.rightindices.==nnzelements[i])
                if index != nothing
                    column = matrixview.leftmatrix*matrixview.rightmatrix[:,index]
                    for j = 1:length(matrixview.leftindices)
                        solution[matrixview.leftindices[j]] += column[j]*nnzvalues[i]
                    end
                end
            end
        end

        return solution
    end

    nnzelements, nnzvalues = nonzeroelements(vector)
    solution = hmatvalues(hmat, nnzelements, nnzvalues, solution)   
    return solution
end

function sparsevectormul2(
    hmat::HMatrix,
    vector::Vector
)
    solution = zeros(size(hmat)[1])

    function nonzeroelements(vector::Vector)
        nnzelements = Int64[]
        nnzvalues = Float64[]
        for i = 1:length(vector)
            if vector[i] != 0
                push!(nnzelements, i)
                push!(nnzvalues, vector[i]) 
            end
        end

        return nnzelements, nnzvalues
    end

    function hmatvalues(hmat::HMatrix, nnzelements, nnzvalues, solution)
        for fullmatrixview in hmat.fullmatrixviews
            for i = 1:length(nnzelements)
                index = findfirst(fullmatrixview.rightindices.==nnzelements[i])
                if index != nothing
                    for j = 1:length(fullmatrixview.leftindices)
                        solution[fullmatrixview.leftindices[j]] +=
                            fullmatrixview.matrix[j,index]*nnzvalues[i]
                    end
                end
            end
        end

        for matrixview in hmat.matrixviews
            for i = 1:length(nnzelements)
                indices = findall(matrixview.rightindices.==nnzelements[i])
                if indices != nothing
                    column = sum(matrixview.leftmatrix*matrixview.rightmatrix[:,indices], dims=2)
                    for j = 1:length(matrixview.leftindices)
                        solution[matrixview.leftindices[j]] += column[j]*nnzvalues[i]
                    end
                end
            end
        end

        return solution
    end

    nnzelements, nnzvalues = nonzeroelements(vector)
    solution = hmatvalues(hmat, nnzelements, nnzvalues, solution)   
    return solution
end