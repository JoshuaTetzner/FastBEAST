function estimate_norm(mat; tol=1e-4, itmax = 1000)
    v = rand(size(mat,2))

    v = v/norm(v)
    itermin = 3
    i = 1
    σold = 1
    σnew = 1
    while (norm(sqrt(σold)-sqrt(σnew))/norm(sqrt(σold)) > tol || i < itermin) && i < itmax
        σold = σnew
        w = mat*v
        x = adjoint(mat)*w
        σnew = norm(x)
        v = x/norm(x)
        i += 1
    end
    return sqrt(σnew)
end

function estimate_reldifference(hmat, refmat; tol=1e-4)
    #if size(hmat) != size(refmat)
    #    error("Dimensions of matrices do not match")
    #end
    
    v = rand(size(hmat,2))

    v = v/norm(v)
    itermin = 3
    i = 1
    σold = 1
    σnew = 1
    while norm(sqrt(σold)-sqrt(σnew))/norm(sqrt(σold)) > tol || i < itermin
        σold = σnew
        w = hmat*v - refmat*v
        x = adjoint(hmat)*w - adjoint(refmat)*w
        σnew = norm(x)
        v = x/norm(x)
        i += 1
    end

    norm_refmat = estimate_norm(refmat, tol=tol)

    return sqrt(σnew)/norm_refmat
end

function getfullmatrix(hmat, Ns,Nt)
    fullhmat = zeros(Nt,Ns)
    for full in hmat.fullmatrixviews
        for i = 1:length(full.rightindices)
            for j = 1:length(full.leftindices)
                fullhmat[full.leftindices[j], full.rightindices[i]] = full.matrix[j,i]
            end
        end
    end
    for comp in hmat.matrixviews
        fullmat = comp.leftmatrix*comp.rightmatrix
        for i = 1:length(comp.leftindices)
            for j = 1:length(comp.rightindices)
                fullhmat[comp.leftindices[i],comp.rightindices[j]] = fullmat[i,j]
            end
        end
    end
    return fullhmat
end