using BEAST
using SparseArrays
using LinearAlgebra

# mapping mesh to quadrature points
function meshtopoints(X0::BEAST.LagrangeBasis, quadorder)

    test_elements, _, _ = assemblydata(X0)
    tshapes = refspace(X0)

    test_eval(x) = tshapes(x, Val{:withcurl})
    qp = quadpoints(test_eval,  test_elements, (quadorder,));
    points = zeros(Float64, length(qp) * length(qp[1,1]), 3)
    ind = 1
    for el in qp
        for i in el
            points[ind, 1] = i.point.cart[1]
            points[ind, 2] = i.point.cart[2]
            points[ind, 3] = i.point.cart[3]
            ind += 1
        end
    end
    
    return points, qp
end

# construction of B matrix, if Ax = b and A = transpose(B)GB
function getBmatrix(qp::Matrix, X0::BEAST.LagrangeBasis)
    rfspace = refspace(X0)
    _, tad, _ = assemblydata(X0)
    len = length(qp) * length(qp[1,1]) * length(tad.data[1,:,1])
    rows = zeros(Int, len)
    cols = zeros(Int, len)
    vals = zeros(Float64, len)
    sind = 1
    for nf in eachindex(X0.geo.faces)
        ind = (nf - 1) * length(qp[1,1])
        for (np, point) in enumerate(qp[1, nf])
            val = rfspace(point.point)
            i = 1
            for (ifd, fd) in tad.data[1,:,nf]
                if ifd != 0
                    rows[sind] = ind + np
                    cols[sind] = ifd
                    vals[sind] = val[i].value * point.weight
                    sind += 1
                    i += 1
                end 
            end 
        end
    end

    return sparse(rows, cols, vals)
end
