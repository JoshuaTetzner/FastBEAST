using PyCall
helmholtz = pyimport("exafmm.helmholtz")

function assemble_hfmm(
    spoints::Matrix{Float64},
    tpoints::Matrix{Float64};
    p=10,
    ncrit=200,
    wavek=10
)
    sources = helmholtz.init_sources(spoints, zeros(length(tpoints[:,1])))
    targets = helmholtz.init_targets(tpoints)

    fmm = helmholtz.HelmholtzFmm(p=p, ncrit=ncrit, wavek=wavek, filename="test_file.dat")

    tree = helmholtz.setup(sources, targets, fmm)
    eval(charges) = eval_hfmm(tree, fmm, charges)
    return eval
end

function eval_hfmm(
    tree,
    fmm,
    charges::Vector{ComplexF64}
)
    
    helmholtz.update_charges(tree, charges)
    helmholtz.clear_values(tree)               
    trg_values = helmholtz.evaluate(tree, fmm)

    return trg_values, fmm.verify(tree.leafs)
end