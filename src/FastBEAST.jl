module FastBEAST
using LinearAlgebra
include("tree/tree.jl")

include("aca.jl")
include("skeletons.jl")
include("hmatrix.jl")
include("utils.jl")
include("beast.jl")
include("sparsevectormul.jl")

export BoundingBox
export getboxframe
export getchildbox
export whichchildbox

export BoxTreeNode
export create_tree
export BoxTreeOptions
export KMeansTreeOptions
export KMeansTreeNode

export aca, allocate_aca_memory
export LazyMatrix

export MatrixBlock, LowRankMatrix

export HMatrix
export buildhmatrix
export adjoint
export estimate_norm
export estimate_reldifference
export getfullmatrix
export nnz
export compressionrate

export hassemble

export sparsevectormul
export colstomv
export sparsevectormul2
export sparsevectormul3
export colstomv2
end
