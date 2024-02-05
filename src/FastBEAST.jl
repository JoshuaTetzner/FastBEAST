module FastBEAST
using LinearAlgebra
include("tree/tree.jl")

include("aca/aca_utils.jl")
include("aca/pivoting.jl")
include("aca/convergence.jl")
include("aca/aca.jl")
include("skeletons.jl")
include("hmatrix.jl")
include("utils.jl")
include("fmm.jl")
include("beast.jl")
include("fmm/operators/FMMoperator.jl")

include("tree/clustertrees/kmeans.jl")
include("h2matrix/h2matrix.jl")
include("h2matrix/fars.jl")
include("h2matrix/nears.jl")

export BoundingBox
export getboxframe
export getchildbox
export whichchildbox

export BoxTreeNode
export create_tree
export BoxTreeOptions
export ExaFMMOptions
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
export nnz
export compressionrate

export create_CT_tree
export value
export computeinteractions

export hassemble
export fmmassemble
end
