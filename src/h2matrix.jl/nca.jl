using LinearAlgebra

function H2Matrix(
    matrixassembler::Function,
    testtree::T,
    sourcetree::T,
    ::Type{I},
    ::Type{K};
    farmatrixassembler=matrixassembler,
    compressor=ACAOptions(),
    multithreading=false,
    verbose=false
) where {I, K, T <: AbstractNode}
    
    

end
