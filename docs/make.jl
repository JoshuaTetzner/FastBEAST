using FastBEAST
using Documenter
makedocs(
         sitename = "FastBEAST.jl",
         modules  = [FastBEAST],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/JoshuaTetzner/FastBEAST",
)