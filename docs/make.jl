using Documenter, FastBEAST

makedocs(
    sitename = "FastBEAST.jl",
    modules = [FastBEAST],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "gstarted.md",
        "Manual" => Any[
            "man/clustering.md",
            "man/hmatrix.md",
            "man/aca.md"
        ],
        "Types and Functions" => "functions.md"
])

deploydocs(
    repo = "github.com/JoshuaTetzner/FastBEAST.git",
    devbranch = "bugfix/documantation",
    versions = nothing,
)