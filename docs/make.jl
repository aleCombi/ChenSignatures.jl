using Documenter
using ChenSignatures

makedocs(;
    modules=[ChenSignatures],
    authors="Alessandro Combi <alecombi94@gmail.com>",
    repo=Remotes.GitHub("aleCombi", "ChenSignatures.jl"),
    sitename="ChenSignatures.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://aleCombi.github.io/ChenSignatures.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
        "How it works" => "algorithm.md",
        "Contributing" => "contributing.md",
    ],
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/aleCombi/ChenSignatures.jl",
    devbranch="master",
    push_preview=false,
)
