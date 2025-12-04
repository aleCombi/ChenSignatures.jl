using Documenter
using ChenSignatures

makedocs(;
    modules=[ChenSignatures],
    authors="Alessandro Combi <alecombi94@gmail.com>",
    repo="https://github.com/aleCombi/ChenSignatures.jl/blob/{commit}{path}#{line}",
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
        "Contributing" => "contributing.md",
    ],
    checkdocs=:exports,
    warnonly=true,
)

deploydocs(;
    repo="github.com/aleCombi/ChenSignatures.jl",
    devbranch="master",
    push_preview=false,
)
