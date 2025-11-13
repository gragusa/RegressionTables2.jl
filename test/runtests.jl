using RDatasets
using RegressionTables2
using FixedEffectModels, GLM, Documenter, Aqua
using Test

##

#=
ambiguities is tested separately since it defaults to recursive=true
but there are packages that have ambiguities that will cause the test
to fail

piracies is disabled because vcov(spec) causes type piracy - this is
intentional and will be moved to CovarianceMatrices.jl in the future
=#
Aqua.test_ambiguities(RegressionTables2; recursive=false)
Aqua.test_all(RegressionTables2; ambiguities=false, piracies=false)

tests = [
        "default_changes.jl",
        "RegressionTables.jl",
        "decorations.jl",
        "label_transforms.jl"
    ]

for test in tests
    @testset "$test" begin
        include(test)
    end
end

DocMeta.setdocmeta!(
    RegressionTables,
    :DocTestSetup,
    quote
        using RegressionTables2
    end;
    recursive=true
)

@testset "Regression Tables Documentation" begin
    doctest(RegressionTables2)
end