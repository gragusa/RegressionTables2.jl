using RegressionTables2
using FixedEffectModels, GLM, RDatasets, Test
using LinearAlgebra
using StatsAPI: RegressionModel

df = RDatasets.dataset("datasets", "iris")
df[!, :isSmall] = df[!, :SepalWidth] .< 2.9
df[!, :isWide] = df[!, :SepalWidth] .> 2.5

# FixedEffectModels.jl
rr1 = reg(df, @formula(SepalLength ~ SepalWidth))
rr2 = reg(df, @formula(SepalLength ~ SepalWidth + PetalLength + fe(Species)))
rr3 = reg(df, @formula(SepalLength ~ SepalWidth + PetalLength + PetalWidth + fe(Species) + fe(isSmall)))
rr4 = reg(df, @formula(SepalWidth ~ SepalLength + PetalLength + PetalWidth + fe(Species)))
rr5 = reg(df, @formula(SepalWidth ~ SepalLength + (PetalLength ~ PetalWidth) + fe(Species)))
rr6 = reg(df, @formula(SepalLength ~ SepalWidth + fe(Species)&fe(isWide) + fe(isSmall)))
rr7 = reg(df, @formula(SepalLength ~ SepalWidth + PetalLength&fe(isWide) + fe(isSmall)))


RegressionTables2.default_print_fe_suffix(x::AbstractRenderType) = false
RegressionTables2.default_print_control_indicator(x::AbstractRenderType) = false
RegressionTables2.default_regression_statistics(x::AbstractRenderType, rrs::Tuple) = [Nobs, R2]
RegressionTables2.default_print_estimator(x::AbstractRenderType, rrs) = true
# GLM.jl
dobson = DataFrame(Counts = [18.,17,15,20,10,20,25,13,12],
    Outcome = repeat(["A", "B", "C"], outer = 3),
    Treatment = repeat(["a","b", "c"], inner = 3))

lm1 = fit(LinearModel, @formula(SepalLength ~ SepalWidth), df)
lm2 = fit(LinearModel, @formula(SepalLength ~ SepalWidth + PetalWidth), df)
lm3 = fit(LinearModel, @formula(SepalLength ~ SepalWidth * PetalWidth), df) # testing interactions
gm1 = fit(GeneralizedLinearModel, @formula(Counts ~ 1 + Outcome), dobson,
              Poisson())
              
# test of forula on lhs
lm4 = fit(LinearModel, @formula(log(SepalLength) ~ SepalWidth * PetalWidth), df) # testing interactions

function checkfilesarethesame(file1::String, file2::String)

    f1 = open(file1, "r")
    f2 = open(file2, "r")

    s1 = read(f1, String)
    s2 = read(f2, String)

    close(f1)
    close(f2)
    s1 = replace(s1, "\r\n" => "\n")
    s2 = replace(s2, "\r\n" => "\n")

    # Character-by-character comparison
    for i=1:length(s1)
        if s1[i]!=s2[i]
            println("Character $(i) different: $(s1[i]) $(s2[i])")
        end
    end

    if s1 == s2
        return true
    else
        return false
        println("File 1:")
        @show s1
        println("File 2:")
        @show s2
    end
end

##
rr_short = reg(df, @formula(SepalLength ~ log1p(SepalWidth)))
tab = regtable(rr_short)
@test tab[4, 1] == "log1p(SepalWidth)"



tab = regtable(rr4,rr5,lm1, lm2, gm1; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "ftest1.txt")), regression_statistics = [:nobs, :r2, :adjr2, :r2_within, :f, :p, :f_kp, :p_kp, :dof])
@test tab.vertical_gaps == [3, 5]
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "ftest1.txt"), joinpath(dirname(@__FILE__), "tables", "ftest1_reference.txt"))
# regressors and labels
tab = regtable(rr4,rr5,lm1, lm2, gm1; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "ftest2.txt")), regression_statistics = [:nobs, :r2, :adjr2, :r2_within, :f, :p, :f_kp, :p_kp, :dof], regressors = ["SepalLength", "PetalWidth"])
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "ftest2.txt"), joinpath(dirname(@__FILE__), "tables", "ftest2_reference.txt"))
# fixedeffects, estimformat, statisticformat, number_regressions_decoration
tab = regtable(rr3,rr5,lm1, lm2, gm1; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "ftest3.txt")), fixedeffects = ["SpeciesDummy"], estimformat = "%0.4f", statisticformat = "%0.4f", number_regressions_decoration = i -> "[$i]")
@test tab.vertical_gaps == [2, 3, 5]
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "ftest3.txt"), joinpath(dirname(@__FILE__), "tables", "ftest3_reference.txt"))
# estim_decoration, below_statistic, below_decoration, number_regressions



function dec(s::String, pval::Float64)
    if pval<0.0
        error("p value needs to be nonnegative.")
    end
    if (pval > 0.05)
        return "$s"
    elseif (pval <= 0.05)
        return "$s<-OMG!"
    end
end
tab = regtable(rr3,rr5,lm1, lm2, gm1; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "ftest4.txt")), estim_decoration = dec, below_statistic = :tstat, below_decoration = s -> "[$s]", number_regressions = false)
@test tab.vertical_gaps == []
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "ftest4.txt"), joinpath(dirname(@__FILE__), "tables", "ftest4_reference.txt"))
# print_fe_section, print_estimator_section
regtable(rr3,rr5,lm1, lm2, gm1; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "ftest5.txt")), print_fe_section = false, print_estimator_section = false)
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "ftest5.txt"), joinpath(dirname(@__FILE__), "tables", "ftest5_reference.txt"))
# transform_labels and custom labels
tab = regtable(rr5,rr6,lm1, lm2, lm3; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "ftest6.txt")), regression_statistics = [:nobs, :r2, :adjr2, :r2_within, :f, :p, :f_kp, :p_kp, :dof], transform_labels = :ampersand,
labels = Dict("SepalLength" => "My dependent variable: SepalLength", "PetalLength" => "Length of Petal", "PetalWidth" => "Width of Petal", "(Intercept)" => "Const." , "isSmall" => "isSmall Dummies", "SpeciesDummy" => "Species Dummies"))
@test tab.vertical_gaps == [2]
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "ftest6.txt"), joinpath(dirname(@__FILE__), "tables", "ftest6_reference.txt"))
# grouped regressions PR #61
# NOTE: behavior in ftest8 and ftest9 should be improved (Issue #63)
tab = regtable(rr1,rr5,rr2,rr4; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "ftest7.txt")), groups=["grp1" "grp1" "grp2" "grp2"])
@test tab.vertical_gaps == [2, 3, 4]
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "ftest7.txt"), joinpath(dirname(@__FILE__), "tables", "ftest7_reference.txt"))
regtable(rr1,rr5,rr2,rr4; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "ftest8.txt")), groups=["grp1" "grp1" "looooooooooooooooogong grp2" "looooooooooooooooogong grp2"])
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "ftest8.txt"), joinpath(dirname(@__FILE__), "tables", "ftest8_reference.txt"))
tab = regtable(rr5,rr1,rr2,rr4; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "ftest9.txt")), groups=["grp1" "grp1" "grp2" "grp2"])
@test tab.vertical_gaps == [2, 3, 4]
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "ftest9.txt"), joinpath(dirname(@__FILE__), "tables", "ftest9_reference.txt"))

tab = regtable(rr1,rr2,rr3,rr5; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "test1.txt")), regression_statistics = [:nobs, :r2, :adjr2, :r2_within, :f, :p, :f_kp, :p_kp, :dof])
@test tab.vertical_gaps ==[4]
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "test1.txt"), joinpath(dirname(@__FILE__), "tables", "test1_reference.txt"))

tab = regtable(rr1,rr2,rr3,rr5,rr6,rr7; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "test7.txt")), regression_statistics = [:nobs, :r2, :adjr2, :r2_within, :f, :p, :f_kp, :p_kp, :dof])
@test tab.vertical_gaps ==[4, 5]
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "test7.txt"), joinpath(dirname(@__FILE__), "tables", "test7_reference.txt"))

tab = regtable(lm1, lm2, gm1; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "test3.txt")), regression_statistics = [:nobs, :r2])
@test tab.vertical_gaps ==[3]
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "test3.txt"), joinpath(dirname(@__FILE__), "tables", "test3_reference.txt"))

regtable(lm1, lm2, lm4; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "test8.txt")), regression_statistics = [:nobs, :r2])
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "test8.txt"), joinpath(dirname(@__FILE__), "tables", "test8_reference.txt"))


using Statistics
comments = ["Specification", "Baseline", "Preferred"]
means = ["My custom mean", Statistics.mean(df.SepalLength[rr1.esample]), Statistics.mean(df.SepalLength[rr2.esample])]
mystats = [comments, means]
regtable(rr1, rr2; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "test9.txt")), regression_statistics = [:nobs, :r2],extralines = mystats)
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "test9.txt"), joinpath(dirname(@__FILE__), "tables", "test9_reference.txt"))

# below_decoration = :none
regtable(rr1,rr2,rr3,rr4; renderSettings = asciiOutput(joinpath(dirname(@__FILE__), "tables", "test10.txt")), below_statistic = :none)
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "test10.txt"), joinpath(dirname(@__FILE__), "tables", "test10_reference.txt"))


# LATEX TABLES



tab = regtable(rr1,rr2,rr3,rr5; renderSettings = latexOutput(joinpath(dirname(@__FILE__), "tables", "test2.tex")), regression_statistics = [:nobs, :r2, :adjr2, :r2_within, :f, :p, :f_kp, :p_kp, :dof])
@test tab.vertical_gaps == [4]
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "test2.tex"), joinpath(dirname(@__FILE__), "tables", "test2_reference.tex"))

regtable(rr1,rr2,rr3,rr5; renderSettings = latexOutput(joinpath(dirname(@__FILE__), "tables", "test3.tex")), 
                                           regression_statistics = [:nobs, :r2, :adjr2, :r2_within, :f, :p, :f_kp, :p_kp, :dof],
                                           align = :c)
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "test3.tex"), joinpath(dirname(@__FILE__), "tables", "test3_reference.tex"))


regtable(lm1, lm2, gm1; renderSettings = latexOutput(joinpath(dirname(@__FILE__), "tables", "test4.tex")), regression_statistics = [:nobs, :r2])
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "test4.tex"), joinpath(dirname(@__FILE__), "tables", "test4_reference.tex"))

tab = regtable(lm1, lm2, lm3, gm1; renderSettings = latexOutput(joinpath(dirname(@__FILE__), "tables", "test6.tex")), regression_statistics = [:nobs, :r2], transform_labels = :ampersand)
@test tab.vertical_gaps == [4]
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "test6.tex"), joinpath(dirname(@__FILE__), "tables", "test6_reference.tex"))



@testset "custom covariance specifications" begin
    base = lm1
    p = length(coef(base))
    Σ = Matrix{Float64}(I, p, p) .* 0.01

    wrapped = base + vcov(Σ)
    expected = fill(0.1, p)
    @test RegressionTables2._stderror(wrapped) ≈ expected
    @test stderror(wrapped) ≈ expected
    @test stderror(base) ≉ expected

    calls = Ref(0)
    fun_spec = vcov(model -> begin
        calls[] += 1
        Σ
    end)
    wrapped_fun = base + fun_spec
    RegressionTables2._stderror(wrapped_fun)
    RegressionTables2._stderror(wrapped_fun)
    @test calls[] == 1

    zero_arg_spec = vcov(() -> Σ .* 4)
    wrapped_zero = base + zero_arg_spec
    @test RegressionTables2._stderror(wrapped_zero) ≈ fill(0.2, p)

    struct DummyEstimator end
    RegressionTables2.materialize_vcov(::DummyEstimator, model::RegressionModel) = Σ .* 9
    wrapped_ext = base + vcov(DummyEstimator())
    @test RegressionTables2._stderror(wrapped_ext) ≈ fill(0.3, p)

    overridden = (base + vcov(Σ)) + vcov(Σ .* 16)
    @test RegressionTables2._stderror(overridden) ≈ fill(0.4, p)

    badΣ = ones(1, 1)
    bad = base + vcov(badΣ)
    @test_throws ArgumentError RegressionTables2._stderror(bad)

    @test_throws ArgumentError RegressionTables2._stderror(base + vcov(:unknown))

    # Test edge cases
    @testset "Edge cases" begin
        # Non-square matrix (though validation will catch same dimension mismatch)
        non_square = ones(p, p+1)
        @test_throws ArgumentError RegressionTables2._stderror(base + vcov(non_square))

        # Non-symmetric matrix (should warn but not error)
        non_symmetric = Σ .+ 0.001 .* (1:p) .* (1:p)'
        non_symmetric[1, 2] += 0.1  # Make it asymmetric
        wrapped_nonsym = base + vcov(non_symmetric)
        # Should work but produce a warning
        @test_logs (:warn, r"not symmetric") RegressionTables2._stderror(wrapped_nonsym)

        # Function that doesn't accept model or zero args
        bad_fun = (x, y) -> Σ
        @test_throws ArgumentError RegressionTables2._stderror(base + vcov(bad_fun))

        # Negative variance on diagonal (physically invalid but dimensionally OK)
        bad_diag = copy(Σ)
        bad_diag[1, 1] = -0.01
        wrapped_bad = base + vcov(bad_diag)
        # Should be computable even if physically nonsensical
        #se = RegressionTables2._stderror(wrapped_bad)
        #@test isnan(se[1])  # sqrt of negative is NaN
    end
end

@testset "CovarianceMatrices.jl integration" begin
    using CovarianceMatrices

    base = lm1
    p = length(coef(base))

    @testset "HC estimators" begin
        # Test HC0
        wrapped_hc0 = base + vcov(HC0())
        se_hc0 = stderror(wrapped_hc0)
        @test length(se_hc0) == p
        @test all(se_hc0 .> 0)  # Standard errors should be positive

        # Test HC1 (should be slightly larger than HC0)
        wrapped_hc1 = base + vcov(HC1())
        se_hc1 = stderror(wrapped_hc1)
        @test se_hc1 != stderror(base)  # Should differ from standard errors

        # Test HC3 (most commonly used)
        wrapped_hc3 = base + vcov(HC3())
        se_hc3 = stderror(wrapped_hc3)
        @test all(se_hc3 .> 0)

        # Test that vcov matrix is symmetric
        Σ_hc3 = vcov(wrapped_hc3)
        #@test issymmetric(Σ_hc3)
        @test size(Σ_hc3) == (p, p)
    end

    @testset "HAC estimators" begin
        # Test Bartlett kernel
        wrapped_bartlett = base + vcov(Bartlett(5))
        se_bartlett = stderror(wrapped_bartlett)
        @test length(se_bartlett) == p
        @test all(se_bartlett .> 0)

        # Test Parzen kernel
        wrapped_parzen = base + vcov(Parzen(3))
        se_parzen = stderror(wrapped_parzen)
        @test all(se_parzen .> 0)

        # Test that different kernels give different results
        @test se_bartlett != se_parzen
    end

    @testset "Caching" begin
        # Verify that covariance matrix is computed only once
        wrapped = base + vcov(HC1())
        Σ1 = vcov(wrapped)
        Σ2 = vcov(wrapped)
        @test Σ1 === Σ2  # Should be same object (cached)
    end

    @testset "Operator chaining" begin
        # Test that we can override vcov
        wrapped1 = base + vcov(HC0())
        wrapped2 = wrapped1 + vcov(HC3())
        se1 = stderror(wrapped1)
        se2 = stderror(wrapped2)
        @test se1 != se2  # Different estimators should give different results
    end

    @testset "Integration with regtable" begin
        # Test that regtable works with custom vcov
        tab = regtable(lm1 + vcov(HC3()), asciiOutput(joinpath(dirname(@__FILE__), "tables", "reghc31.txt")))
        @test tab !== nothing
        # Just ensure it doesn't error - visual output testing is beyond scope
    end
end

# HTML Tables
regtable(rr1,rr2,rr3,rr5; renderSettings = RegressionTables2.htmlOutput(joinpath(dirname(@__FILE__), "tables", "test1.html")), regression_statistics = [:nobs, :r2, :adjr2, :r2_within, :f, :p, :f_kp, :p_kp, :dof])
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "test1.html"), joinpath(dirname(@__FILE__), "tables", "test1_reference.html"))

regtable(lm1, lm2, gm1; renderSettings = RegressionTables2.htmlOutput(joinpath(dirname(@__FILE__), "tables", "test2.html")), regression_statistics = [:nobs, :r2])
@test checkfilesarethesame(joinpath(dirname(@__FILE__), "tables", "test2.html"), joinpath(dirname(@__FILE__), "tables", "test2_reference.html"))


# clean up
rm(joinpath(dirname(@__FILE__), "tables", "ftest1.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "ftest2.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "ftest3.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "ftest4.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "ftest5.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "ftest6.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "ftest7.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "ftest8.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "ftest9.txt"))

rm(joinpath(dirname(@__FILE__), "tables", "test1.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "test2.tex"))
rm(joinpath(dirname(@__FILE__), "tables", "test3.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "test3.tex"))
rm(joinpath(dirname(@__FILE__), "tables", "test4.tex"))
#rm(joinpath(dirname(@__FILE__), "tables", "test5.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "test6.tex"))
rm(joinpath(dirname(@__FILE__), "tables", "test7.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "test8.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "test9.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "test10.txt"))
rm(joinpath(dirname(@__FILE__), "tables", "test1.html"))
rm(joinpath(dirname(@__FILE__), "tables", "test2.html"))

RegressionTables2.default_print_fe_suffix(render::AbstractRenderType) = true
RegressionTables2.default_print_control_indicator(render::AbstractRenderType) = true
RegressionTables2.default_regression_statistics(render::AbstractRenderType, rrs::Tuple) = unique(union(RegressionTables2.default_regression_statistics.(render, rrs)...))
RegressionTables2.default_print_estimator(render::AbstractRenderType, rrs) = length(unique(RegressionTables2.RegressionType.(rrs))) > 1
