module RegressionTables2MixedModelsExt


using MixedModels, RegressionTables2, StatsModels, Statistics, Distributions

RegressionTables2.default_regression_statistics(rr::MixedModel) = [Nobs, LogLikelihood]

function RegressionTables2.RegressionType(x::MixedModel)
    if islinear(x)
        RegressionType(Normal())
    else
        RegressionType(x.resp) # uses the GLM extension
    end
end

function RegressionTables2._coefnames(x::MixedModel)
    r = formula(x).rhs
    out = if isa(r, Tuple)
        RegressionTables2.get_coefname(r[1])
    else
        RegressionTables2.get_coefname(r)
    end
    if !isa(out, AbstractVector)
        out = [out]
    end
    out
end

# k is which coefficient or standard error to standardize
RegressionTables2.standardize_coef_values(x::MixedModel, val, k) =
    RegressionTables2.standardize_coef_values(std(modelmatrix(x)[:, k]), std(response(x)), val)

RegressionTables2.can_standardize(x::MixedModel) = true

function RegressionTables2.other_stats(x::MixedModel, s::Symbol)
    if s == :randomeffects
        f = formula(x)
        if length(f.rhs) == 1
            return Dict{Symbol, Vector{Pair}}()
        end
        out = RegressionTables2.RandomEffectCoefName[]
        vals = x.σs
        out_vals = Float64[]
        for re in f.rhs[2:end]
            rhs_sym = re.rhs |> Symbol
            rhs_name = RegressionTables2.CoefName(String(rhs_sym))
            lhs_sym = Symbol.(coefnames(re.lhs))
            lhs_names = RegressionTables2.get_coefname(re.lhs)
            if isa(lhs_sym, AbstractVector)
                for (ls, ln) in zip(lhs_sym, lhs_names)
                    val = vals[rhs_sym][ls]
                    push!(out, RegressionTables2.RandomEffectCoefName(rhs_name, ln))
                    push!(out_vals, val)
                end
            else# just one term
                push!(out, RegressionTables2.RandomEffectCoefName(rhs_name, lhs_names))
                push!(out_vals, vals[rhs_sym][lhs_sym])
            end
        end
        out .=> RegressionTables2.RandomEffectValue.(out_vals)
    else
        nothing
    end
end
end