
#=
These are the necessary functions to create a table from a regression result.
If the regression result does not provide a function by default, then
within an extension, it is possible to define the necessary function.
=#

##############################################################################
##
## Custom covariance specifications
##
##############################################################################

"""
    VcovSpec{T}

Covariance specification that can be attached to a `RegressionModel`
using the `model + vcov(spec)` syntax to override the variance–covariance matrix used in tables.

The type parameter `T` determines how the covariance matrix is computed:
- `AbstractMatrix`: The matrix is used directly
- `Function`: The function is called to compute the matrix
- Other types: Dispatched to [`RegressionTables.materialize_vcov`](@ref) for extension by third-party packages
"""
struct VcovSpec{T}
    source::T
end

"""
    struct RegressionModelWithVcov{M,S} <: RegressionModel

Wraps a regression model together with a covariance specification. Standard errors, `vcov`, and statistics
derived from them now draw from the attached specification, while all other queries are delegated to `model`.
Construct these via `rr + vcov(spec)`.
"""
struct RegressionModelWithVcov{M<:RegressionModel,T} <: RegressionModel
    model::M
    spec::VcovSpec{T}
    cache::Base.RefValue{Union{Nothing, AbstractMatrix}}
    function RegressionModelWithVcov(model::M, spec::VcovSpec{T}) where {M<:RegressionModel,T}
        new{M,T}(model, spec, Ref{Union{Nothing, AbstractMatrix}}(nothing))
    end
end

"""
    _formula(x::RegressionModel)

Generally a passthrough for the `formula` function from the `StatsModels` package.
Note tha the `formula` function returns the `FormulaSchema`.

This function is only used internally in the [`RegressionTables._responsename`](@ref)
and [`RegressionTables._coefnames`](@ref) functions. Therefore, if the `RegressionModel`
uses those two functions without using `formula`, this function is not necessary.
"""
_formula(x::RegressionModel) = formula(x)
_formula(x::RegressionModelWithVcov) = _formula(x.model)

"""
    _responsename(x::RegressionModel)

Returns the name of the dependent variable in the regression model.
The default  is to return a parsed version of the left hand side of formula ([`RegressionTables._formula`](@ref)),
but if that is not available, then it will return the `StatsAPI.responsename` function.
"""
function _responsename(x::RegressionModel)
    try
        out = get_coefname(_formula(x).lhs)
    catch
        out = get_coefname(responsename(x))
    end
    if isa(out, AbstractVector)
        out = first(out)
    end
    out
end
_responsename(x::RegressionModelWithVcov) = _responsename(x.model)

"""
    _coefnames(x::RegressionModel)

Returns a vector of the names of the coefficients in the regression model.
The default  is to return a parsed version of the formula ([`RegressionTables._formula`](@ref)),
but if that is not available, then it will return the `StatsAPI.coefnames` function.
"""
function _coefnames(x::RegressionModel)
    try
        out = get_coefname(_formula(x).rhs)
    catch
        out = get_coefname(coefnames(x))
    end
    if !isa(out, AbstractVector)
        out = [out]
    end
    out
end
_coefnames(x::RegressionModelWithVcov) = _coefnames(x.model)

"""
    RegressionTables.vcov(spec)

Convert `spec` into a covariance specification that can be attached to a regression model.
Accepted inputs include matrices, functions (with zero or one argument), and any custom object
for which [`RegressionTables.materialize_vcov`](@ref) is defined. The result can be added to a
model via `model + vcov(spec)` so that the provided variance–covariance matrix is used when
displaying standard errors.
"""
vcov(x::VcovSpec) = x

##############################################################################
##
## TYPE PIRACY WARNING: The following method causes type piracy
## This extends StatsAPI.vcov for all types, which we don't own.
## TODO: Move this to CovarianceMatrices.jl or refactor the API
##
##############################################################################
vcov(spec) = VcovSpec(spec)

# Dispatch on VcovSpec based on source type
materialize_vcov(spec::VcovSpec{<:AbstractMatrix}, model) = spec.source

function materialize_vcov(spec::VcovSpec{<:Function}, model)
    f = spec.source
    if applicable(f, model)
        return f(model)
    elseif applicable(f)
        return f()
    else
        throw(ArgumentError("Provided covariance function does not accept zero or one argument."))
    end
end

# For external estimators, unwrap and dispatch on the estimator type itself
materialize_vcov(spec::VcovSpec{T}, model) where {T} = materialize_vcov(spec.source, model)

"""
    RegressionTables.materialize_vcov(estimator, model::RegressionModel)

Produce the variance–covariance matrix for `model` given an estimator object.
This is the extension point for third-party estimators: define a method that
returns the desired matrix and `RegressionTables` will cache and reuse it.

For example, to integrate with CovarianceMatrices.jl, define:
```julia
function RegressionTables.materialize_vcov(estimator::CovarianceMatrices.RobustVariance, model)
    return StatsAPI.vcov(estimator, model)
end
```
"""
function materialize_vcov(estimator, model)
    throw(ArgumentError("""
        No method to compute a covariance matrix for $(typeof(estimator)).

        If this is from CovarianceMatrices.jl, ensure the package is loaded with `using CovarianceMatrices`.
        Otherwise, define: `RegressionTables.materialize_vcov(::$(typeof(estimator)), model::RegressionModel)`
        """))
end

function _validate_vcov_dimensions(model, Σ)
    ncoef = length(_coef(model))
    m, n = size(Σ)

    # Check dimensions match number of coefficients
    if m != ncoef || n != ncoef
        throw(ArgumentError("Custom covariance matrix must be $(ncoef)×$(ncoef). Got size $(size(Σ))."))
    end

    # Check matrix is square (redundant but explicit)
    if m != n
        throw(ArgumentError("Covariance matrix must be square. Got size $(size(Σ))."))
    end

    # Check for symmetry (covariance matrices should be symmetric)
    if !issymmetric(Σ)
        @warn "Covariance matrix is not symmetric. This may indicate an error in computation."
    end
end

function _custom_vcov(rr::RegressionModelWithVcov)
    Σ = rr.cache[]
    if Σ === nothing
        Σ = materialize_vcov(rr.spec, rr.model)
        if !(Σ isa AbstractMatrix)
            throw(ArgumentError("Custom covariance specification must return an AbstractMatrix. Got $(typeof(Σ))."))
        end
        _validate_vcov_dimensions(rr.model, Σ)
        rr.cache[] = Σ
    end
    Σ
end

function _custom_stderror(rr::RegressionModelWithVcov)
    Σ = _custom_vcov(rr)
    sqrt.(map(i -> Σ[i, i], axes(Σ, 1)))
end

import Base: +

function +(rr::RegressionModel, spec::VcovSpec)
    RegressionModelWithVcov(rr, spec)
end

function +(spec::VcovSpec, rr::RegressionModel)
    rr + spec
end

function +(rr::RegressionModelWithVcov, spec::VcovSpec)
    RegressionModelWithVcov(rr.model, spec)
end

function +(spec::VcovSpec, rr::RegressionModelWithVcov)
    rr + spec
end

# delegate StatsAPI methods to the wrapped model, except for vcov/stderror which use the custom specification
coef(x::RegressionModelWithVcov) = coef(x.model)
stderror(x::RegressionModelWithVcov) = _custom_stderror(x)
dof_residual(x::RegressionModelWithVcov) = dof_residual(x.model)
responsename(x::RegressionModelWithVcov) = responsename(x.model)
coefnames(x::RegressionModelWithVcov) = coefnames(x.model)
islinear(x::RegressionModelWithVcov) = islinear(x.model)
nobs(x::RegressionModelWithVcov) = nobs(x.model)
vcov(x::RegressionModelWithVcov) = _custom_vcov(x)

"""
    _coef(x::RegressionModel)

Returns a vector of the coefficients in the regression model.
By default, is just a passthrough for the `coef` function from the `StatsModels` package.
"""
_coef(x::RegressionModel) = coef(x)
_coef(x::RegressionModelWithVcov) = _coef(x.model)

"""
    _stderror(x::RegressionModel)

Returns a vector of the standard errors of the coefficients in the regression model.
By default, is just a passthrough for the `stderror` function from the `StatsModels` package.
"""
_stderror(x::RegressionModel) = stderror(x)
_stderror(x::RegressionModelWithVcov) = _custom_stderror(x)

"""
    _dof_residual(x::RegressionModel)

Returns the degrees of freedom of the residuals in the regression model.
By default, is just a passthrough for the `dof_residual` function from the `StatsModels` package.
"""
_dof_residual(x::RegressionModel) = dof_residual(x)
_dof_residual(x::RegressionModelWithVcov) = _dof_residual(x.model)

"""
    _pvalue(x::RegressionModel)

Returns a vector of the p-values of the coefficients in the regression model.
"""
function _pvalue(x::RegressionModel)
    tt = _coef(x) ./ _stderror(x)
    ccdf.(Ref(FDist(1, _dof_residual(x))), abs2.(tt))
end

"""
    _islinear(x::RegressionModel)

Returns a boolean indicating whether the regression model is linear.
"""
_islinear(x::RegressionModel) = islinear(x)
_islinear(x::RegressionModelWithVcov) = _islinear(x.model)

"""
    can_standardize(x::RegressionModel)

Returns a boolean indicating whether the coefficients can be standardized.
standardized coefficients are coefficients that are scaled by the standard deviation of the
variables. This is useful for comparing the relative importance of the variables in the model.

This is only possible of the `RegressionModel` includes the model matrix or the
standard deviation of the dependent variable. If the `RegressionModel` does not include
either of these, then this function should return `false`.

See also [`RegressionTables.standardize_coef_values`](@ref).
"""
function can_standardize(x::T) where {T<:RegressionModel}
    @warn "standardize_coef is not possible for $T"
    false
end
can_standardize(x::RegressionModelWithVcov) = can_standardize(x.model)

"""
    standardize_coef_values(std_X, std_Y, val)

Standardizes the coefficients by the standard deviation of the variables.
This is useful for comparing the relative importance of the variables in the model.

This function is only used if the [`RegressionTables.can_standardize`](@ref) function returns `true`.

### Arguments
- `std_X::Real`: The standard deviation of the independent variable.
- `std_Y::Real`: The standard deviation of the dependent variable.
- `val::Real`: The value to be standardized (either the coefficient or the standard error).

!!! note
    If the standard deviation of the independent variable is 0, then the interpretation of the
    coefficient is how many standard deviations of the dependent variable away from 0 is the intercept.
    In this case, the function returns `val / std_Y`.

    Otherwise, the function returns `val * std_X / std_Y`.
"""
function standardize_coef_values(std_X, std_Y, val)
    if std_X == 0 # constant has 0 std, so the interpretation is how many Y std away from 0 is the intercept
        val / std_Y
    else
        val * std_X / std_Y
    end
end

transformer(s::Nothing, repl_dict::AbstractDict) = s
function transformer(s, repl_dict::AbstractDict)
    for (old, new) in repl_dict
        s = replace(s, old => new)
    end
    return s
end

replace_name(s::Union{AbstractString, AbstractCoefName}, exact_dict, repl_dict) = get(exact_dict, s, transformer(s, repl_dict))
replace_name(s::Tuple{<:AbstractCoefName, <:AbstractString}, exact_dict, repl_dict) = (replace_name(s[1], exact_dict, repl_dict), s[2])
replace_name(s::Nothing, args...) = s

RegressionType(x::RegressionModel) = _islinear(x) ? RegressionType(Normal()) : RegressionType("NL")
RegressionType(x::RegressionModelWithVcov) = RegressionType(x.model)

make_reg_stats(rr, stat::Type{<:AbstractRegressionStatistic}) = stat(rr)
make_reg_stats(rr, stat) = stat
make_reg_stats(rr, stat::Pair{<:Any, <:AbstractString}) = make_reg_stats(rr, first(stat)) => last(stat)

default_regression_statistics(x::AbstractRenderType, rr::RegressionModel) = default_regression_statistics(rr)
"""
    default_regression_statistics(rr::RegressionModel)

Returns a vector of [`AbstractRegressionStatistic`](@ref) objects. This is used to display the
statistics in the table. This is customizable for each `RegressionModel` type. The default
is to return a vector of `Nobs` and `R2`.
"""
default_regression_statistics(rr::RegressionModel) = [Nobs, R2]
default_regression_statistics(rr::RegressionModelWithVcov) = default_regression_statistics(rr.model)


"""
    other_stats(rr::RegressionModel, s::Symbol)

Returns any other statistics to be displayed. This is used (if the appropriate extension is loaded)
to display the fixed effects in a FixedEffectModel (or GLFixedEffectModel),
clusters in those two, or Random Effects in a MixedModel. For other regressions, this
returns `nothing`.
"""
other_stats(x::RegressionModel, s::Symbol) = nothing
other_stats(x::RegressionModelWithVcov, s::Symbol) = other_stats(x.model, s)
