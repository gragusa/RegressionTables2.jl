# Render type system for table construction
# This allows flexible table construction using different render types before converting to PrettyTables

"""
    abstract type AbstractRenderType end

Compatibility type for old rendering system.
Used internally during table construction, then converted to PrettyTables format.
"""
abstract type AbstractRenderType end

Base.broadcastable(o::AbstractRenderType) = Ref(o)

# Minimal render type implementations for compatibility
abstract type AbstractAscii <: AbstractRenderType end
abstract type AbstractLatex <: AbstractRenderType end
abstract type AbstractHtml <: AbstractRenderType end

struct AsciiTable <: AbstractAscii end
struct LatexTable <: AbstractLatex end
struct LatexTableStar <: AbstractLatex end
struct HtmlTable <: AbstractHtml end

# DataRow compatibility type
"""
    mutable struct DataRow{T<:AbstractRenderType}
        data::Vector
        align::String
        print_underlines::Vector{Bool}
    end

Compatibility type for old DataRow system.
Used internally during table construction.
"""
mutable struct DataRow{T<:AbstractRenderType}
    data::Vector
    align::String
    print_underlines::Vector{Bool}
    render::T

    function DataRow(
        data::Vector,
        align,
        print_underlines,
        render::T;
        kwargs...
    ) where {T<:AbstractRenderType}
        new{T}(data, align, print_underlines, render)
    end
end

DataRow(x::DataRow) = x
(::Type{T})(x::DataRow) where {T<:AbstractRenderType} = DataRow(x.data, x.align, x.print_underlines, T())

function DataRow(
    data::Vector;
    align="l" * "r" ^ (length(data) - 1),
    print_underlines=zeros(Bool, length(data)),
    render::AbstractRenderType=AsciiTable(),
    combine_equals=false,
    colwidths=nothing
)
    if isa(print_underlines, Bool)
        print_underlines = fill(print_underlines, length(data))
    end

    if combine_equals
        # Combine equal consecutive values
        combined_data = []
        combined_align = ""
        combined_underlines = Bool[]

        i = 1
        while i <= length(data)
            val = data[i]
            if val == ""
                push!(combined_data, val)
                combined_align *= align[i]
                push!(combined_underlines, print_underlines[i])
                i += 1
            else
                # Find consecutive equal values
                j = i
                while j < length(data) && data[j+1] == val && val != ""
                    j += 1
                end

                if j > i
                    # Multiple equal values - create pair
                    push!(combined_data, val => (i:j))
                    combined_align *= align[i]
                    push!(combined_underlines, print_underlines[i])
                else
                    push!(combined_data, val)
                    combined_align *= align[i]
                    push!(combined_underlines, print_underlines[i])
                end
                i = j + 1
            end
        end

        DataRow(combined_data, combined_align, combined_underlines, render)
    else
        DataRow(data, align, print_underlines, render)
    end
end

Base.length(x::DataRow) = length(x.data)
Base.size(x::DataRow) = (length(x.data),)

# Rendering functions - minimal implementations
label_p(render::AbstractRenderType) = "p"
label_p(render::AbstractLatex) = "\$p\$"
label_p(render::AbstractHtml) = "<i>p</i>"

wrapper(render::AbstractRenderType, s) = s
interaction_combine(render::AbstractRenderType) = " & "
interaction_combine(render::AbstractLatex) = " \$\\times\$ "
interaction_combine(render::AbstractHtml) = " &times; "

extra_cell_space(::AbstractRenderType) = 0
categorical_equal(render::AbstractRenderType) = ":"
random_effect_separator(render::AbstractRenderType) = " | "

label_ols(render::AbstractRenderType) = "OLS"
label_iv(render::AbstractRenderType) = "IV"

label_distribution(render::AbstractRenderType, d::D) where {D <: UnivariateDistribution} = string(Base.typename(D).wrapper)
label_distribution(render::AbstractRenderType, d::NegativeBinomial) = "Negative Binomial"
label_distribution(render::AbstractRenderType, d::InverseGaussian) = "Inverse Gaussian"

below_decoration(render::AbstractRenderType, s) = "($s)"
number_regressions_decoration(render::AbstractRenderType, s) = "($s)"
fe_suffix(render::AbstractRenderType) = " Fixed Effects"
cluster_suffix(render::AbstractRenderType) = " Clustering"
fe_value(render::AbstractRenderType, v) = v ? "Yes" : ""

# Base.repr implementations for compatibility
Base.repr(render::AbstractRenderType, x; args...) = "$x"
Base.repr(render::AbstractRenderType, x::Pair; args...) = repr(render, first(x); args...)
Base.repr(render::AbstractRenderType, x::Int; args...) = format(x, commas=true)

function Base.repr(render::AbstractRenderType, x::Float64; digits=3, commas=true, str_format=nothing, args...)
    if str_format !== nothing
        cfmt(str_format, x)
    else
        format(x; precision=digits, commas)
    end
end

Base.repr(render::AbstractRenderType, x::Nothing; args...) = ""
Base.repr(render::AbstractRenderType, x::Missing; args...) = ""
Base.repr(render::AbstractRenderType, x::AbstractString; args...) = String(x)
Base.repr(render::AbstractRenderType, x::Bool; args...) = x ? "Yes" : ""

# LaTeX-specific repr for multicolumn
function Base.repr(render::AbstractLatex, val::Pair; align="c", args...)
    s = repr(render, first(val); args...)
    if length(s) == 0 && length(last(val)) == 1
        s
    else
        "\\multicolumn{$(length(last(val)))}{$align}{$s}"
    end
end

# HTML-specific repr for colspan
function Base.repr(render::AbstractHtml, val::Pair; align="c", args...)
    s = repr(render, first(val); args...)
    if length(s) == 0
        s
    else
        # Return special marker for colspan
        s
    end
end

# Default functions
colsep(::AbstractRenderType) = "   "
colsep(::AbstractLatex) = " & "
colsep(::AbstractHtml) = ""

tablestart(::AbstractRenderType) = ""
tableend(::AbstractRenderType) = ""
toprule(::AbstractRenderType) = ""
midrule(::AbstractRenderType) = ""
bottomrule(::AbstractRenderType) = ""
linestart(::AbstractRenderType) = ""
lineend(::AbstractRenderType) = ""

default_align(render::AbstractRenderType) = :r
default_header_align(render::AbstractRenderType) = :c

# Additional repr methods for regression statistics and coefficient names
# These are placeholders - full implementations are in the included files

# For regression statistics
function Base.repr(render::AbstractRenderType, x::AbstractRegressionStatistic; digits=3, args...)
    repr(render, value(x); digits, args...)
end

function Base.repr(render::AbstractRenderType, x::AbstractR2; digits=3, args...)
    repr(render, value(x); digits, args...)
end

function Base.repr(render::AbstractRenderType, x::AbstractUnderStatistic; digits=3, args...)
    below_decoration(render, repr(render, value(x); digits, commas=false, args...))
end

function Base.repr(render::AbstractRenderType, x::ConfInt; digits=3, args...)
    below_decoration(render, repr(render, value(x)[1]; digits) * ", " * repr(render, value(x)[2]; digits))
end

function Base.repr(render::AbstractRenderType, x::CoefValue; digits=3, args...)
    estim_decorator(render, repr(render, value(x); digits, commas=false, args...), x.pvalue)
end

function Base.repr(render::AbstractRenderType, x::Type{V}; args...) where {V <: AbstractRegressionStatistic}
    label(render, V)
end

function Base.repr(render::AbstractRenderType, x::Type{RegressionType}; args...)
    label(render, x)
end

function Base.repr(render::AbstractRenderType, x::Tuple; args...)
    join(repr.(render, x; args...), " ")
end

# For coefficient names
function Base.repr(render::AbstractRenderType, x::AbstractCoefName; args...)
    repr(render, value(x); args...)
end

function Base.repr(render::AbstractRenderType, x::FixedEffectCoefName; args...)
    repr(render, value(x); args...) * fe_suffix(render)
end

function Base.repr(render::AbstractRenderType, x::ClusterCoefName; args...)
    repr(render, value(x); args...) * cluster_suffix(render)
end

function Base.repr(render::AbstractRenderType, x::InteractedCoefName; args...)
    join(repr.(render, value(x); args...), interaction_combine(render))
end

function Base.repr(render::AbstractRenderType, x::CategoricalCoefName; args...)
    "$(value(x))$(categorical_equal(render)) $(x.level)"
end

function Base.repr(render::AbstractRenderType, x::InterceptCoefName; args...)
    "(Intercept)"
end

function Base.repr(render::AbstractRenderType, x::HasControls; args...)
    repr(render, value(x); args...)
end

function Base.repr(render::AbstractRenderType, x::RegressionNumbers; args...)
    number_regressions_decoration(render, repr(render, value(x); args...))
end

function Base.repr(render::AbstractRenderType, x::Type{V}; args...) where {V <: HasControls}
    label(render, V)
end

function Base.repr(render::AbstractRenderType, x::RegressionType; args...)
    x.is_iv ? label_iv(render) : repr(render, value(x); args...)
end

function Base.repr(render::AbstractRenderType, x::D; args...) where {D <: UnivariateDistribution}
    string(Base.typename(D).wrapper)
end

function Base.repr(render::AbstractRenderType, x::InverseGaussian; args...)
    "Inverse Gaussian"
end

function Base.repr(render::AbstractRenderType, x::NegativeBinomial; args...)
    "Negative Binomial"
end

function Base.repr(render::AbstractRenderType, x::Normal; args...)
    label_ols(render)
end

function Base.repr(render::AbstractRenderType, x::RandomEffectCoefName; args...)
    repr(render, x.rhs; args...) * random_effect_separator(render) * repr(render, x.lhs; args...)
end

function Base.repr(render::AbstractRenderType, x::FixedEffectValue; args...)
    fe_value(render, value(x))
end

function Base.repr(render::AbstractRenderType, x::ClusterValue; args...)
    repr(render, value(x) > 0; args...)
end

function Base.repr(render::AbstractRenderType, x::RandomEffectValue; args...)
    repr(render, value(x); args...)
end

# LaTeX-specific labels
label(::AbstractLatex, x::Type{Nobs}) = "\$N\$"
label(::AbstractLatex, x::Type{R2}) = "\$R^2\$"
label(::AbstractLatex, x::Type{FStat}) = "\$F\$"
