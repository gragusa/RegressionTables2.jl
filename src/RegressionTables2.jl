module RegressionTables2

    ##############################################################################
    ##
    #   TODO:
    #
    #   FUNCTIONALITY: (asterisk means priority)
    #   - write more serious tests
    #
    #   TECHNICAL:
    #   - Formatting option: string (or function) for spacer rows
    #
    ##
    ##############################################################################


    ##############################################################################
    ##
    ## Dependencies
    ##
    ##############################################################################

    #using DataFrames

    using StatsBase
    using StatsModels
    using Statistics
    using StatsAPI
    import StatsAPI: coef, stderror, dof_residual, responsename, coefnames, islinear, nobs, vcov

    using Distributions
    using Format
    using LinearAlgebra: issymmetric
    using PrettyTables
    
    ##############################################################################
    ##
    ## Exported methods and types
    ##
    ##############################################################################

    export regtable, RegressionTable
    export Nobs, R2, R2McFadden, R2CoxSnell, R2Nagelkerke,
    R2Deviance, AdjR2, AdjR2McFadden, AdjR2Deviance, DOF, LogLikelihood, AIC, BIC, AICC,
    FStat, FStatPValue, FStatIV, FStatIVPValue, R2Within, PseudoR2, AdjPseudoR2
    export TStat, StdError, ConfInt, RegressionType

    # Customization functions
    export add_hline!, remove_hline!, set_alignment!, add_formatter!, set_backend!, merge_kwargs!

    export make_estim_decorator
    export vcov


    ##############################################################################
    ##
    ## Load files
    ##
    ##############################################################################

    # compatibility layer for old rendering system
    include("compat/render_compat.jl")

    # main types
    include("RegressionStatistics.jl")
    include("coefnames.jl")
    include("regressionResults.jl")

    # main settings
    include("decorations/default_decorations.jl")
    include("label_transforms/default_transforms.jl")

    # table structure (PrettyTables-based)
    include("regressiontable.jl")

    # main functions
    include("regtable.jl")

end
