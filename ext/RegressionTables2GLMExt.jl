module RegressionTables2GLMExt

using GLM, RegressionTables2, StatsModels, Statistics

RegressionTables2.default_regression_statistics(rr::LinearModel) = [Nobs, R2]
RegressionTables2.default_regression_statistics(rr::StatsModels.TableRegressionModel{T}) where {T<:GLM.AbstractGLM} = [Nobs, R2McFadden]

RegressionTables2.RegressionType(x::StatsModels.TableRegressionModel{T}) where {T<:GLM.AbstractGLM} = RegressionType(x.model)
RegressionTables2.RegressionType(x::StatsModels.TableRegressionModel{T}) where {T<:LinearModel} = RegressionType(x.model)

# k is which coefficient or standard error to standardize
RegressionTables2.standardize_coef_values(x::StatsModels.TableRegressionModel, val, k) =
    RegressionTables2.standardize_coef_values(std(modelmatrix(x)[:, k]), std(response(x)), val)

RegressionTables2.can_standardize(x::StatsModels.TableRegressionModel) = true

RegressionTables2.RegressionType(x::LinearModel) = RegressionType(Normal())
RegressionTables2.RegressionType(x::GLM.LmResp) = RegressionType(Normal())
RegressionTables2.RegressionType(x::GeneralizedLinearModel) = RegressionType(x.rr)
RegressionTables2.RegressionType(x::GLM.GlmResp{Y, D, L}) where {Y, D, L} = RegressionType(D)


end