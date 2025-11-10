module RegressionTables2CovarianceMatricesExt

using RegressionTables2
using CovarianceMatrices
using StatsAPI
using StatsBase

"""
    RegressionTables2.materialize_vcov(estimator::CovarianceMatrices.AVarEstimator, model)

Compute the variance-covariance matrix for a regression model using a CovarianceMatrices.jl estimator.

This method enables integration between RegressionTables2.jl and CovarianceMatrices.jl, allowing
users to specify robust variance estimators (HC0-HC5, HAC, cluster-robust, etc.) when creating
regression tables.

# Examples
```julia
using RegressionTables2, CovarianceMatrices, GLM, DataFrames

# Fit a regression model
df = DataFrame(y = randn(100), x1 = randn(100), x2 = randn(100))
model = lm(@formula(y ~ x1 + x2), df)

# Create table with HC3 robust standard errors
regtable(model + vcov(HC3()))

# Or with HAC standard errors
regtable(model + vcov(HAC(Bartlett, 5)))
```
"""
function RegressionTables2.materialize_vcov(
    estimator::CovarianceMatrices.AVarEstimator,
    model::StatsAPI.RegressionModel
)
    # CovarianceMatrices.jl v0.22+ uses a three-argument API: vcov(estimator, variance_form, model)
    # For RegressionTables, we use Misspecified() as the default variance form
    return StatsBase.vcov(estimator, model)
end

end
