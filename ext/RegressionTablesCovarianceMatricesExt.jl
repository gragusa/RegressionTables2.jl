module RegressionTablesCovarianceMatricesExt

using RegressionTables
using CovarianceMatrices
using StatsAPI

"""
    RegressionTables.materialize_vcov(estimator::CovarianceMatrices.AVarEstimator, model)

Compute the variance-covariance matrix for a regression model using a CovarianceMatrices.jl estimator.

This method enables integration between RegressionTables.jl and CovarianceMatrices.jl, allowing
users to specify robust variance estimators (HC0-HC5, HAC, cluster-robust, etc.) when creating
regression tables.

# Examples
```julia
using RegressionTables, CovarianceMatrices, GLM, DataFrames

# Fit a regression model
df = DataFrame(y = randn(100), x1 = randn(100), x2 = randn(100))
model = lm(@formula(y ~ x1 + x2), df)

# Create table with HC3 robust standard errors
regtable(model + vcov(HC3()))

# Or with HAC standard errors
regtable(model + vcov(HAC(Bartlett, 5)))
```
"""
function RegressionTables.materialize_vcov(
    estimator::CovarianceMatrices.AVarEstimator,
    model::StatsAPI.RegressionModel
)
    return StatsAPI.vcov(estimator, model)
end

end
