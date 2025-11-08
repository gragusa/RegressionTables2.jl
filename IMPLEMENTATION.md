# Custom Covariance Specifications in RegressionTables.jl

## Problem Statement

RegressionTables.jl traditionally relied on whatever variance–covariance matrix a regression model exposed through
`StatsAPI.vcov(model)` (or the derived `stderror`). Users frequently need to display alternative standard errors—for
example heteroskedasticity‑robust estimators or clustered covariances produced by CovarianceMatrices.jl. Because the
package only accepted bare `RegressionModel`s, there was no ergonomic way to inject a custom variance estimator without
mutating the model itself. The goal is to let users choose the covariance estimator per model/per column while keeping
the existing API backwards compatible and allowing third‑party packages to plug in without tight coupling.

## Solution Overview

1. **Covariance specification objects**: Introduce `AbstractVcovSpec` along with concrete helpers for matrices and
   callables. A helper function `RegressionTables.vcov(spec)` turns matrices, functions, or custom estimator objects into
   `AbstractVcovSpec`s.
2. **Model wrapper**: Define `RegressionModelWithVcov` that wraps any `RegressionModel` plus a spec. Users create these
   by writing `model + vcov(spec)`. The wrapper still subtypes `RegressionModel`, so it flows through the rest of the
   package unchanged.
3. **Extension point**: Provide `RegressionTables.materialize_vcov(spec, model)` for external packages (e.g.
   CovarianceMatrices.jl) to return the actual matrix on demand. Matrices and functions already have default methods.
4. **Caching and delegation**: `_custom_vcov` caches the realized matrix inside the wrapper to avoid recomputation.
   Standard errors, `vcov`, and dependent quantities pull from this cache; everything else delegates to the original
   model.
5. **Documentation and tests**: README/docs now explain the API, and tests cover matrices, callables, overriding specs,
   and failure paths.

## Implementation Notes

- **File layout**: All logic lives in `src/regressionResults.jl`, alongside other helpers that adapt `RegressionModel`s.
- **Imports/exports**: `StatsAPI.vcov` is imported so the wrapper can override it. The package now exports `vcov` so users
  call `RegressionTables.vcov`.
- **Wrapper behavior**:
  - `RegressionModelWithVcov` stores the original model, the spec, and a `Ref` cache.
  - `_custom_vcov` materializes the matrix via `materialize_vcov` and checks its dimensions against the coefficient
    vector.
  - `_custom_stderror` returns the square roots of diagonal entries; `_stderror` dispatches to it when the wrapper is
    present.
  - Delegation methods forward all other `StatsAPI` queries (coef, dof, nobs, etc.) to the wrapped model so existing
    statistics keep working.
- **User API**:
  - `model + vcov(matrix)` uses a provided matrix verbatim.
  - `model + vcov(model -> matrix)` or `model + vcov(() -> matrix)` compute the matrix lazily.
  - Any other object can participate if `RegressionTables.materialize_vcov(obj, model)` is defined, giving external
    packages a clean hook.
- **Error handling**: Dimension mismatches, non-matrix returns, or unknown specs throw informative `ArgumentError`s so
  misconfigurations surface quickly.
- **Testing**: Added to `test/RegressionTables.jl` to assert correctness and caching semantics without depending on
  CovarianceMatrices.jl.

This architecture keeps the core API simple for end users, isolates the customization surface for partner packages, and
maintains backward compatibility with existing regression types.
