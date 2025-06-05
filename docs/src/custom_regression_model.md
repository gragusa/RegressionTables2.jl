
# Implementing Custom Regression Models

## Example 1

I am not sure I fully understand what you are asking for, it might help to see some of your code to understand what it currently looks like so it can fit into the `RegressionModel` type.

I think what you are asking for is a basic regression model. So first, you need to create a type for your regression model:
```julia
using StatsAPI, StatsModels, RegressionTables, RDatasets, Statistics
struct MyStatsModel <: RegressionModel
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    dof::Int
    dof_residual::Int
    nobs::Int
    rss::Float64
    tss::Float64
    coefnames::Vector{String}
    responsename::String
    formula::FormulaTerm
    formula_schema::FormulaTerm
end
```

For your specific use case, there might be other things that you need to add to that.

You then need to match the StatsAPI elements to your new type:
```julia
StatsAPI.coef(m::MyStatsModel) = m.coef
StatsAPI.coefnames(m::MyStatsModel) = m.coefnames
StatsAPI.vcov(m::MyStatsModel) = m.vcov
StatsAPI.dof(m::MyStatsModel) = m.dof
StatsAPI.dof_residual(m::MyStatsModel) = m.dof_residual
StatsAPI.nobs(m::MyStatsModel) = m.nobs
StatsAPI.rss(m::MyStatsModel) = m.rss
StatsAPI.nulldeviance(m::MyStatsModel) = m.tss
StatsAPI.islinear(m::MyStatsModel) = true
StatsAPI.deviance(m::MyStatsModel) = StatsAPI.rss(m)
StatsAPI.mss(m::MyStatsModel) = nulldeviance(m) - StatsAPI.rss(m)
StatsModels.formula(m::MyStatsModel) = m.formula_schema
StatsModels.formula(m::MyStatsModel) = m.formula_schema

#edit add:
StatsAPI.r2(m::MyStatsModel) = StatsAPI.r2(m, :devianceratio)
```
(I find looking at [FixedEffectModels.jl/src/FixedEffectModel.jl at master · FixedEffects/FixedEffectModels.jl (github.com)](https://github.com/FixedEffects/FixedEffectModels.jl/blob/master/src/FixedEffectModel.jl) a reasonably straightforward set of how to implement this).

Then you would need a function that creates that type:
```julia
function StatsAPI.fit(::Type{MyStatsModel}, f::FormulaTerm, df::DataFrame)
    df = dropmissing(df)
    f_schema = apply_schema(f, schema(f, df))
    y, X = modelcols(f_schema, df)
    response_name, coefnames_exo = coefnames(f_schema)
    n, p = size(X)
    β = X \ y
    ŷ = X * β
    res = y - ŷ
    rss = sum(abs2, res)
    tss = sum(abs2, y .- mean(y))
    dof = p
    dof_residual = n - p
    vcov = inv(X'X) * rss / dof_residual
    MyStatsModel(β, vcov, dof, dof_residual, n, rss, tss, coefnames_exo, response_name, f, f_schema)
end
```

This will work with RegressionTables
```julia
rr1 = StatsAPI.fit(MyStatsModel, @formula(Sales ~ 1 + Price + NDI), df)
rr2 = StatsAPI.fit(MyStatsModel, @formula(Sales ~ 1 + Price), df)
rr3 = StatsAPI.fit(MyStatsModel, @formula(Sales ~ 1 + NDI), df)
regtable(rr1, rr2, rr3)
--------------------------------------------------
                              Sales
              ------------------------------------
                     (1)          (2)          (3)
--------------------------------------------------
(Intercept)   138.480***   139.734***   132.981***
                 (1.427)      (1.521)      (1.538)
Price          -0.938***    -0.230***
                 (0.054)      (0.019)
NDI             0.007***                 -0.001***
                 (0.000)                   (0.000)
--------------------------------------------------
N                  1,380        1,380        1,380
R2                 0.209        0.097        0.034
--------------------------------------------------
```
```
(I find looking at [FixedEffectModels.jl/src/FixedEffectModel.jl at master · FixedEffects/FixedEffectModels.jl (github.com)](https://github.com/FixedEffects/FixedEffectModels.jl/blob/master/src/FixedEffectModel.jl) a reasonably straightforward set of how to implement this).

Then you would need a function that creates that type:
```julia
function StatsAPI.fit(::Type{MyStatsModel}, f::FormulaTerm, df::DataFrame)
    df = dropmissing(df)
    f_schema = apply_schema(f, schema(f, df))
    y, X = modelcols(f_schema, df)
    response_name, coefnames_exo = coefnames(f_schema)
    n, p = size(X)
    β = X \ y
    ŷ = X * β
    res = y - ŷ
    rss = sum(abs2, res)
    tss = sum(abs2, y .- mean(y))
    dof = p
    dof_residual = n - p
    vcov = inv(X'X) * rss / dof_residual
    MyStatsModel(β, vcov, dof, dof_residual, n, rss, tss, coefnames_exo, response_name, f, f_schema)
end
```

This will work with RegressionTables
```julia
rr1 = StatsAPI.fit(MyStatsModel, @formula(Sales ~ 1 + Price + NDI), df)
rr2 = StatsAPI.fit(MyStatsModel, @formula(Sales ~ 1 + Price), df)
rr3 = StatsAPI.fit(MyStatsModel, @formula(Sales ~ 1 + NDI), df)
regtable(rr1, rr2, rr3)
--------------------------------------------------
                              Sales
              ------------------------------------
                     (1)          (2)          (3)
--------------------------------------------------
(Intercept)   138.480***   139.734***   132.981***
                 (1.427)      (1.521)      (1.538)
Price          -0.938***    -0.230***
                 (0.054)      (0.019)
NDI             0.007***                 -0.001***
                 (0.000)                   (0.000)
--------------------------------------------------
N                  1,380        1,380        1,380
R2                 0.209        0.097        0.034
--------------------------------------------------
```

## Example 2

### My suggestions

[quote="junder873, post:8, topic:113477, full:true"]
### islinear and `RegressionType`

`islinear` is part of the StatsAPI, which is why it is implemented. You can actually avoid it by instead defining `RegressionType` for your type:

```
RegressionTables.RegressionType(rr::MyStatsModel) = RegressionTables.RegressionType("My type")
```

The reason you are seeing the current behavior is line 142 of [RegressionTables.jl/src/regressionResults.jl at master · jmboehm/RegressionTables.jl (github.com)](https://github.com/jmboehm/RegressionTables.jl/blob/master/src/regressionResults.jl):

```
RegressionType(x::RegressionModel) = islinear(x) ? RegressionType(Normal()) : RegressionType("NL")
```

So you would basically override that. You can also suppress that line of the result by setting `regtable(...; print_estimator=false` or for your entire session with `RegressionTables.default_print_estimator(render::AbstractRenderType, rrs) = false`.

### Regression Summary Statistics

In general, it might be worthwhile to look through the extensions on the RegressionTables package. Each of those is written to modify the default table results. For example, if you do not want any regression statistics for your data, you can run:

```
RegressionTables.default_regression_statistics(rr::MyStatsModel) = Symbol[]
```

However, note that if you are running multiple types of regressions in one table (e.g., GLM + MyStatsModel), then the default GLM statistics will force the printing of those statistics anyway.

Another alternative is you can remove (or change the order) of sections as you want by changing the `section_order` argument: `regtable(...; section_order=...)` or the default:

```
RegressionTables.default_section_order(render::AbstractRenderType) = [:groups, :depvar, :number_regressions, :break, :coef, :break, :fe, :break, :randomeffects, :break, :clusters, :break, :regtype, :break, :controls, :break, :stats, :extralines]
```

So if you do not want the RegressionType or statistics, remove `:regtype` and `:stats` from that list.

### A Suggestion Based on What I think you are doing

The underlying design of RegressionTables is that `extralines` is more of a last resort. So, in addition to changing the `RegressionType` and `default_regression_statistics` to fit your needs, there are two other ways to implement custom statistics: 1) define other statistics and 2) use `other_stats`

Just to modify what I did before a little, add to the struct a `special_value`:

```
    formula_schema::FormulaTerm
    special_value::String
end
```

and to the function something to that:

```
function StatsAPI.fit(::Type{MyStatsModel}, f::FormulaTerm, df::DataFrame; special_value::String="ABC")
...
    MyStatsModel(β, vcov, dof, dof_residual, n, rss, tss, coefnames_exo, response_name, f, f_schema, special_value)
end
```

And just to give an example of defining your own `RegressionType`:

```
RegressionTables.RegressionType(rr::MyStatsModel) = RegressionType("MyModel")
```

#### Defining New Statistics

It is possible to define new RegressionStatistics in RegressionTables, these will appear alongside other statistics:

```
struct MyStatistic <: RegressionTables.AbstractRegressionStatistic
    val::Union{String, Nothing}
end
MyStatistic(rr::RegressionModel) = MyStatistic(nothing)
MyStatistic(rr::MyStatsModel) = MyStatistic(rr.special_value)
RegressionTables.label(render::AbstractRenderType, x::Type{MyStatistic}) = "My Statistic"
RegressionTables.default_regression_statistics(rr::MyStatsModel) = [Nobs, MyStatistic]

using FixedEffectModels
rr1 = StatsAPI.fit(MyStatsModel, @formula(Sales ~ 1 + Price + NDI), df)
rr2 = StatsAPI.fit(MyStatsModel, @formula(Sales ~ 1 + Price), df; special_value="123")
rr3 = StatsAPI.fit(MyStatsModel, @formula(Sales ~ 1 + NDI), df; special_value="ABC")
rr4 = reg(df, @formula(Sales ~ Price + NDI))
tab = regtable(rr1, rr2, rr3, rr4)

----------------------------------------------------------------
                                     Sales
               -------------------------------------------------
                      (1)          (2)          (3)          (4)
----------------------------------------------------------------
(Intercept)    138.480***   139.734***   132.981***   138.480***
                  (1.427)      (1.521)      (1.538)      (1.427)
Price           -0.938***    -0.230***                 -0.938***
                  (0.054)      (0.019)                   (0.054)
NDI              0.007***                 -0.001***     0.007***
                  (0.000)                   (0.000)      (0.000)
----------------------------------------------------------------
Estimator         MyModel      MyModel      MyModel          OLS
----------------------------------------------------------------
N                   1,380        1,380        1,380        1,380
My Statistic          ABC          123          ABC
R2                  0.209        0.097        0.034        0.209
----------------------------------------------------------------
```

#### Using `other_stats`

The `other_stats` function is useful for defining a new section. It is how [FixedEffectModels.jl](https://juliaregistries.github.io/General/packages/redirect_to_repo/FixedEffectModels) and [MixedModels.jl](https://juliaregistries.github.io/General/packages/redirect_to_repo/MixedModels) implement necessary pieces. You need to create an `other_stats` function to fit your needs and then add whatever symbol you created to `default_section_order`:

```
function RegressionTables.other_stats(rr::MyStatsModel, s::Symbol)
    if s == :my_stat
        return ["Some Descriptor" => MyStatistic(rr)]
    else
        return nothing
    end
end
RegressionTables.default_section_order(render::AbstractRenderType) = [:groups, :depvar, :number_regressions, :break, :coef, :break, :fe, :break, :randomeffects, :break, :clusters, :break, :regtype, :break, :controls, :break, :stats, :break, :my_stat, :extralines]

rr1 = StatsAPI.fit(MyStatsModel, @formula(Sales ~ 1 + Price + NDI), df)
rr2 = StatsAPI.fit(MyStatsModel, @formula(Sales ~ 1 + Price), df; special_value="123")
rr3 = StatsAPI.fit(MyStatsModel, @formula(Sales ~ 1 + NDI), df; special_value="ABC")
rr4 = reg(df, @formula(Sales ~ Price + NDI))
tab = regtable(rr1, rr2, rr3, rr4)

-------------------------------------------------------------------
                                        Sales
                  -------------------------------------------------
                         (1)          (2)          (3)          (4)
-------------------------------------------------------------------
(Intercept)       138.480***   139.734***   132.981***   138.480***
                     (1.427)      (1.521)      (1.538)      (1.427)
Price              -0.938***    -0.230***                 -0.938***
                     (0.054)      (0.019)                   (0.054)
NDI                 0.007***                 -0.001***     0.007***
                     (0.000)                   (0.000)      (0.000)
-------------------------------------------------------------------
Estimator            MyModel      MyModel      MyModel          OLS
-------------------------------------------------------------------
N                      1,380        1,380        1,380        1,380
R2                     0.209        0.097        0.034        0.209
-------------------------------------------------------------------
Some Descriptor          ABC          123          ABC
-------------------------------------------------------------------
```

Conveniently, `other_stats` can be used to create as many new sections as you want, so if you need those broken out, that is possible.
[/quote]



### Result

```julia
using Optim, NLSolversBase
using ForwardDiff
using NamedArrays
using StatsAPI, StatsModels, RegressionTables, Statistics, LinearAlgebra, HypothesisTests


n = 40                              # Number of observations
nvar = 2                            # Number of variables
β = ones(nvar) * 3.0                # True coefficients
x = [ 1.0   0.156651				# X matrix of explanatory variables plus constant
 1.0  -1.34218
 1.0   0.238262
 1.0  -0.496572
 1.0   1.19352
 1.0   0.300229
 1.0   0.409127
 1.0  -0.88967
 1.0  -0.326052
 1.0  -1.74367
 1.0  -0.528113
 1.0   1.42612
 1.0  -1.08846
 1.0  -0.00972169
 1.0  -0.85543
 1.0   1.0301
 1.0   1.67595
 1.0  -0.152156
 1.0   0.26666
 1.0  -0.668618
 1.0  -0.36883
 1.0  -0.301392
 1.0   0.0667779
 1.0  -0.508801
 1.0  -0.352346
 1.0   0.288688
 1.0  -0.240577
 1.0  -0.997697
 1.0  -0.362264
 1.0   0.999308
 1.0  -1.28574
 1.0  -1.91253
 1.0   0.825156
 1.0  -0.136191
 1.0   1.79925
 1.0  -1.10438
 1.0   0.108481
 1.0   0.847916
 1.0   0.594971
 1.0   0.427909]

ε = [   0.5539830489065279             # Errors
        -0.7981494315544392
        0.12994853889935182
  0.23315434715658184
 -0.1959788033050691
 -0.644463980478783
 -0.04055657880388486
 -0.33313251280917094
 -0.315407370840677
  0.32273952815870866
  0.56790436131181
  0.4189982390480762
 -0.0399623088796998
 -0.2900421677961449
 -0.21938513655749814
 -0.2521429229103657
  0.0006247891825243118
 -0.694977951759846
 -0.24108791530910414
  0.1919989647431539
  0.15632862280544485
 -0.16928298502504732
  0.08912288359190582
  0.0037707641031662006
 -0.016111044809837466
  0.01852191562589722
 -0.762541135294584
 -0.7204431774719634
 -0.04394527523005201
 -0.11956323865320413
 -0.6713329013627437
 -0.2339928433338628
 -0.6200532213195297
 -0.6192380993792371
  0.08834918731846135
 -0.5099307915921438
  0.41527207925609494
 -0.7130133329859893
 -0.531213372742777
 -0.09029672309221337]

y = x * β + ε;                      # Generate Data



struct CustomModel <: RegressionModel
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    dof_residual::Int
    nobs::Int
    coefnames::Vector{String}
    responsename::String
    LogLikelihood::Float64
    BIC::Float64
    BS::Float64
    civec::Matrix{Float64}
end

struct MyStatistic <: RegressionTables.AbstractRegressionStatistic
    val::Union{Float64, Nothing}
end


StatsAPI.coef(m::CustomModel) = m.coef
RegressionTables._coefnames(m::CustomModel) = RegressionTables.get_coefname(m.coefnames)
StatsAPI.coefnames(m::CustomModel) = m.coefnames
StatsAPI.responsename(m::CustomModel) = m.responsename
RegressionTables._responsename(m::CustomModel) = RegressionTables.get_coefname(m.responsename)
StatsAPI.vcov(m::CustomModel) = m.vcov
StatsAPI.nobs(m::CustomModel) = m.nobs
StatsAPI.dof_residual(m::CustomModel) = m.dof_residual
RegressionTables.RegressionType(m::CustomModel) = RegressionTables.RegressionType("My type")

StatsAPI.loglikelihood(m::CustomModel) = m.LogLikelihood;


StatsAPI.bic(m::CustomModel) = m.BIC


function Base.repr(render::AbstractRenderType, x::ConfInt; digits=RegressionTables.default_digits(render, x), args...)
    if RegressionTables.value(x) == (0, 0) # 0 == 0.000
        repr(render, "restricted")
    else
        RegressionTables.below_decoration(render, repr(render, RegressionTables.value(x)[1]; digits) * ", " * Base.repr(render::AbstractRenderType, RegressionTables.value(x)[2]; digits))
    end
end

function StatsAPI.confint(m::CustomModel; level::Real = 0.95)
    m.civec
end

MyStatistic(m::CustomModel) = MyStatistic(nothing)
MyStatistic(m::CustomModel) = MyStatistic(m.BS)
RegressionTables.label(render::AbstractRenderType, x::Type{MyStatistic}) = "Normality"
RegressionTables.default_regression_statistics(m::CustomModel) = [Nobs,MyStatistic,LogLikelihood,BIC]

RegressionTables.default_symbol(render::AbstractRenderType) = ""


function StatsAPI.fit(::Type{CustomModel},X,Y,s_v,restricted_parameters)
    
    function Log_Likelihood(X, Y, params::AbstractVector{T}, restricted_parameters) where T
    
        full_parameter_vector = vcat(params,restricted_parameters)
        n = size(X,1);
        σ = exp(full_parameter_vector["Sigma"])
        llike = -n/2*log(2π) - n/2* log(σ^2) - (sum((Y - X[:,1] * full_parameter_vector["Beta 1"] - X[:,2] * full_parameter_vector["Beta 2"]).^2) / (2σ^2))
        llike = -llike
    end
    
    func = TwiceDifferentiable(vars -> Log_Likelihood(x, y, vars, s_v_r),s_v; autodiff=:forward);
    
    opt = optimize(func, s_v)
    
    parameters = Optim.minimizer(opt)
    
    var_cov_matrix = inv(hessian!(func,parameters))

    padding = zeros(length(parameters),length(restricted_parameters));
    augmented_var_cov_matrix = [var_cov_matrix padding;
                                padding'     I(length(restricted_parameters))]

    n = size(x,1);

    names_param = names(vcat(parameters,restricted_parameters),1);

    

    ll = -opt.minimum

    left_end = parameters - 1.96 * sqrt.(diag(var_cov_matrix))
    right_end = parameters + 1.96 * sqrt.(diag(var_cov_matrix))

    #apply transformations where needed
    left_end["Sigma"] = exp(left_end["Sigma"]);
    right_end["Sigma"] = exp(right_end["Sigma"]);
    parameters["Sigma"] = exp(parameters["Sigma"]);
    
    #Diagnostics
    all_parameters = vcat(parameters,restricted_parameters)
    residuals = y - x * all_parameters[["Beta 1","Beta 2"]];
    BIC = -2 * ll + log(n) * length(parameters);
    BS = JarqueBeraTest(residuals; adjusted=true).JB;
    
    civec = vcat(hcat(left_end.array,right_end.array),[0 0]);

    #create fitted model and return that
    CustomModel(all_parameters.array,augmented_var_cov_matrix.array,n-length(s_v),n,names_param,"Outcome",ll,BIC,BS,civec)
    
end


s_v = NamedArray(Array{Float64}(undef,2));
setnames!(s_v,["Beta 2","Sigma"],1)
s_v["Beta 2"] = 1;
s_v["Sigma"] = 0.05;

s_v_r = NamedArray(Array{Float64}(undef,1));
setnames!(s_v_r,["Beta 1"],1);
s_v_r["Beta 1"] = 3.


rr1 = StatsAPI.fit(CustomModel,x,y,s_v,s_v_r)


RegressionTables.regtable(rr1,below_statistic = ConfInt)

```

Output

```julia 
-------------------------------
                     Outcome
-------------------------------
Beta 2                    3.069
                 (2.926, 3.212)
Sigma                     0.406
                 (0.326, 0.506)
Beta 1                    3.000
                     restricted
-------------------------------
N                            40
Normality                 1.115
Log Likelihood          -20.722
BIC                      48.823
-------------------------------
```