module RegressionTables2TypstryExt

using Typstry, RegressionTables2

Typstry.show_typst(io::IO, ::TypstContext, rt::RegressionTable{<:AbstractTypst}) = print(io, rt)

end
