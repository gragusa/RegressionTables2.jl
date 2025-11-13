"""
    mutable struct RegressionTable
        data::Matrix{Any}
        header::Vector{Vector{String}}
        header_align::Vector{Symbol}
        body_align::Vector{Symbol}
        hlines::Vector{Int}
        formatters::Vector
        highlighters::Vector
        backend::Union{Symbol, Nothing}
        pretty_kwargs::Dict{Symbol, Any}
    end

A container for regression table data that uses PrettyTables.jl for rendering.

# Fields
- `data`: Matrix of table data (both header and body combined)
- `header`: Vector of header rows (for multi-level headers)
- `header_align`: Alignment for header columns (:l, :c, :r)
- `body_align`: Alignment for body columns (:l, :c, :r)
- `hlines`: Positions of horizontal lines (row indices)
- `formatters`: Vector of PrettyTables formatters
- `highlighters`: Vector of PrettyTables highlighters
- `backend`: Rendering backend (:text, :html, :latex, or nothing for auto-detection)
- `pretty_kwargs`: Additional keyword arguments to pass to PrettyTables.pretty_table

# Display
The table automatically selects the appropriate backend based on MIME type:
- Terminal/REPL: Markdown backend (text)
- Jupyter/HTML context: HTML backend
- LaTeX context: LaTeX backend

You can override the backend using `set_backend!` or by setting the `backend` field directly.

# Customization
After creating a table, you can customize it using:
- `add_hline!`: Add horizontal lines
- `set_alignment!`: Change column alignment
- `add_formatter!`: Add custom formatters
- `set_backend!`: Change the rendering backend
- `merge_kwargs!`: Add arbitrary PrettyTables.jl options

# Examples
```julia
julia> rt = modelsummary(model1, model2)  # Creates a RegressionTable

julia> add_hline!(rt, 5)  # Add horizontal line after row 5

julia> set_backend!(rt, :latex)  # Force LaTeX backend

julia> merge_kwargs!(rt; title="My Regression Results")  # Add PrettyTables options
```
"""
mutable struct RegressionTable
    data::Matrix{Any}
    header::Vector{Vector{String}}
    header_align::Vector{Symbol}
    body_align::Vector{Symbol}
    hlines::Vector{Int}
    formatters::Vector
    highlighters::Vector
    backend::Union{Symbol, Nothing}
    pretty_kwargs::Dict{Symbol, Any}

    function RegressionTable(
        data::Matrix{Any},
        header::Vector{Vector{String}},
        header_align::Vector{Symbol},
        body_align::Vector{Symbol};
        hlines::Vector{Int}=Int[],
        formatters::Vector=[],
        highlighters::Vector=[],
        backend::Union{Symbol, Nothing}=nothing,
        pretty_kwargs::Dict{Symbol, Any}=Dict{Symbol, Any}()
    )
        new(data, header, header_align, body_align, hlines, formatters, highlighters, backend, pretty_kwargs)
    end
end

# Convenience constructor for simple matrices
function RegressionTable(
    header::Vector{String},
    body::Matrix{Any};
    header_align::Union{Vector{Symbol}, Nothing}=nothing,
    body_align::Union{Vector{Symbol}, Nothing}=nothing,
    kwargs...
)
    ncols = length(header)
    @assert size(body, 2) == ncols "Header and body must have same number of columns"

    # Default alignments: left for first column, right for others
    if header_align === nothing
        header_align = [:l; fill(:c, ncols - 1)]
    end
    if body_align === nothing
        body_align = [:l; fill(:r, ncols - 1)]
    end

    RegressionTable(
        body,
        [header],
        header_align,
        body_align;
        kwargs...
    )
end

"""
    add_hline!(rt::RegressionTable, position::Int)

Add a horizontal line after the specified row position.
Row numbering includes header rows.
"""
function add_hline!(rt::RegressionTable, position::Int)
    if position ∉ rt.hlines
        push!(rt.hlines, position)
        sort!(rt.hlines)
    end
    rt
end

"""
    remove_hline!(rt::RegressionTable, position::Int)

Remove a horizontal line at the specified row position.
"""
function remove_hline!(rt::RegressionTable, position::Int)
    filter!(x -> x != position, rt.hlines)
    rt
end

"""
    set_alignment!(rt::RegressionTable, col::Int, align::Symbol; header::Bool=false)

Set the alignment for a specific column.
Set `header=true` to change header alignment instead of body alignment.
"""
function set_alignment!(rt::RegressionTable, col::Int, align::Symbol; header::Bool=false)
    @assert align in (:l, :c, :r) "Alignment must be :l, :c, or :r"
    if header
        rt.header_align[col] = align
    else
        rt.body_align[col] = align
    end
    rt
end

"""
    add_formatter!(rt::RegressionTable, f)

Add a PrettyTables formatter to the table.
See PrettyTables.jl documentation for formatter syntax.
"""
function add_formatter!(rt::RegressionTable, f)
    push!(rt.formatters, f)
    rt
end

"""
    set_backend!(rt::RegressionTable, backend::Symbol)

Set the rendering backend.
Valid backends: :text, :html, :latex, or :auto (nothing) for automatic detection.
"""
function set_backend!(rt::RegressionTable, backend::Union{Symbol, Nothing})
    if backend !== nothing
        @assert backend in (:text, :html, :latex) "Backend must be :text, :html, :latex, or nothing"
    end
    rt.backend = backend
    rt
end

"""
    merge_kwargs!(rt::RegressionTable; kwargs...)

Merge additional keyword arguments to pass to PrettyTables.pretty_table.
This allows you to use any PrettyTables.jl option for customization.

# Examples
```julia
merge_kwargs!(rt; title="My Results", title_alignment=:c)
merge_kwargs!(rt; vcrop_mode=:middle, crop_num_lines_at_end=10)
```
"""
function merge_kwargs!(rt::RegressionTable; kwargs...)
    merge!(rt.pretty_kwargs, Dict{Symbol, Any}(kwargs))
    rt
end

# Make RegressionTable act like a matrix for compatibility
Base.size(rt::RegressionTable) = size(rt.data)
Base.size(rt::RegressionTable, i::Int) = size(rt.data, i)
Base.getindex(rt::RegressionTable, i::Int, j::Int) = rt.data[i, j]
function Base.setindex!(rt::RegressionTable, val, i::Int, j::Int)
    rt.data[i, j] = val
    rt
end

# Helper functions to convert alignment symbols to PrettyTables format
function _pt_alignment(align::Vector{Symbol})
    return align
end

function _convert_alignment_char(c::Char)
    c == 'l' ? :l : (c == 'c' ? :c : :r)
end

function _convert_alignment_string(s::String)
    [_convert_alignment_char(c) for c in s]
end

# Main printing function using PrettyTables
"""
    _render_table(io::IO, rt::RegressionTable, backend::Symbol)

Internal function to render the table using PrettyTables.jl.
"""
function _render_table(io::IO, rt::RegressionTable, backend::Symbol)
    # Prepare the full data matrix (header + body)
    nheader = length(rt.header)

    # Build alignment vector (header uses header_align, body uses body_align)
    alignment = rt.body_align

    # Adjust hlines to account for PrettyTables' header handling
    # PrettyTables puts an automatic line after headers, so we need to adjust our hlines
    hlines_adjusted = copy(rt.hlines)

    # PrettyTables configuration based on backend
    kwargs = copy(rt.pretty_kwargs)

    if backend == :text
        kwargs[:backend] = Val(:text)
        kwargs[:tf] = PrettyTables.tf_markdown
        # Add horizontal lines
        if !isempty(hlines_adjusted)
            kwargs[:body_hlines] = hlines_adjusted
        end
        kwargs[:alignment] = alignment
        kwargs[:header_alignment] = rt.header_align

    elseif backend == :html
        kwargs[:backend] = Val(:html)
        kwargs[:tf] = PrettyTables.tf_html_minimalist
        kwargs[:alignment] = alignment
        kwargs[:header_alignment] = rt.header_align
        # HTML doesn't have the same hlines concept, but we can use CSS classes

    elseif backend == :latex
        kwargs[:backend] = Val(:latex)
        kwargs[:tf] = PrettyTables.tf_latex_booktabs
        # Add horizontal lines
        if !isempty(hlines_adjusted)
            kwargs[:body_hlines] = hlines_adjusted
        end
        kwargs[:alignment] = alignment
        kwargs[:header_alignment] = rt.header_align
    end

    # Add formatters if any
    if !isempty(rt.formatters)
        kwargs[:formatters] = tuple(rt.formatters...)
    end

    # Add highlighters if any
    if !isempty(rt.highlighters)
        kwargs[:highlighters] = tuple(rt.highlighters...)
    end

    # Render using PrettyTables
    PrettyTables.pretty_table(
        io,
        rt.data,
        header=rt.header;
        kwargs...
    )
end

# MIME-based display methods
function Base.show(io::IO, ::MIME"text/plain", rt::RegressionTable)
    backend = rt.backend === nothing ? :text : rt.backend
    _render_table(io, rt, backend)
end

function Base.show(io::IO, ::MIME"text/html", rt::RegressionTable)
    backend = rt.backend === nothing ? :html : rt.backend
    _render_table(io, rt, backend)
end

function Base.show(io::IO, ::MIME"text/latex", rt::RegressionTable)
    backend = rt.backend === nothing ? :latex : rt.backend
    _render_table(io, rt, backend)
end

# Default show method (uses text/plain)
function Base.show(io::IO, rt::RegressionTable)
    show(io, MIME("text/plain"), rt)
end

# Write to file
function Base.write(filename::String, rt::RegressionTable)
    open(filename, "w") do io
        # Detect backend from file extension
        backend = rt.backend
        if backend === nothing
            ext = lowercase(splitext(filename)[2])
            if ext == ".tex"
                backend = :latex
            elseif ext in (".html", ".htm")
                backend = :html
            else
                backend = :text
            end
        end
        _render_table(io, rt, backend)
    end
end

#=============================================================================
Compatibility constructor for old DataRow-based system
=============================================================================#

"""
    RegressionTable(data::Vector{DataRow{T}}, align::String, breaks::Vector{Int}) where {T<:AbstractRenderType}

Constructor that converts DataRow-based tables to PrettyTables-based format.
This allows modelsummary() to build tables using the DataRow system internally.
"""
function RegressionTable(
    data::Vector{DataRow{T}},
    align::String,
    breaks::Vector{Int}=Int[],
    colwidths::Vector{Int}=Int[]
) where {T<:AbstractRenderType}
    # Convert DataRow vector to matrix format
    nrows = length(data)
    if nrows == 0
        error("Cannot create table from empty DataRow vector")
    end

    # Determine table structure
    # First rows with underlines are headers, rest are body
    header_rows = Int[]
    for (i, row) in enumerate(data)
        if any(row.print_underlines)
            push!(header_rows, i)
        else
            break  # Once we hit a row without underlines, we're in the body
        end
    end

    nheader = length(header_rows)
    if nheader == 0
        # No header rows, treat first row as header
        nheader = 1
        header_rows = [1]
    end

    # Determine number of columns from first row
    # Handle multicolumn cells (Pairs)
    function count_cols(row::DataRow)
        n = 0
        for item in row.data
            if isa(item, Pair)
                n += length(last(item))
            else
                n += 1
            end
        end
        n
    end

    ncols = count_cols(data[1])

    # Convert rows to flat vectors (expanding multicolumn cells)
    function expand_row(row::DataRow, ncols::Int)
        result = fill("", ncols)
        col = 1
        for item in row.data
            if isa(item, Pair)
                # Multicolumn cell
                value = repr(row.render, first(item))
                span = length(last(item))
                # Put value in first column of span
                result[col] = value
                # Mark other columns as part of multicolumn (we'll handle in PrettyTables)
                for j in 1:(span-1)
                    result[col + j] = ""  # Empty for now, PrettyTables will handle merging
                end
                col += span
            else
                result[col] = repr(row.render, item)
                col += 1
            end
        end
        result
    end

    # Build header matrix
    header_matrix = Matrix{String}(undef, nheader, ncols)
    for (i, row_idx) in enumerate(header_rows)
        header_matrix[i, :] = expand_row(data[row_idx], ncols)
    end

    # Build body matrix
    body_start = nheader + 1
    nbody = nrows - nheader
    body_matrix = Matrix{Any}(undef, nbody, ncols)
    for i in 1:nbody
        body_matrix[i, :] = expand_row(data[body_start + i - 1], ncols)
    end

    # Convert alignment string to symbol vector
    align_vec = [c == 'l' ? :l : (c == 'c' ? :c : :r) for c in align]

    # Ensure we have alignment for all columns
    while length(align_vec) < ncols
        push!(align_vec, :r)
    end

    # Build header as vector of vectors (one per header row)
    header_vecs = [String[header_matrix[i, j] for j in 1:ncols] for i in 1:nheader]

    # Create alignment vectors
    header_align = fill(:c, ncols)
    header_align[1] = :l  # First column left-aligned

    body_align = copy(align_vec[1:ncols])

    # Adjust breaks (they're 1-indexed in old system, need to subtract header rows)
    adjusted_breaks = [b - nheader for b in breaks if b > nheader]

    # Determine backend from render type
    backend = if T <: AbstractLatex
        :latex
    elseif T <: AbstractHtml
        :html
    else
        nothing  # Auto-detect
    end

    # Create new RegressionTable
    rt = RegressionTable(
        body_matrix,
        header_vecs,
        header_align,
        body_align;
        hlines=adjusted_breaks,
        backend=backend
    )

    return rt
end
