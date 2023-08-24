module HackORtron

using CSV, DataFrames

export perceptron, FILE_TEST, FILE_TRAIN

FILE_TEST = "data/test.csv"
FILE_TRAIN = "data/train.csv"

BIAS = 0.0
COLUMNS = [:Pclass, :Age, :SibSp, :Parch, :Fare]
WEIGHTS = [-1.0, -1.0, 1.0, 1.0, 1.0]

function perceptron(
        input::DataFrame, 
        bias::Float64 = BIAS, 
        columns::Vector{Symbol} = COLUMNS,
        weights::Vector{Float64} = WEIGHTS
    )

    # Input
    x = input[:, columns]
    x = Matrix(x)
    for col in eachcol(x)
        replace!(col, missing => 0.0)
        col ./= maximum(col)
    end

    # Calculations
    # y = b + w * x
    y = bias .+ x * weights

    # Output
    minVal, _ = findmin(y)
    maxVal, _ = findmax(y)
    avg = (minVal + maxVal) / 2
    y = Int.(y .> avg)
end

function run(
        input_file::String, 
        output_file::String
    )

    df = CSV.read(input_file, DataFrame)
    y = perceptron(df)
    out = join(string.(y), "\n")
    open(output_file, "w") do f
        write(f, out)
    end
end

end # module
