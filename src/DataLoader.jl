# 定义用于道路检测的数据加载器
using Flux
using ImageCore
using ImageIO
using FileIO
using CSV
using DataFrames
import Base.length, Base.getindex


# CSV文件和数据目录的路径
metadata_csv_path = "/home/yu/道路检测数据集/Massachusetts/metadata.csv" # 替换为metadata.csv文件实际的路径
data_dir = "/home/yu/道路检测数据集/Massachusetts/" # 替换为实际的图像文件夹路径

# 读取CSV文件并筛选出训练集数据
metadata = CSV.read(metadata_csv_path, DataFrame)

# 加载图像和标签的函数
function load_pair(image_path, label_path)
    image = load(joinpath(data_dir, image_path))
    label = load(joinpath(data_dir, label_path))
    # 将标签二值化: 背景为0, 道路为1
    label = label .> 0.5 # 假设标签图像是白色道路，黑色背景
    return (permutedims(channelview(image), (2, 3, 1)) .|> Float32, channelview(label) .|> Float32)
end

struct Massachusetts
    image_files::Vector{String}
    label_files::Vector{String}
end

Massachusetts(metadata, split::String="train") = Massachusetts(
    filter(row -> row.split == split, metadata).tiff_image_path .|> String,
    filter(row -> row.split == split, metadata).tif_label_path
)

getobs(data::Massachusetts, idx::Int) = cat.(load_pair(data.image_files[idx], data.label_files[idx]), dims=4)
function getobs(data::Massachusetts, idx::Vector{Int})
    xs = []
    ys = []
    for i in idx
        x, y = getobs(data, i)
        push!(xs, x)
        push!(ys, y)
    end
    return cat(xs..., dims=4), cat(ys..., dims=4)
end
getindex(data::Massachusetts, idx) = getobs(data, idx)
numobs(data::Massachusetts) = Base.length(data.image_files)
length(data::Massachusetts) = numobs(data)

# 创建DataLoader
massachusetts_dataset(split::String="train", batchsize=12) = Flux.DataLoader(Massachusetts(metadata, split), batchsize=batchsize, shuffle=true)

# 定义损失函数
# 用于使用Flux损失函数的框架
"""
    calculate_loss(ŷ, y)
    计算我的道路检测任务网络训练之后的损失函数，由于道路检测是2分类，所以标签是单通道的
    Arguments:
    * `ŷ`: 网络输出，x × y × 3 × batch_size Array{Float32, 4}
    * `y`: 标签，x × y × 1 × batch_size BitArray{4}
    * `loss_function`: 损失函数，默认为Flux.dice_coeff_loss
"""
function calculate_loss(ŷ, y, loss_function=Flux.dice_coeff_loss)
    # 拉直y和ŷ
    flatten(x) = reshape(x, (size(x)[1] * size(x)[2], size(x)[3], size(x)[4]))
    return loss_function(softmax(ŷ, dims=3) |> flatten, cat(y, 1 .- y, dims=3) |> flatten)
end

# 训练
include("Unet.jl")
model = Unet(3, 2) |> gpu
for epoch in 1:1
    @time for (x, y) in massachusetts_dataset("val", 1) |> gpu
        ŷ = model(x)
        @show loss = calculate_loss(ŷ, y)
        @show grads = Flux.gradient(m -> calculate_loss(m(x), y), model)
        Flux.Optimise.update!(opt, model, grads[1])
        # 在这里进行训练
        # x 是图像批次，y 是标签批次
    end
end