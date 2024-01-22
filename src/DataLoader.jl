# 定义用于道路检测的数据加载器
using Flux
using DataLoaders
using ImageCore
using ImageIO
using FileIO
using CSV
using DataFrames

# CSV文件和数据目录的路径
metadata_csv_path = "path/to/Massachusetts/metadata.csv" # 替换为metadata.csv文件实际的路径
data_dir = "path/to/Massachusetts/tiff" # 替换为实际的图像文件夹路径

# 读取CSV文件并筛选出训练集数据
metadata = CSV.read(metadata_csv_path, DataFrame)
train_metadata = filter(row -> row.split == "train", metadata)

# 加载图像和标签的函数
function load_pair(image_path, label_path)
    image = load(joinpath(data_dir, image_path))
    label = load(joinpath(data_dir, label_path))
    # 将标签二值化: 背景为0, 道路为1
    label = label .> 0.5 # 假设标签图像是白色道路，黑色背景
    return (Float32.(channelview(image)), Float32.(channelview(label)))
end

# 数据集加载器，返回一个批次的图像和标签
function get_batch(metadata, batch_size)
    indices = randperm(nrow(metadata))[1:batch_size]
    batch_images = []
    batch_labels = []
    for i in indices
        img, lbl = load_pair(metadata[i, :png_image_path], metadata[i, :png_label_path])
        push!(batch_images, img)
        push!(batch_labels, lbl)
    end
    return cat(batch_images..., dims=4), cat(batch_labels..., dims=4)
end

# 数据加载器的参数
batch_size = 8

# 创建DataLoader
train_loader = DataLoader(() -> get_batch(train_metadata, batch_size),
    batchsize=batch_size, shuffle=true)

# 使用DataLoader
for (x, y) in train_loader
    # 在这里进行训练
    # x 是图像批次，y 是标签批次
end