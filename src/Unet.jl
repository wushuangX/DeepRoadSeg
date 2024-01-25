# U-net的Flux实现，Flux v0.14.9， julia v1.10.0; 
# 网络结构参考[milesial/Pytorch-UNet: PyTorch implementation of the U-Net for image semantic segmentation with high quality images](https://github.com/milesial/Pytorch-UNet)
using Flux, CUDA
using Flux: @functor

# 定义网络

# 1. 定义网络组件
# 1.1. 卷积块
"""(convolution => [BN] => ReLU) * 2"""
DoubleConv(in_channels, out_channels) = Chain(
    Conv((3, 3), in_channels => out_channels, pad=1, bias=false),
    BatchNorm(out_channels),
    relu,
    Conv((3, 3), out_channels => out_channels, pad=1, bias=false),
    BatchNorm(out_channels),
    relu)


# 1.2. 下采样块
DownBlock(in_channels, out_channels) = Chain(
    MaxPool((2, 2)),
    DoubleConv(in_channels, out_channels),
)

# 1.3. 上采样块
"""上采样块，同时输入编码器的输入实现跳跃连接"""
struct UpBlock
    upsample
end

@functor UpBlock

UpBlock(in_channels::Int, out_channels::Int) =
    UpBlock(Chain(
        ConvTranspose((2, 2), in_channels => out_channels, stride=2, bias=false),
    ))

function (m::UpBlock)(x, bridge)
    x = m.upsample(x)
    # 下采样部分的图像大小是大于上采样部分的，因此需要把bridge的大小缩小，或者把x放大
    # 第一种实现方式，使用切片（copy and crop）
    # bridge_width, bridge_height = size(bridge)[1:2]
    # target_width, target_height = size(x)[1:2]
    # start_x = div(bridge_width - target_width, 2) + 1
    # end_x = start_x + target_width - 1
    # start_y = div(bridge_height - target_height, 2) + 1
    # end_y = start_y + target_height - 1
    # # debug
    # println("bridge_height: $bridge_height, bridge_width: $bridge_width")
    # println("target_height: $target_height, target_width: $target_width")
    # println("start_x: $start_x, end_x: $end_x")
    # println("start_y: $start_y, end_y: $end_y")
    # println("size(x): $(size(x)), size(bridge): $(size(bridge))")
    # bridge = bridge[start_x:end_x, start_y:end_y, :, :]

    # 第二种实现方式，使用pad
    diff_x = size(bridge, 1) - size(x, 1)
    diff_y = size(bridge, 2) - size(x, 2)
    if diff_x == 1
        x = Flux.pad_zeros(x, (0, 1, 0, 0), dims=[1, 2])
    end
    if diff_y == 1
        x = Flux.pad_zeros(x, (0, 0, 0, 1), dims=[1, 2])
    end
    x = Flux.pad_zeros(x, (diff_x ÷ 2, diff_x ÷ 2, diff_y ÷ 2, diff_y ÷ 2), dims=[1, 2])
    return cat(x, bridge, dims=3)
end

# 1.4. 最后一层卷积
OutConv(in_channels::Int, out_channels::Int) =
    Conv((1, 1), in_channels => out_channels, bias=false)

# 2. 定义网络结构
# 2.1. 定义U-net
"""U-Net"""
struct Unet
    conv_down_blocks
    up_blocks
end

@functor Unet

function Unet(channels::Int=1, labels::Int=channels)
    conv_down_blocks = Chain(
        DoubleConv(channels, 64),
        DownBlock(64, 128),
        DownBlock(128, 256),
        DownBlock(256, 512),
        DownBlock(512, 1024),
    )
    up_blocks = Chain(
        UpBlock(1024, 512),
        DoubleConv(1024, 512),
        UpBlock(512, 256),
        DoubleConv(512, 256),
        UpBlock(256, 128),
        DoubleConv(256, 128),
        UpBlock(128, 64),
        DoubleConv(128, 64),
        OutConv(64, labels),
    )
    return Unet(conv_down_blocks, up_blocks)
end

function (m::Unet)(x::AbstractArray)
    op = m.conv_down_blocks[1](x)
    x1 = m.conv_down_blocks[2](op)
    x2 = m.conv_down_blocks[3](x1)
    x3 = m.conv_down_blocks[4](x2)
    x4 = m.conv_down_blocks[5](x3)
    up_x1 = m.up_blocks[1](x4, x3)
    up_x1 = m.up_blocks[2](up_x1)
    up_x2 = m.up_blocks[3](up_x1, x2)
    up_x2 = m.up_blocks[4](up_x2)
    up_x3 = m.up_blocks[5](up_x2, x1)
    up_x3 = m.up_blocks[6](up_x3)
    up_x4 = m.up_blocks[7](up_x3, op)
    up_x4 = m.up_blocks[8](up_x4)
    # print every layer's size
    println("op: $(size(op))")
    println("x1: $(size(x1))")
    println("x2: $(size(x2))")
    println("x3: $(size(x3))")
    println("x4: $(size(x4))")
    println("up_x1: $(size(up_x1))")
    println("up_x2: $(size(up_x2))")
    println("up_x3: $(size(up_x3))")
    println("up_x4: $(size(up_x4))")
    return m.up_blocks[9](up_x4)
end