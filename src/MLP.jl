using Flux

#　import MNIST data
using MLDatasets
train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

# use one-hot encoding
using Flux: onehotbatch
train_y = onehotbatch(train_y, 0:9)
test_y = onehotbatch(test_y, 0:9)

# reshape data
train_x = reshape(train_x, 28 * 28, :)
test_x = reshape(test_x, 28 * 28, :)

# define model
model = Chain(
    Dense(28 * 28, 32, relu),
    Dense(32, 10),
    softmax)

# define loss function
using Flux: crossentropy
loss(x, y) = crossentropy(model(x), y)

# define optimizer
using Flux: Descent
opt = Descent(0.1)

# define accuracy function
using Flux: accuracy
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

# train model
using Flux: @epochs
@epochs 10 Flux.train!(loss, params(model), zip(train_x, train_y), opt)

# test model
accuracy(test_x, test_y)
