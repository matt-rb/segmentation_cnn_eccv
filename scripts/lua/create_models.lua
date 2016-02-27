-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 18/01/16 - 21.16
-- ------------------------------- 
-- Desc : Creat torch models functions
-- -------------------------------

-------------------------------------
-- converting nn module to nn.DataParallelTable.
-- @param model : torch nn model
-- @param nGPU : number of GPUs
-- @output model : parallel model
-------------------------------------
function makeDataParallel(model, nGPU)
    if nGPU > 1 then
        print('converting module to nn.DataParallelTable')
        assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
        local model_single = model
        model = nn.DataParallelTable(1)
        for i=1, nGPU do
            cutorch.setDevice(i)
            model:add(model_single:clone():cuda(), i)
        end
        cutorch.setDevice(opt.GPU)
    end
    return model
end

-------------------------------------
-- Generate Standard Alexnet model for Torch.
-- model is a torch replication of the model described in the AlexNet nips 2012:
-- http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
-- and correspond caffe model:
-- https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
-- @param nGPU : number of GPUs (if needs parallel)
-- @output model : alexnet model
-------------------------------------
function create_std_alex(nGPU)
    require 'cudnn'
    local model = nn.Sequential()
    model:add(cudnn.SpatialConvolution(3, 96, 11, 11, 4, 4, 0, 0, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
    model:add(cudnn.SpatialMaxPooling(3,3,2,2))
    model:add(cudnn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2, 2))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
    model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    model:add(cudnn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1, 2))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1, 2))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    if nGPU then
        model:cuda()
        model = makeDataParallel(model, nGPU)
    end
    --classifier
    model:add(nn.View(256*6*6))
    model:add(nn.Linear(256*6*6, 4096))
    model:add(cudnn.ReLU(true))
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(4096, 4096))
    model:add(cudnn.ReLU(true))
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(4096, 1000))
    model:add(cudnn.LogSoftMax())

    model:cuda()
    return model
end

-------------------------------------
-- Generate fully Convolutional Alexnet model for Torch.
-- @param nGPU : number of GPUs (if needs parallel)
-- @output model : alexnet_fullconv model
-------------------------------------
function create_fullconv_alex(nGPU)
    require 'cudnn'
    local model = nn.Sequential()
    model:add(cudnn.SpatialConvolution(3, 96, 11, 11, 4, 4, 0, 0, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
    model:add(cudnn.SpatialMaxPooling(3,3,2,2))
    model:add(cudnn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2, 2))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialCrossMapLRN(5, 0.000100, 0.7500, 1.000000))
    model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    model:add(cudnn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1, 1))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1, 2))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1, 2))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 0, 0):ceil())
    model:add(cudnn.SpatialConvolution(256, 4096, 6, 6, 1, 1, 0, 0, 1))
    model:add(cudnn.ReLU(true))
    model:add(nn.Dropout(0.500000))
    model:add(cudnn.SpatialConvolution(4096, 4096, 1, 1, 1, 1, 0, 0, 1))
    model:add(cudnn.ReLU(true))
    model:add(nn.Dropout(0.500000))
    model:add(cudnn.SpatialConvolution(4096, 1000, 1, 1, 1, 1, 0, 0, 1))
    model:cuda()
    if nGPU then
        model = makeDataParallel(model, nGPU)
    end
    return model
end
