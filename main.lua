-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 23/02/16 - 13.34
-- ------------------------------- 
-- Desc :
-- -------------------------------


require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'cudnn'

require("scripts.lua.common")
require("scripts.lua.SmoothBatchPairwiseCosineSimilarity")
require("scripts.lua.SmoothPairwiseCosineSimilarity")

model_file = dummy_model

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

--nClasses = opt.nClasses
input = torch.Tensor(2,3,3):fill(1)
input[1] = input[1]+1;
input[1][2][2] = input[1][2][2]+1;
input[2][2][2] = input[2][2][2]+1;

label = torch.Tensor(9,9):fill(1)

--paths.dofile('scripts.lua.trainutil.lua')

print(opt)

-- 1. load model

model = torch.load(model_file)
--print(model)
--if opt.backend == 'cudnn' then
--      require 'cudnn'
--      cudnn.convert(model, cudnn)
--elseif opt.backend ~= 'nn' then
--      error'Unsupported backend'
--end

print('=> Model')
print(model)

-- 2. Create Criterion
criterion = nn.BCECriterion()
print('=> Criterion')
print(criterion)

-- 3. Convert model to CUDA
print('==> Converting model to CUDA')
-- model is converted to CUDA in the init script itself
-- model = model:cuda()
criterion:cuda()
collectgarbage()



cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)
print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('train.lua')
paths.dofile('test.lua')

epoch = opt.epochNumber

for i=1,opt.nEpochs do
   train()
   test()
   epoch = epoch + 1
end



