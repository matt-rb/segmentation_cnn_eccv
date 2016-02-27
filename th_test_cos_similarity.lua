-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 21/02/16 - 21.46
-- ------------------------------- 
-- Desc :
-- -------------------------------
require 'optim'
require("scripts.lua.common")
require("scripts.lua.SmoothBatchPairwiseCosineSimilarity")
require("scripts.lua.SmoothPairwiseCosineSimilarity")

net = nn.Sequential()

reshapemodule = nn.Reshape(2, 9)
net:add(reshapemodule)
transmodule = nn.Transpose({2,3})
--transmodule = nn.Transpose({1,2})
net:add(transmodule)

branches = nn.ConcatTable()
branches:add(nn.Identity())
branches:add(nn.Identity())
--net:add(branches)

--print(net)
cosimilaritymodoul = nn.SmoothBatchPairwiseCosineSimilarity()
--cosimilaritymodoul = nn.SmoothPairwiseCosineSimilarity()
net:add(cosimilaritymodoul)

input = torch.Tensor(2,3,3):fill(1)
--input[1] = input[1]+1;
input[1][2][2] = input[1][2][2]+1;
input[2][2][2] = input[2][2][2]+1;
print(input)

batch_input = torch.Tensor(4,2,3,3)
batch_input[1] = input
batch_input[2]=input
batch_input[3]=input
batch_input[4]=input

--print(batch_input)
torch.save(dummy_model, net)
--pred = net:forward(batch_input)

--print(pred)

targets = torch.Tensor(4,9,9):fill(1)
target = torch.Tensor(9,9):fill(1)
--target[1][2]=0
--target[2][1]=0
targets[1]=target
targets[2]=target
targets[3]=target
targets[4]=target

criterion = nn.BCECriterion()
--print (target)
net:training()
local parameters, gradParameters = net:getParameters()
--for i = 1,10 do
--    print("forward: "..i)
--    print(criterion:forward(net:forward(batch_input), targets))
--    net:zeroGradParameters()
--    print("backward: "..i)
--    net:backward(batch_input, criterion:backward(net.output, targets))
--    net:updateParameters(0.01)
--end
optimState = {
         learningRate = 0.01,
         learningRateDecay = 0.0,
         dampening = 0.0,
}

print (gradParameters)
local err, outputs
for i = 1,10 do
   feval = function(x)
      net:zeroGradParameters()
      outputs = net:forward(batch_input)
      err = criterion:forward(outputs, targets)
      print(err)
      local gradOutputs = criterion:backward(outputs, targets)
      net:backward(batch_input, gradOutputs)
      return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)
end
print(err)