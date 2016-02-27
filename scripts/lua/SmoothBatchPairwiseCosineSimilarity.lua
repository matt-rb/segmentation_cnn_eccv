--[[
Input: a Tensor of batch inputs NxMxD, where
  N = Batch size
  MxD = Size of input matrix
Output: a M-by-M matrix
Each element is an approximation of the cosine similarity between rows in the
 matrix. It's an approximation since we add a constant to the
denominator of the cosine similarity function to remove the singularity when
one of the inputs is zero. 
--]]

local SmoothBatchPairwiseCosineSimilarity, parent = torch.class('nn.SmoothBatchPairwiseCosineSimilarity', 'nn.Module')

function SmoothBatchPairwiseCosineSimilarity:__init(smoothen)
  parent.__init(self)
  self.smooth = smoothen or 1e-5
end

function SmoothBatchPairwiseCosineSimilarity:updateOutput(input)

  self.gradInput:resizeAs(input)
  assert(input:dim() == 3, 'only mini-batch supported (3D tensor), got '
             .. input:dim() .. 'D tensor instead')
  local nBatch = input:size(1)
  self.output = torch.Tensor(nBatch,input:size(2),input:size(2)):fill(0)
  for sampleNo=1, nBatch do
    self.rownorms = torch.cmul(input[sampleNo], input[sampleNo]):sum(2):sqrt():view(input[sampleNo]:size(1))
    self.colnorms = self.rownorms
    self.rowcol = torch.ger(self.rownorms,self.colnorms);
    self.dot = input[sampleNo] * (input[sampleNo]:t());
    self.output[sampleNo] = torch.cdiv(self.dot, self.rowcol + self.smooth)
  end
  return self.output
end

function SmoothBatchPairwiseCosineSimilarity:updateGradInput(input, gradOutput)
  assert(input:dim() == 3, 'only mini-batch supported (3D tensor), got '
             .. input:dim() .. 'D tensor instead')
  local nBatch = input:size(1)

 for sampleNo=1, nBatch do
    local nrow = input[sampleNo]:size(1);
    local ndim = input[sampleNo]:size(2);
  
    self.gradInput[sampleNo] = self.gradInput[sampleNo] or input[sampleNo].new()
    -- M gradient
    self.gradInput[sampleNo] = torch.cdiv(gradOutput[sampleNo], self.rowcol + self.smooth)*input[sampleNo]
    local scale = torch.cmul(self.output[sampleNo], (torch.repeatTensor(self.colnorms,nrow,1)))
      :cdiv(self.rowcol + self.smooth)
      :cmul(gradOutput[sampleNo]):sum(2)
      :cdiv(self.rownorms+self.smooth)
    self.gradInput[sampleNo]:add(torch.cmul(-torch.repeatTensor(scale,1,ndim), input[sampleNo]))
  end
  return self.gradInput
end