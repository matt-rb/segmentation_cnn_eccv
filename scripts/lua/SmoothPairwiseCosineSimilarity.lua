--[[
Input: a table of two inputs {M, k}, where
  M = an n-by-d matrix
  k = an m-by-d matrix
Output: a n-by-m matrix
Each element is an approximation of the cosine similarity between  a row in k and the 
corresponding row of M. It's an approximation since we add a constant to the
denominator of the cosine similarity function to remove the singularity when
one of the inputs is zero. 
--]]

local SmoothPairwiseCosineSimilarity, parent = torch.class('nn.SmoothPairwiseCosineSimilarity', 'nn.Module')

function SmoothPairwiseCosineSimilarity:__init(smoothen)
  parent.__init(self)
  --self.gradInput = {}
  self.smooth = smoothen or 1e-5
end

function SmoothPairwiseCosineSimilarity:updateOutput(input)
  --local M, k = unpack(input)
  self.gradInput:resizeAs(input)
--   assert(M:size(2)==k:size(2),"ERROR: dimensions are not equal !!!")
--  self.rownorms = torch.cmul(M, M):sum(2):sqrt():view(M:size(1))
--  self.colnorms = torch.cmul(k, k):sum(2):sqrt():view(k:size(1))
--  self.rowcol = torch.ger(self.rownorms,self.colnorms);
--  self.dot = M * (k:t());
--  self.output:set(torch.cdiv(self.dot, self.rowcol + self.smooth))
  self.rownorms = torch.cmul(input, input):sum(2):sqrt():view(input:size(1))
  self.colnorms = self.rownorms
  self.rowcol = torch.ger(self.rownorms,self.colnorms);
  self.dot = input * (input:t());
  self.output:set(torch.cdiv(self.dot, self.rowcol + self.smooth))
  print ("gradinput")
  print (self.gradInput)
  print ("input")
  print (input)
  return self.output
end

function SmoothPairwiseCosineSimilarity:updateGradInput(input, gradOutput)
  --local M, k = unpack(input)
  local nrow = input:size(1);
  local ncol = input:size(1);
  local ndim = input:size(2);
  print ("gradOutput")
  print(gradOutput)


  self.gradInput = self.gradInput or input.new()
  --self.gradInput[2] = self.gradInput[2] or input[2].new()
  

  -- M gradient
  self.gradInput:set(torch.cdiv(gradOutput, self.rowcol + self.smooth)*input)
    local scale = torch.cmul(self.output, (torch.repeatTensor(self.colnorms,nrow,1)))
      :cdiv(self.rowcol + self.smooth)
      :cmul(gradOutput):sum(2)
      :cdiv(self.rownorms+self.smooth)
    self.gradInput:add(torch.cmul(-torch.repeatTensor(scale,1,ndim), input))

--  -- k gradient
--  self.gradInput[2]:set(torch.cdiv(gradOutput, self.rowcol + self.smooth):t()* M)
--    local scale = torch.cmul(self.output, (torch.repeatTensor(self.rownorms,ncol,1):t()))
--      :cdiv(self.rowcol + self.smooth)
--      :cmul(gradOutput):sum(1)
--      :cdiv(self.colnorms+self.smooth)
--    self.gradInput[2]:add(torch.cmul(-torch.repeatTensor(scale,ndim,1):t(), k))

  print ("gradinput")
  print (self.gradInput)
  return self.gradInput
end