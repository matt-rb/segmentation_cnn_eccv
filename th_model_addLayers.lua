-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 18/01/16 - 19.01
-- ------------------------------- 
-- Desc : Add Binary quantization and similarity Layers
-- make a new model "th_model_fc7_bin8_eccv" from "th_model_full_conv_fc7"
-- -------------------------------

require("scripts.lua.common")
require("scripts.lua.DepthBatchNormalization")
require("scripts.lua.SmoothPairwiseCosineSimilarity")
matio = require 'matio'

-- setup parameters
input_dir='data/'
img_name = 'hava.jpg'
output_dir='output/'
dis_resize = 1
sq=0
result_text_file = 'output/seurgury_report.txt'
bin_size=8
mat_file='output/itq_out/fc7_bin_'..bin_size..'.mat'

-- load files and model
img = load_image(input_dir..img_name, dis_resize,sq)
img_list = torch.Tensor(2,3, img:size(2), img:size(3))
img_list[1] = img
img_list[2] = img
print ("Image size :\n"..tostring(img:size()))

mean_fc7 = matio.load(mat_file,'mean_fc7')
project_mat = matio.load(mat_file,'project_mat')
net_conv = torch.load(th_model_full_conv_fc7)

net_conv:evaluate()
print ("1 - Original_net test")
y_conv = net_conv:forward(img_list:cuda())
print ("output size :\n"..tostring(y_conv:size()))

print("Saving output to : ", output_dir..'hava.jpg.mat' , "\n")
output_mat = output_dir..'hava.jpg.mat'
matio.save(output_mat,y_conv[1]:float())


-- add depth batchNormalization layer to model th_model_full_conv_fc7
print("add spatial batchNormalization layer to model th_model_full_conv_fc7... \n")
dnormmodule = cudnn.SpatialBatchNormalization(4096,false)
dnormmodule.running_mean = mean_fc7[1]
std=torch.Tensor(mean_fc7:size(2)):zero()+1
dnormmodule.running_std = std:cuda()
--print(hconvmodule.weight = project_mat:transpose(1,2)
dnormmodule.name = 'dpnorm'
dnormmodule:cuda()
net_conv:add(dnormmodule)

print ("2 - spatial batchNormalization test")
y_conv = net_conv:forward(img_list:cuda())
print ("output size :\n"..tostring(y_conv:size()))
--print(tostring(dnormmodule.running_mean))


-- add binary quantization layer
convmodule = cudnn.SpatialConvolution(4096, bin_size, 1, 1, 1, 1, 0, 0, 1)
convmodule.weight:copy(project_mat:transpose(1,2))
convmodule.name = 'hconv_8'
convmodule:cuda()
net_conv:add(convmodule)

print ("3 - binary quantization layer test ")
y_conv = net_conv:forward(img_list:cuda())
print ("output size :\n"..tostring(y_conv:size()))


-- add cosine similarity

branches = nn.ConcatTable()
branches:add(nn.Identity())
branches:add(nn.Identity())
branches:cuda()
net_conv:add(branches)

--cosimilaritymodoul = nn.SmoothPairwiseCosineSimilarity()
--cosimilaritymodoul:cuda()
--net_conv:add(cosimilaritymodoul)

print ("4 - SmoothPairwiseCosineSimilarity layer test ")
y_conv = net_conv:forward(img_list:cuda())
print (y_conv)

--Add sign layer
--print("Add sign layer to the... \n")
--sofmaxmodule = cudnn.ReLU()
--sofmaxmodule:cuda()
--net_conv:add(sofmaxmodule)



--
--print ("Image size :\n"..tostring(img:size()))
--net_conv:evaluate()
--print("HI")
--
--y_conv = net_conv:forward(img_list:cuda())
--print("HI2")
--print ("output size :\n"..tostring(y_conv:size()))
--
print("Saving output to : ", output_dir..'batch_test.mat' , "\n")
output_mat = output_dir..'batch_test.mat'
matio.save(output_mat,y_conv:float())


print("Saving 'th_model_fc7_bin8_eccv' model: \n", net_conv)
torch.save(th_model_fc7_bin8_eccv,net_conv)

-- report summary
res = "New model 'th_model_fc7_bin8_eccv' is : \n"..model2text(net_conv).."\n"
torch.save(result_text_file, res,'ascii')

