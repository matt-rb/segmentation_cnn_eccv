-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 21/01/16 - 21.05
-- ------------------------------- 
-- Desc :
-- -------------------------------

require("scripts.lua.common")
matio = require 'matio'

-- directories setup
feats_output_dir= 'output/'
input_dir='data/'
img_name = 'moto.jpg'
output_mat = feats_output_dir..img_name..'.mat'
dis_resize = 1
sq=0

-- load net
print ("extract feats to : "..output_mat)
net_conv = torch.load(th_model_fcn_alexnet)
--net_conv = torch.load(th_model_full_conv_vgg16_fc7)
print (net_conv)
-- disable flips, dropouts and batch normalization
net_conv:evaluate()

-- img = image.load(img_name)
img = load_image(input_dir..img_name, dis_resize,sq)
print ("Image size :\n"..tostring(img:size()))

y_conv = net_conv:forward(img)
print ("output size :\n"..tostring(y_conv:size()))
print ("save file")
matio.save(output_mat,y_conv:float())
