-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 17/01/16 - 19.14
-- ------------------------------- 
-- Desc : global require libs and setups
-- -------------------------------

require 'cudnn'
require 'inn'
require 'image'
require 'torch'
require("scripts.lua.utils")
require("scripts.lua.create_models")
-- require("scripts_lua.utils")


dummy_model = 'model/dummy_model.net'
-- common directories and files

th_model_original_alex = 'model/alexnet.net'
th_model_full_conv_992_alex = 'model/alexnet_full_conv_992.net'
th_model_full_conv_1000_alex = 'model/alexnet_full_conv_1000.net'
th_model_full_conv_fc6 = 'model/th_model_full_conv_fc6.net'
th_model_full_conv_fc7 = 'model/th_model_full_conv_fc7.net'
th_model_full_conv_vgg16 = 'model/vgg16_full_conv.net'
th_model_full_conv_vgg16_fc6 = 'model/th_model_full_conv_vgg16_fc6.net'
th_model_full_conv_vgg16_fc7 = 'model/th_model_full_conv_vgg16_fc7.net'

-- binary models alex
th_model_fc7_bin8 = 'model/th_model_fc7_bin8.net'
th_model_fc6_bin8 = 'model/th_model_fc6_bin8.net'
th_model_fc7_bin8_eccv = 'model/th_model_fc7_bin8_eccv.net'

-- fcn-alex Trevor Darrell
th_model_fcn8s = 'model/fcn-8s-pascal.net'
th_model_fcn32s = 'model/fcn32.net'
proto_fcnalex_pascal = 'model/fcn-alexnet-pascal-fc7.prototxt'
model_fcnalex_pascal= 'model/fcn-alexnet-pascal.caffemodel'
th_model_fcnalex_pascal = 'model/fcnalex_pascal.net'
th_model_fcnalex_pascal_fc7 = 'model/fcnalex_pascal_fc7.net'

-- binary models vgg16
th_model_vgg16_fc7_bin8 = 'model/th_model_vgg16_fc7_bin8.net'
th_model_vgg16_fc6_bin8 = 'model/th_model_vgg16_fc6_bin8.net'