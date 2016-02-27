-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 27/01/16 - 23.15
-- ------------------------------- 
-- Desc : Extract fc6 and fc7 output feats for dataset and save to MATLAB mat file
-- -------------------------------

require("scripts.lua.common")
matio = require 'matio'

-- directories setup
image_dir= 'data/pascal'
feats_output_dir= 'output/feat_pascal'
file_list = scandir(image_dir, '.jpg')
-- input standard alex
-- dis_resize = 227
-- input standard vgg
dis_resize = 4
sq=0

-- load net
net_conv = torch.load(th_model_full_conv_fc7)
-- disable flips, dropouts and batch normalization
net_conv:evaluate()
print ("NET:\n"..model2text(net_conv))

-- Extract features
for img_idx=118, table.getn(file_list) do
    print ("Extract patch No."..img_idx..'/'..table.getn(file_list))
    img_name = image_dir..'/'..file_list[img_idx]
    img = load_image(img_name, dis_resize,sq)
    y_conv = net_conv:forward(img:cuda())
    -- save to mat file
    print(tostring(y_conv:size(1))..'x'..tostring(y_conv:size(2))..'x'..tostring(y_conv:size(3)))
    output_mat = feats_output_dir..'/'..file_list[img_idx]..'.mat'
    matio.save(output_mat,y_conv:float())
    collectgarbage()
end
print ("Done.")

--x = torch.cat(y_conv,y_conv,1)
