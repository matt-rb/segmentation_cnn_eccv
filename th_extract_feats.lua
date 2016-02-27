-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 21/01/16 - 14.55
-- ------------------------------- 
-- Desc : Extract fc6 and fc7 output feats and save to MATLAB mat file
-- -------------------------------

require("scripts.lua.common")
matio = require 'matio'

-- directories setup
image_dir= 'data/pascal'
feats_output_dir= 'output/vgg_feats_fc7'
file_list = scandir(image_dir, '.jpg')
-- input standard alex
-- dis_resize = 227
-- input standard vgg
dis_resize = 224

-- load net
net_conv = torch.load(th_model_full_conv_vgg16_fc7)
-- disable flips, dropouts and batch normalization
net_conv:evaluate()
print ("NET:\n"..model2text(net_conv))

-- Extract features
img_list = torch.Tensor(5,3, dis_resize, dis_resize)
for batch_idx=1,3424 do
    print ("Extract patch No."..batch_idx)
    for img_idx=1,5 do
        img_name = image_dir..'/'..file_list[batch_idx*img_idx]
        img = load_image(img_name, dis_resize)
        img_list[img_idx] = img
        --image.save(file_list[batch_idx*img_idx],img_list[img_idx])
    end
    y_conv = net_conv:forward(img_list:cuda())
    -- save to mat file
    output_mat = feats_output_dir..'/batch_'..batch_idx..'.mat'
    matio.save(output_mat,y_conv:float())
    collectgarbage()
end
print ("Done.")

--x = torch.cat(y_conv,y_conv,1)
