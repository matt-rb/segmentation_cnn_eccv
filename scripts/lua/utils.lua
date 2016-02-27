-- -------------------------------
-- Created by Mahdyar Ravanbakhsh.
-- Date: 17/01/16 - 18.44
-- ------------------------------- 
-- Desc :
-- -------------------------------
require 'image'
require 'cudnn'

-------------------------------------
-- load an RGB image as batch with specific size.
-- @param image_dir : string - image name.
-- @param batch_size : int - batch size.
-- @param resize_to : int - output image squred size.
-- @output img : resizsed batch image
-------------------------------------
function load_image2batch(image_dir, batch_size, resize_to)
    img = image.load(image_dir)
    dest = torch.Tensor(3, resize_to,resize_to)
    img = image.scale(dest, img)
    img = img:resize(batch_size,3,resize_to,resize_to)
    return img
end

-------------------------------------
-- load an RGB image as batch with specific size.
-- @param image_dir : string - image name.
-- @param resize_to : int - output image squred size.
-- @output img : resizsed batch image
-------------------------------------
function load_image(image_dir, resize_to, sq, sub_min)
    sq = sq or (sq == nil and 1)
    sub_min = sub_min or (sub_min == nil and 0)
    mean = torch.Tensor(3)
    mean[1]=123.68
    mean[2]=116.779
    mean[3]=103.939
    resize_val_x = 0
    resize_val_y = 0
    img = image.load(image_dir)
    if sub_min==1 then
        print 'hi'
        mean = mean:repeatTensor(img:size(3),img:size(2),1)
        mean = mean:transpose(1,3)
        img = img - mean
    end
    if sq==1 then
        resize_val_x = resize_to
        resize_val_y = resize_to
    else
        resize_val_x = img:size(3)*resize_to
        resize_val_y = img:size(2)*resize_to
    end
        dest = torch.Tensor(3, resize_val_y, resize_val_x)
        img = image.scale(dest, img)
    img = img:resize(3,resize_val_y, resize_val_x)
    return img
end


-------------------------------------
-- Convert linear module to 1x1 Convolution
-- you just need to provide the linear module you want to convert,
-- and the dimensions of the field of view of the linear layer
-- example :
-- input = torch.rand(3,6,6)
-- m = nn.Linear(3*6*6,10)
-- mm = convertLinear2Conv1x1(m,{6,6})
-- @param linmodule : linear module.
-- @param in_size : int - dimensions of the field of view.
-- @output convmodule : Convolution module
-------------------------------------
function convert_linear2conv1x1(linmodule,in_size)
   local s_in = linmodule.weight:size(2)/(in_size[1]*in_size[2])
   local s_out = linmodule.weight:size(1)
   local convmodule = cudnn.SpatialConvolution(s_in,s_out,in_size[1],in_size[2],
       1,1)
   convmodule.weight:copy(linmodule.weight:transpose(1,2))
   convmodule.bias:copy(linmodule.bias)
   return convmodule
end


-------------------------------------
-- Compute distance between two 2-D feat arrays.
-- @param x_mat : string - image name.
-- @param y_mat : int - batch size.
-- @param axis : int (1,2) - set "1" for vertical feat vectors, "2" for
-- horizontal. default value is "1"
-- @output dist_list : distance between X,Y. dist [N x 1]
-------------------------------------
function compute_dist(x_mat, y_mat, axis)
    -- set default value for axis and validate input vectors
    axis = axis or (axis == nil and 1)
    if x_mat:dim() > 2 or y_mat:dim() > 2 or axis > 2 or axis < 1 then
        return 0
    end

    samples = 1
    if x_mat:dim() > 1 then
        samples = x_mat:size(axis)
    elseif axis == 2 then
        samples = x_mat:size(1)
    end
    dist_list = torch.zeros(samples)
    for i = 1, samples do
        dist_list[i] =torch.sqrt (torch.sum ((x_mat-y_mat):pow(2)))
    end
    return dist_list
end


-------------------------------------
-- Convert nn net model to string.
-- @param model : torch nn model
-- @output str_model : summary of model as string
-------------------------------------
function model2text(model)
    return model:__tostring__()
end

-------------------------------------
-- Get list of files in a directory.
-- @param directory : string, adress of directory.
-- @param extension : string, file extension i.e ',jpg' '.png'.
-- @output file_list :  array of all the files for the given directory.
-------------------------------------
function scandir(directory, extension)
    require 'lfs'
    local i, file_list, popen = 0, {}, io.popen
    for filename in popen('ls -a "'..directory..'"'):lines() do
        if lfs.attributes(directory..'/'..filename,"mode") == "file" then
            if extension then
                if extension == get_file_extension(filename) then
                    i = i + 1
                    file_list[i] = filename
                end
            else
                i = i + 1
                file_list[i] = filename
            end
        end
    end
    print (table.getn(file_list).." file(s) read.")
    return file_list
end

-------------------------------------
-- Get a file extension.
-- @param filename : string, the name of file
-- @output extension :  string, file extension i.e ',jpg' '.png'.
-------------------------------------
function get_file_extension(filename)
  return filename:match("^.+(%..+)$")
end