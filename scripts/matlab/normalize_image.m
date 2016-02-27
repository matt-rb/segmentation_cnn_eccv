function img = normalize_image( img, save_path )
% Get an image, do preprocessing based on alex-net input data
% first resize to 256x256 and subtract mean_data and save with original
% size
%   input:
%   - img : given image MxNx3
%   - mean_data : mean of image net 1x3
%   - save_path : save output file name

mean_data = [123.68 116.779 103.939];
img_org_size= size(img);
mean_data = repmat(mean_data,[img_org_size(1) 1]);
mean_data = repmat(mean_data,[img_org_size(2) 1]);
mean_data = reshape(mean_data,[img_org_size(1) img_org_size(2) 3]);
%im_data = imresize(img, [size(mean_data,1) size(mean_data,1)], 'bilinear');
%im_data = single(im_data);
img = single(img) - mean_data;
%img = imresize(im_data, [img_org_size(1) img_org_size(2)], 'bilinear');
imwrite(img,save_path,'jpg');
end

