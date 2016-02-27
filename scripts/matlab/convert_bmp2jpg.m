function convert_bmp2jpg( image_dir )
%CONVERT_BMP2JPG Summary of this function goes here
%   Detailed explanation goes here
    dispstat('','init');
    dispstat('Converting BMP images to JPG...','keepthis');
    img_list = dir(image_dir);
    
    for i=1:size(img_list,1)
        if ~img_list(i).isdir
            dispstat(['Reading ' num2str(i) '/' num2str(size(img_list,1))]);
            temp=imread([ image_dir '/' img_list(i).name],'bmp');
            imwrite(temp,[ image_dir '/jpg/' img_list(i).name '.jpg'],'jpg');
        end
    end
    
    
end

