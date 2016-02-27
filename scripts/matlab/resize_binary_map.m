function new_bin_map = resize_binary_map( src_image, bin_map )

size_img_org = size(src_image);
size_bin_map = size(bin_map);


x_r = round(size_img_org(2)/size_bin_map(2));
y_r = round(size_img_org(1)/size_bin_map(1));

if length(size_bin_map) == 2
    new_bin_map = zeros(max(size_img_org(1),y_r*size_bin_map(1)),max(size_img_org(2),x_r*size_bin_map(2)));
else
    new_bin_map = zeros(max(size_img_org(1),y_r*size_bin_map(1)),max(size_img_org(2),x_r*size_bin_map(2)),3);
end

for y_idx=1:size_bin_map(1)
    for x_idx=1:size_bin_map(2)
        if length(size_bin_map) == 2
            new_bin_map(y_r*(y_idx-1)+1:y_idx*y_r, x_r*(x_idx-1)+1:x_idx*x_r)... 
            = ones(y_r,x_r)*bin_map(y_idx,x_idx);
        else
            tt = repmat(bin_map(y_idx,x_idx,:),[x_r 1]);
            tt = repmat(tt,[y_r 1]);
            tt = reshape(tt,[y_r x_r 3]);
            new_bin_map(y_r*(y_idx-1)+1:y_idx*y_r, x_r*(x_idx-1)+1:x_idx*x_r,:)... 
            = tt; 
        end
    end
end

if length(size_bin_map) == 2
    new_bin_map = new_bin_map(1:size_img_org(1),1:size_img_org(2));
else
    new_bin_map = new_bin_map(1:size_img_org(1),1:size_img_org(2),:);
end
end

