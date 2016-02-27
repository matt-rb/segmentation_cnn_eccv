
img_name ='Link to Untitled Folder/3063.jpg';
mean_data = load('ilsvrc_2012_mean.mat', 'mean_data');
save_path = 'airplane.jpg';
mean_data = mean_data.mean_data;
mean_data = mean_data(:, :, [3, 2, 1]);
mean_data = permute(mean_data, [2, 1, 3]);
img = imread(img_name);

img2 = normalize_image( img, mean_data, save_path );
figure;
subplot(1,2,1), subimage(img2)
subplot(1,2,2), subimage(img)

