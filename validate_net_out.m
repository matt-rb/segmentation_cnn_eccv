
clear;
close all;

img_org_name = 'data/moto.jpg';
load('output/moto.jpg.mat');
feat = x;
h_img_size = size(feat,2);
w_img_size = size(feat,3);
bin_size = 8;

%% load fc7 feats and normalize/binrize manualy
load(['output/itq_out/fc7_bin_' num2str(bin_size) '.mat']);
project_mat = pca_mapping * itq_rot_mat;

tt = repmat(mean_fc7(1,:),[h_img_size 1]);

tt2 = repmat(tt,[w_img_size 1]);
tt2 = reshape(tt2,[w_img_size h_img_size 4096]);
tt2 = permute(tt2,[3 2 1]);

data_features=bsxfun(@minus,tt2,feat);
data_features=bsxfun(@rdivide,data_features,sqrt(sum(data_features.^2,2)));
    
c = multiprod(project_mat', data_features);
result = sign(max(c,0));

result = permute(result,[2 3 1]);

%% load hconv8 feats
load('output/batch_test.mat');
feat_net = reshape(x(1,:,:,:),[bin_size h_img_size w_img_size]);
feat_net = sign(max(feat_net,0));
feat_net = permute(feat_net,[2 3 1]);

%%compute differences should be zero
diff = feat_net-result;
sum(sum(nonzeros(feat_net-result)))

img_net = zeros(h_img_size,w_img_size);
img = zeros(h_img_size,w_img_size);
for i=1:h_img_size
    for j=1:w_img_size
        %img_net(i,j) = bi2de( reshape(feat_net(i,j,:),[1 bin_size]), 'left-msb');
        img(i,j) = bi2de( reshape(result(i,j,:),[1 bin_size]), 'left-msb');
        %img(i,j,2) = bi2de( reshape(result(i,j,9:16),[1 8]),  'left-msb');
        %img(i,j,3) = bi2de( reshape(result(i,j,17:24),[1 8]),  'left-msb');
    end
end

imagesc(img)
figure
imagesc(img_net)
