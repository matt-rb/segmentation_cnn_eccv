function [ itq_bin_mat ] = test_itq( data_features, itq_rot_mat, pca_mapping, normalize_data, mean_data)
% Mapp feature vectors to itq binreies
%
%   input:
%       -data_features : feature vectors
%       -itq_rot_mat : ITQ rotation matrix
%       -pca_mapping : PCA eigen vectors
%
%   output:
%       -itq_bin_mat: feature vectors mapped binaries.

if not(exist('mean_data','var'))
  mean_data = mean(data_features);
end

if not(exist('normalized_data','var'))
    normalize_data= true;
end

if normalize_data
    data_features=bsxfun(@minus,data_features,mean_data);
    data_features=bsxfun(@rdivide,data_features,sqrt(sum(data_features.^2,2)));
end

projected_pca = data_features * pca_mapping;
itq_bin_mat = sign(max(projected_pca * itq_rot_mat,0));

end

