function [ normalized_fv, mean_data  ] = normalize_features( feature_vectors, mean_data )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if (nargin<2) || isempty(mean_data)
  mean_data = mean(feature_vectors);
end


feature_vectors=bsxfun(@minus,feature_vectors,mean_data);
normalized_fv=bsxfun(@rdivide,feature_vectors,sqrt(sum(feature_vectors.^2,2)));

end

