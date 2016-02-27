function [ merged_list, segments ] = merge_voxel( segments, segments_val, relation_mat_sp, th, merge_one_ng )
%% merge segments based on given threshold and segments map
%   input:
%   - segments : segment map, same size as image include voxel index per
%   pixel. output of superpixel
%   - segments_val : 2 dimention matrix [index_of_voxel value_of_voxel]
%   - relation_mat_sp : sparse relation matrix. (Am) output of superpixel
%   - th : threshold to merge voxels.
%   - merge_one_ng : merge voxels with one nighber to the nighber
%   output:
%   - merged_list : list of voxels and merged
%   - segments : updates merged segments map
%%
relation_mat = full(relation_mat_sp);
merged_list=segments_val(:,1);
segments_val_bin = zeros(size(segments_val,1),9);
segments_val_bin(:,1) = segments_val(:,1);
segments_val_bin(:,2:end) = de2bi( segments_val(:,2),8, 'left-msb');
for i=1: size(segments_val,1)
    if(length(find(relation_mat(i,:)))==1 && merge_one_ng)
        ng = find(relation_mat(i,:));
        merged_list(i)=merged_list(ng);
        segments(segments(:)==i) = ng;
    else
        nighbers = find(relation_mat(i,i:end))+i-1;
        for n=1:length(nighbers)
            %if abs(segments_val(i,2)-segments_val(nighbers(n),2))<th
             if pdist([segments_val_bin(i,2:end) ; segments_val_bin(nighbers(n),2:end)],'minkowski',1)<th
                src = min (merged_list(i), nighbers(n));
                des = max (merged_list(i), nighbers(n));
                merged_list(des)=merged_list(src);
                segments(segments(:)==des) = src;
            end
        end
    end
    
end
end

