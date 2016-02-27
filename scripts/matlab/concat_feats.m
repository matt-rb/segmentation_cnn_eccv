function [ feats ] = concat_feats( feat_dir )
%% Merge several feat file to unique
%   input:
%       feat_dir : string - direcory of feat files
%   output:
%       feats : merged feat file
%%
    dispstat('','init');
    dispstat('Merging feat files...','timestamp','keepthis');
    feat_list = dir(feat_dir);
    feats = [];
    for i=1:size(feat_list,1)
        if ~feat_list(i).isdir
            dispstat(['Reading ' num2str(i) '/' num2str(size(feat_list,1))]);
            load([ feat_dir '/' feat_list(i).name])
            if (size(feats,2)== size(feats,3))
                x = permute(x, [1 3 2]);
            end
            feats = [ feats , x ];
        end
    end
end

