function  relation_mat_sp  = get_segment_neighbours( segments )
%% Generate a sparse matrix of segment neighbours
%   input:
%   - segments : labelled matrix of segments
%   output:
%   - relation_mat_sp : relation sparse matrix of neighbours
%%
    seg_lbl = unique(segments);
    temp_rel_mat = zeros(size(seg_lbl,1));
    
    for seg_idx = 1 : size(seg_lbl,1)
        ng_list = segment_neighbours(segments, seg_lbl(seg_idx));
        temp_rel_mat(seg_idx,ng_list)=1;
    end
    relation_mat_sp = sparse(temp_rel_mat);
end

