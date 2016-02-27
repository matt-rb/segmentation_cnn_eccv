function ng_list = segment_neighbours(segments, seg_label)
%% find neighbours of given segment
%   input : 
%   - segments : labelled matrix of segments
%   - seg_label : label of object to find neighbours of
%   outout:
%   - ng_list : neighbour Labels
%%

    object = segments == seg_label;
    se = ones(5);   % 8-connectivity for neighbours - could be changed
    neighbours = imdilate(object, se) & ~object;
    ng_list = unique(segments(neighbours));

end