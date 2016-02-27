function [tempo,new_img_cls1, segments_final] = seg_binary_output( segments,img_1,th)
%
%
%     tempo = zeros(size(segments,1),size(segments,2));
%     for segnum=1:max(segments(:))
%         PP=(segments==segnum);
%           D=PP.*img_1;
% 
%           D1=PP.*(sum(D(:))/(nnz(D(:))+eps));
%         tempo=tempo+D1;
%          segg(segnum,2) = (sum(D(:))/(nnz(D(:))+eps));
%         segg(segnum,1) =  uint8(segnum);
%     end
%  imagesc(tempo)
%----------------------
tempo =zeros(size(segments,1),size(segments,2));
for segnum=1:max(segments(:))
    majority=0;
    idx=0;
    PP=(segments==segnum);
    D=PP.*img_1;
    
    idx = unique(nonzeros(unique(D)));
    if(~isempty(idx))
        for idxnum=1:length(idx)
            majority(idxnum)= sum((D(:) ==idx(idxnum)));
        end
        [ss,I]=max(majority(:));
        D1=PP.*idx(I);
        tempo=tempo+D1;
        segg(segnum,2) = idx(I);
        segg(segnum,1) =  segnum;
    else
        segg(segnum,2) = 0;
        segg(segnum,1) =  segnum;
    end
end

%----------------_

%--------------------------------------

new_img_cls1 =segments;
if(length(segg(:,2))>7)
D=pdist(segg(:,2));%%DM2_norm related to orientation
Z = linkage(D);
F = cluster(Z,5);
for jj =1:max(new_img_cls1(:))
new_img_cls1(new_img_cls1==jj)=F(jj);
end
end


%--------------------------------------------
relation_mat_sp  = get_segment_neighbours( segments );
[ ~, segments_final ] = merge_voxel( segments, segg, relation_mat_sp, th,1 );
% imagesc(segments_final)