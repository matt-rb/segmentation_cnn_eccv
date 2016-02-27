function [result]= seg_evaluation(GT,imgseg)

for gtnum=1:length((unique(nonzeros(GT))))
    idx = (unique(nonzeros(GT)));
    gt=(GT==idx(gtnum));
    D=uint8(gt).*imgseg;
    for ii =1:length(nonzeros(unique(D)))
        indexseg= nonzeros(unique(D));
        sumGT_elm_overlap = (imgseg==indexseg(ii));
        %         imagesc(sumGT_elm_overlap)
        [Jaccard,Dice,rfp,rfn]=seg_evaluate(gt,sumGT_elm_overlap);
        result.acc_gt_el(1,ii) = Jaccard;
        result.acc_gt_el(2,ii) = rfp;
        result.acc_gt_el(3,ii) = rfn;
    end
    [m,I] = max(result.acc_gt_el(1,:));
    result.acc_seg(gtnum) = max(result.acc_gt_el(1,I));
    result.rfp1(gtnum) = max(result.acc_gt_el(2,I));
    result.rfn1(gtnum) = max(result.acc_gt_el(3,I));
    
    
    
end

result.acc_img = mean(result.acc_seg);
result.rfp = mean(result.rfp1);
result.rfn =  mean(result.rfn1);
