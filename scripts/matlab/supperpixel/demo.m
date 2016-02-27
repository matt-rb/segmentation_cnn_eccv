   im=imread('cat.jpg');
%     for ind=1:3
%         imo(:,:,ind)=im;
%     end
    [l, Am, Sp, d] = slic(im, 100, 10,1.5, 'mean');
   segments =l;
   for ind=1:3
        imo(:,:,ind)=segments;
   end
   figure;imshow(segments, [1 max(max(segments))]);
   %imshow(mat2gray(imo));