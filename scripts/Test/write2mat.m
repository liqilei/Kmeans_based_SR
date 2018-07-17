% =========================================================================
% Transform an images dateset to .mat with scale of [2,3,4]
% =========================================================================

%% initialization
im_path = '/home/qilei/qilei/Set5_bmp';
im_dir = dir(fullfile(im_path, '*bmp'));
im_num = length(im_dir);
%% bicubic interpolation
for img_idx = 1 : im_num
    im  = imread(fullfile(im_path, im_dir(img_idx).name));
    fprintf('Processing %s\n', fullfile(im_path, im_dir(img_idx).name));
    if size(im,3)>1
        im = rgb2ycbcr(im);
        im = im(:, :, 1);
    end
    for up_scale = 2 : 4
        im_gt = modcrop(im, up_scale);
        im_gt_y = double(im_gt);
        [hei,wid] = size(im_gt_y);
        im_l = imresize(im_gt_y, 1/up_scale, 'bicubic');
        im_b_y = imresize(im_l, up_scale, 'bicubic');
        save(['Set5_mat/' im_dir(img_idx).name(1:end-4)  '_x' num2str(up_scale) '.mat'], 'im_b_y', 'im_gt_y');
    end
end