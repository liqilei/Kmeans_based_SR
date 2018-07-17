% =========================================================================
% Test code for the VDSR model
% This code is modified from Super-Resolution Convolutional Neural Networks (SRCNN)
% *importat*
% function im_sr = VDSR_matconv(im_b, model, model_type)
% for model_type option:
%     model_type = 0 : our model with biash
%     model_type = 1 : official model 
%     model_type = 2 : internet model(without bias)
%     model_type = 3 : our model without biash
% =========================================================================
% Overlapping method
close all;
clear all;
dbstop if error
run matconvnet-1.0-beta25/matlab/vl_setupnn;

%% read ground truth image
im_path = '/media/qilei/李启磊18678106384/model/Set5_bmp';
im_dir = dir(fullfile(im_path, '*bmp'));
im_num = length(im_dir);
%% initialization
use_model = 0;
model = '/media/qilei/李启磊18678106384/model/VDSRtotestmatconv/experiments/VDSR_original/epoch/best_epoch.mat';
% use_model = 1;
% model = '/media/qilei/李启磊18678106384/model/VDSRtotestmatconv/experiments/VDSR_original/epoch/VDSR_Official.mat';
% use_model = 2;
% model = '/home/qilei/qilei/VDSRtotestmatconv/experiments/VDSR_original/epoch/best_epoch_50_14.mat';
% use_model = 3;
% model = '/media/qilei/李启磊18678106384/model/07.16/best_epoch.mat';

stride = 1; % same as the convolution stride
size_input = 41;
patchim_b = zeros(size_input, size_input, 1, 1);
for up_scale = 2 : 4
    %% work on illuminance only
    for img_idx = 1:im_num
        im  = imread(fullfile(im_path, im_dir(img_idx).name));
        results(img_idx).name = im_dir(img_idx).name;
        
        if size(im,3)>1
            im = rgb2ycbcr(im);
            im = im(:, :, 1);
        end
        im_gnd = modcrop(im, up_scale);
        im_gnd = single(im_gnd)/255;
        [hei,wid] = size(im_gnd);
        
        %% bicubic interpolation
        im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
        im_b = imresize(im_l, up_scale, 'bicubic');
        %% VDSR
        tic;
        im_sr = VDSR_matconv(im_b, model, use_model); %0 for zhenli, 1 for offcial, 2 for internet, 3 for zheli without bias      
        im_h = im_sr;
        results(img_idx).time = toc;
        
        %% remove border
        im_h = shave(uint8(im_h * 255.0), [up_scale, up_scale]);
        im_gnd = shave(uint8(im_gnd * 255.0), [up_scale, up_scale]);
        im_b = shave(uint8(im_b * 255.0), [up_scale, up_scale]);

        %% compute PSNR
        psnr_bic = compute_psnr(im_gnd,im_b);
        pred_psnr = compute_psnr(im_gnd,im_h);
        ssim_bic = ssim(im_gnd, im_b);
        total_ssim = ssim(im_gnd, im_h);
        results(img_idx).psnr_bic = psnr_bic;
        results(img_idx).psnr_vdsr = pred_psnr;
        results(img_idx).ssim_bic = ssim_bic;
        results(img_idx).ssim_vdsr = total_ssim;
    end
    
    pred_psnr = 0;
    bic_psnr = 0;
    total_ssim = 0;
    for i = 1:im_num
        pred_psnr = pred_psnr + results(i).psnr_vdsr;
        total_ssim = total_ssim + results(i).ssim_vdsr;
        bic_psnr = bic_psnr + results(i).psnr_bic;
    end
    
    pred_psnr = pred_psnr/im_num;
    bic_psnr = bic_psnr/im_num;
    avg_ssim = total_ssim/im_num;
    
    fprintf('up_scale = %d\n', up_scale);
    fprintf('VDSR average perd_PSNR in %s is %f\n', im_path, pred_psnr);
    fprintf('VDSR average bic_PSNR in %s is %f\n', im_path, bic_psnr);
end