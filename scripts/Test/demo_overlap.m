% =========================================================================
% Test code for Super-Resolution Convolutional Neural Networks (SRCNN)
%
% Reference
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution, 
%   in Proceedings of European Conference on Computer Vision (ECCV), 2014
%
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Image Super-Resolution Using Deep Convolutional Networks,
%   arXiv:1501.00092
%
% Chao Dong
% IE Department, The Chinese University of Hong Kong
% For any question, send email to ndc.forward@gmail.com
% =========================================================================
% Overlapping method
close all;
clear all;

run matconvnet-1.0-beta25/matlab/vl_setupnn;

%% read ground truth image
% im_path = 'Set5/';
im_path = '/home/ruby/SRdata/Set14';
im_dir = dir(fullfile(im_path, '*bmp'));
im_num = length(im_dir);

%% initialization
up_scale = 3;
num_cluster = 2;
model = '/home/ruby/Kmeans_based_SR/experiments/VDSR_in1f64b4_x3/epoch/best_epoch.mat';
% kmeanspath = 'kmeansBest.mat';   % The best model 
kmeanspath = ['/home/ruby/Kmeans_based_SR/datasets/c' num2str(num_cluster) '/kmeans.mat'];
load(kmeanspath);

stride = 1; % same as the convolution stride
size_input = 41;
patchim_b = zeros(size_input, size_input, 1, 1);

%% work on illuminance only
for img_idx = 1:im_num 
    im  = imread(fullfile(im_path, im_dir(img_idx).name));
    fprintf('Processing %s\n', fullfile(im_path, im_dir(img_idx).name));
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

    %% Kmeans and Overlapping computing(This procedure can perform offline)
    fprintf('KMeans...\n');
    count = 0;
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1            
            subim_b = im_b(x : x+size_input-1, y : y+size_input-1);
            count=count+1;
            patchim_b(:, :, 1, count) = subim_b;
        end
    end

    permute_data = permute(patchim_b,[4 1 2 3]);
    data_cluster = reshape(permute_data, size(patchim_b,4), size(patchim_b,1)*size(patchim_b,2));
    coeff = compute_coeff(data_cluster, C)';
    clear data_cluster permute_data patchim_b

    fprintf('Computing coeffMatrix using overlapping...\n');
    count = 0;  % reset this parameters for computing coefficients
    cofMap = zeros(hei, wid, num_cluster);
    cntMap = zeros(hei, wid);

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            count = count+1;
            for i = 1 :num_cluster
                cofMap(x : x+size_input-1, y : y+size_input-1, i) = cofMap(x : x+size_input-1, y : y+size_input-1, i) + coeff(count, i);
            end
            cntMap(x : x+size_input-1, y : y+size_input-1) = cntMap(x : x+size_input-1, y : y+size_input-1) + 1;
        end
    end
    
    for i = 1 :num_cluster
        cofMap(:,:,i) = cofMap(:,:,i)./cntMap;
    end

    %% UBDSR
    tic;
    fprintf('UBDSR...\n');
    % im_sr = UBDSR_matconv(im_b, model, num_cluster); 
    im_sr = VDSR_matconv(im_b, model, num_cluster);
    % im_sr = SRCNN_matconv(im_b, model, num_cluster); 

    im_h = im_sr;
 
%     im_h = im_sr(:,:,1)*0.5 + im_sr(:,:,2)*0.5

%     im_h = (im_sr(:,:,1).*cofMap(:,:,2)) + (im_sr(:,:,2).*cofMap(:,:,1));

%     im_h = 0;
%     for i = 1 :num_cluster
%         im_h = im_h + (im_sr(:,:,i).*cofMap(:,:,i));
%     end
    results(img_idx).time = toc;    

    %% remove border
    im_h = shave(uint8(im_h * 255), [up_scale, up_scale]);
    im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
    im_b = shave(uint8(im_b * 255), [up_scale, up_scale]);

    %% compute PSNR
    psnr_bic = compute_psnr(im_gnd,im_b);
    total_psnr = compute_psnr(im_gnd,im_h);
    ssim_bic = ssim(im_gnd, im_b);
    total_ssim = ssim(im_gnd, im_h);
    
    results(img_idx).psnr_bic = psnr_bic;
    results(img_idx).psnr_ubdsr = total_psnr;
    results(img_idx).ssim_bic = ssim_bic;
    results(img_idx).ssim_ubdsr = total_ssim;

    %% show results
    % fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bic);
    % fprintf('PSNR for UBDSR Reconstruction: %f dB\n', psnr_srcnn);

%     figure, imshow(im_b); title('Bicubic Interpolation');
%     figure, imshow(im_h); title('UBDSR Reconstruction');

    %imwrite(im_b, ['Bicubic Interpolation' '.bmp']);
    %imwrite(im_h, ['SRCNN Reconstruction' '.bmp']);
end

total_psnr = 0;
total_ssim = 0;
for i = 1:im_num
    total_psnr = total_psnr + results(i).psnr_ubdsr;
    total_ssim = total_ssim + results(i).ssim_ubdsr;
end

avg_psnr = total_psnr/im_num;
avg_ssim = total_ssim/im_num;

fprintf('UBDSR average PSNR in %s is %f\n', im_path, avg_psnr);
fprintf('UBDSR average SSIM in %s is %f\n', im_path, avg_ssim);