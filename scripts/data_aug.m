clear; close all;
folder = '../datasets/Train_Image';
savepath = '../datasets/Train_Image_aug/';

if ~exist(savepath, 'dir')
    mkdir(savepath);
end

filepaths = [dir(fullfile(folder, '*.jpg'));dir(fullfile(folder, '*.bmp'))];
     
for i = 1 : length(filepaths)
    filename = filepaths(i).name;
    fprintf('No.%d -- Processing %s\n', i, fullfile(folder, filename));
    [add, im_name, type] = fileparts(filepaths(i).name);
    image = imread(fullfile(folder, filename));
    
    for angle = 0: 1 : 3
        im_rot = rot90(image, angle);
        imwrite(im_rot, [savepath im_name, '_rot' num2str(angle*90) '.bmp']);    
    end
    
    for flip_dim = 1 : 2
        im_flip = flip(image, flip_dim);
        imwrite(im_flip, [savepath im_name, '_f' num2str(flip_dim) '.bmp']);
    end
end