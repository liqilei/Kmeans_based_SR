clear;close all;
%% settings
folder = '../datasets/Train_Image';

size_input = 33;
size_label = 21;
scale = 3;
stride = 14;
num_cluster = 2;

savefolder = ['../datasets/H5Data/x' num2str(scale)];

if exist(savefolder, 'dir')
   fprintf('Warning: replacing existing dir %s \n', savefolder);
   rmdir(savefolder, 's');
end 
mkdir(savefolder);

savetrainpath = [savefolder '/trainc' num2str(num_cluster) '.h5'];
savecoeffpath = [savefolder '/traincofc' num2str(num_cluster) '.h5'];
savekmeans = [savefolder '/kmeansc' num2str(num_cluster) '.mat'];

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
filepaths = [dir(fullfile(folder,'*.bmp'));dir(fullfile(folder,'*.jpg'));dir(fullfile(folder,'*.png'))];
    
for i = 1 : length(filepaths)
    fprintf('Generating Data...picture_no:%d\n',i);
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
    
    im_label = modcrop(image, scale);
    [hei,wid] = size(im_label);
    
    % Gaussian Processing then downsampling and unsampling
    % gaussian_filter = fspecial('gaussian', [3 3], 1.6);
    % im_blur = imfilter(im_label, gaussian_filter, 'replicate');
    % im_input = imresize(imresize(im_blur, 1/scale,'bicubic'),[hei,wid],'bicubic');
    
    % Direct Downsampling and Unsampling
     im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            % subim_bic = im_input(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
            subim_gt = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
            count = count+1;
            data(:, :, 1, count) = subim_input;
            label(:, :, 1, count) = subim_gt;
        end
    end
end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% cluster data... 
fprintf('Clustering....\n');
permute_data = permute(data,[4 1 2 3]);
data_cluster = reshape(permute_data, size(data,4), size(data,1)*size(data,2));
[idx, C] = kmeans(data_cluster, num_cluster);
% tic;
% for i = 1 : count 
%     dist1=sqrt(sum((data_cluster(i,:)-C(1,:)).^2));
%     dist2=sqrt(sum((data_cluster(i,:)-C(2,:)).^2));
%     coeff(1,i)=dist2/(dist1+dist2);
%     coeff(2,i)=dist1/(dist1+dist2); 
% end
% toc;
coeff = compute_coeff(data_cluster, C);
clear data_cluster permute_data

%% writing kmeans_center into Mat & HDF5
save(savekmeans,'C');
% h5create(savekmeans,'/data',[2 size_input*size_input],'Datatype', 'single');
% h5write(savekmeans,'/data',single(C), [1,1], size(C));

%% writing coeff into HDF5 
chunksz = 128;
totalct = 0;
if exist(savecoeffpath, 'file')
   fprintf('Warning: replacing existing file %s \n', savecoeffpath);
   delete(savecoeffpath);
end  

for i = 1 : num_cluster
   h5create(savecoeffpath,['/coeff' num2str(i)],[1 Inf],'Datatype', 'single','ChunkSize',[1 chunksz]);
end

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    for i = 1 : num_cluster
       batchcoeff(1,:,i)= coeff(i,last_read+1:last_read+chunksz);    
    end
    startloc = struct('dat',[1 totalct+1]);
    for i = 1 : num_cluster
       h5write(savecoeffpath,['/coeff' num2str(i)],single(batchcoeff(:,:,i)), startloc.dat, size(batchcoeff(:,:,i)));    
    end
    info=h5info(savecoeffpath);
    curr_dat_sz=info.Datasets(1).Dataspace.Size;    
    
    totalct = curr_dat_sz(end);
end
h5disp(savecoeffpath);

%% writing training data to HDF5
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savetrainpath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savetrainpath);
