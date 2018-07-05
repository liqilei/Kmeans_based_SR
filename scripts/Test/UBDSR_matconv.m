function im_sr = UBDSR_matconv(im_b, model, num_cluster)

load(model);
im_b = padarray(im_b,[4 4],'replicate','both'); % If we use vl_nnconv padding methods, it will pad the boundary using zero value.

convfea = vl_nnconv(gpuArray(im_b), gpuArray(patch_extraction.weight),  gpuArray(patch_extraction.bias));
convfea = vl_nnrelu(convfea);
convfea = vl_nnconv(convfea, gpuArray(mapping.weight), gpuArray(mapping.bias));
convfea = vl_nnrelu(convfea);
convfea = padarray(convfea,[2 2],'replicate','both');

for i = 1 : num_cluster
    eval(['finalconv','=','reconstruct_', num2str(i-1), ';'])
    im_sr(:,:,i) = vl_nnconv(convfea, gpuArray(finalconv.weight), gpuArray(finalconv.bias));
end
im_sr = gather(im_sr);
end