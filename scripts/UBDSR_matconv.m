function im_sr = UBDSR_matconv(im_b, model, num_cluster)

load(model);

%model.weight_1{3} = reshape(model.weight_1{3}, size(model.weight_1{3}, 1), size(model.weight_1{3}, 1), size(model.weight_1{3}, 3), 1);
%model.weight_2{3} = reshape(model.weight_2{3}, size(model.weight_2{3}, 1), size(model.weight_2{3}, 1), size(model.weight_2{3}, 3), 1);
im_b = padarray(im_b,[4 4],'replicate','both'); % If we use vl_nnconv padding methods, it will pad the boundary using zero value.

convfea = vl_nnconv(im_b, model.weight{1}, model.bias{1});
convfea = vl_nnrelu(convfea);
convfea = vl_nnconv(convfea, model.weight{2}, model.bias{2});
convfea = vl_nnrelu(convfea);
convfea = padarray(convfea,[2 2],'replicate','both');

for i = 1 :num_cluster
    im_sr(:,:,i) = vl_nnconv(convfea, model.weight{3}{i}, model.bias{3}{i});
end

end