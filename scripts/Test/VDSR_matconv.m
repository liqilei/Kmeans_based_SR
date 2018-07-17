function im_sr = VDSR_matconv(im_b, model, model_type)

load(model);
im_b = gpuArray(im_b);

if model_type == 1
    for i = 1:20
        model.weight{i} = gpuArray(model.weight{i});
        model.bias{i} = gpuArray(model.bias{i});
    end
    weight = model.weight;
    bias = model.bias;
    layer_num = size(weight, 2);
    % x = padarray(im_b, [1 1], 'replicate', 'both');
    convfea = vl_nnconv(im_b, weight{1}, bias{1}, 'Pad', 1);
    for i = 2 : layer_num
        convfea = vl_nnrelu(convfea);
        % convfea = padarray(convfea, [1 1], 'replicate', 'both');
        convfea = vl_nnconv(convfea, weight{i}, bias{i}, 'Pad', 1);
    end
    im_sr= convfea + im_b;
    
elseif (model_type == 0)
    
    % x = padarray(im_b, [1 1],'replicate','both'); % If we use vl_nnconv padding methods, it will pad the boundary using zero value.
    convfea = vl_nnconv(gpuArray(im_b), gpuArray(conv_in.weight),  gpuArray(conv_in.bias), 'Pad', 1);
    convfea = vl_nnrelu(convfea);
    % convfea = padarray(convfea,[1 1],'replicate','both');
    
    for i = 1 : 18
        eval(['conv','=','conv_', num2str(i-1), ';'])
        convfea = vl_nnconv(convfea, gpuArray(conv.weight), gpuArray(conv.bias), 'Pad', 1);
        convfea = vl_nnrelu(convfea);
        % convfea = padarray(convfea, [1 1], 'replicate', 'both');
    end
    
    res = vl_nnconv(convfea, gpuArray(conv_out.weight), gpuArray(conv_out.bias), 'Pad', 1);
    im_sr = res + im_b;
    
elseif (model_type == 2)
    
    % x = padarray(im_b, [1 1],'replicate','both'); % If we use vl_nnconv padding methods, it will pad the boundary using zero value.
    convfea = vl_nnconv(gpuArray(im_b), gpuArray(input_weight.weight),  gpuArray([]), 'Pad', 1);
    convfea = vl_nnrelu(convfea);
    % convfea = padarray(convfea,[1 1],'replicate','both');
    
    for i = 1 : 18
        eval(['conv','=','residual_layer_', num2str(i-1), '_conv_weight;'])
        convfea = vl_nnconv(convfea, gpuArray(conv.weight), gpuArray([]), 'Pad', 1);
        convfea = vl_nnrelu(convfea);
        % convfea = padarray(convfea, [1 1], 'replicate', 'both');
    end
    
    res = vl_nnconv(convfea, gpuArray(output_weight.weight), gpuArray([]), 'Pad', 1);
    im_sr = res + im_b;
    
    
    
elseif (model_type == 3)
    
    % x = padarray(im_b, [1 1],'replicate','both'); % If we use vl_nnconv padding methods, it will pad the boundary using zero value.
    convfea = vl_nnconv(gpuArray(im_b), gpuArray(conv_in.weight),  gpuArray([]), 'Pad', 1);
    convfea = vl_nnrelu(convfea);
    % convfea = padarray(convfea,[1 1],'replicate','both');
    
    for i = 1 : 18
        eval(['conv','=','conv_', num2str(i-1), ';'])
        convfea = vl_nnconv(convfea, gpuArray(conv.weight), gpuArray([]), 'Pad', 1);
        convfea = vl_nnrelu(convfea);
        % convfea = padarray(convfea, [1 1], 'replicate', 'both');
    end
    
    res = vl_nnconv(convfea, gpuArray(conv_out.weight), gpuArray([]), 'Pad', 1);
    im_sr = res + im_b;
    
end

im_sr = gather(im_sr);