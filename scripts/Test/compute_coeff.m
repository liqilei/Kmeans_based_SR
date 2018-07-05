function coeff = compute_coeff(data_cluster, C)
    num_cluster = size(C, 1);
    num_data = size(data_cluster, 1);
    
    mulcoeff = ones(num_data, 1);
    D = zeros(num_data, num_cluster);    
    for i = 1 : num_cluster
        D(:, i) = pdist2(data_cluster, C(i, :), 'euclidean');
        mulcoeff = mulcoeff.* D(:, i);
    end
    
    weights = zeros(num_data, num_cluster);
    for i = 1 : num_cluster
        weights(:, i) = mulcoeff./ D(:, i);
    end
    
    coeff = zeros(num_cluster, num_data);
    for i = 1 : num_cluster
        coeff(i, :) = (weights(:, i)./sum(weights, 2))';
    end
end