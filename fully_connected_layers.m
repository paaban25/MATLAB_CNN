function output = fully_connected_layers (input, weight, bias )
    %Let there be M neurons in this layer, and the input must be flattened
    %of (N X 1) , the output will be of size (M X 1) 
    % output= weight(input)+bias, size of bias is also (M X 1), size of
    % weight is (M X N)

    input = input(:);

    if size(weight, 2) ~= size(input, 1)
        error('Dimension mismatch: Weight matrix and input vector sizes do not match.');
    end

    output= matrix_multiplication (weight,input) + bias;
end