function [W_fc, b_fc, W_conv, b_conv] = backpropagation(X, Y_true, W_fc, b_fc, W_conv, b_conv, A_conv, A_pool, A_fc, alpha)
    % ---- INPUT DIMENSIONS ----
    % X      : (H_in, W_in, C_in)         -> Input image (height, width, channels)
    % Y_true : (num_classes, 1)           -> One-hot encoded ground truth
    % W_fc   : (num_classes, fc_dim)      -> Weights for Fully Connected layer
    % b_fc   : (num_classes, 1)           -> Bias for Fully Connected layer
    % W_conv : (fH, fW, C_in, num_filters)-> Conv layer weights
    % b_conv : (1, num_filters)           -> Conv layer biases
    % A_conv : (H_c, W_c, num_filters)    -> Activation after convolution
    % A_pool : (H_p, W_p, num_filters)    -> Activation after pooling
    % A_fc   : (fc_dim, 1)                -> Flattened vector passed to FC layer
    % alpha  : Scalar                     -> Learning rate
    
    % ---- STEP 1: Compute Softmax Output ----
    % Z_out  : (num_classes, 1)           -> FC output before softmax
    % Z_out = W_fc * A_fc + b_fc; 
    A_fc = A_fc(:);
    Z_out = matrix_multiplication(W_fc,A_fc) + b_fc;
    Y_pred = softmax(Z_out); % (num_classes, 1)
    
    % ---- STEP 2: Compute Gradient at Output Layer ----
    % delta_out : (num_classes, 1) -> Gradient of loss w.r.t. output layer
    delta_out = Y_pred - Y_true;  
    
    % Compute Gradients for Fully Connected Layer
    % dW_fc  : (num_classes, fc_dim)
    % dW_fc = delta_out * A_fc';  
    dW_fc = matrix_multiplication (delta_out,A_fc');
    
    % db_fc  : (num_classes, 1)
    db_fc = sum(delta_out, 2);  
    
    % ---- STEP 3: Backpropagate to Pooling Layer ----
    % delta_pool : (H_p, W_p, num_filters) -> Backpropagated gradient to pooling layer
    % delta_pool = reshape(W_fc' * delta_out, size(A_pool));  
    delta_pool = reshape(matrix_multiplication(W_fc' , delta_out), size(A_pool)); 
    
    % ---- STEP 4: Backpropagate Through Pooling Layer ----
    % delta_conv : (H_c, W_c, num_filters) -> Gradient for convolution layer
    delta_conv = zeros(size(A_conv));  
    
    [m, n, num_filters] = size(A_pool); % H_p = m, W_p = n
    
    pool_size = 2; % Assuming 2×2 pooling
    
    for f = 1:num_filters
        for i = 1:m
            for j = 1:n
                % Identify the pooling region
                row_start = (i-1) * pool_size + 1;
                row_end = row_start + pool_size - 1;
                col_start = (j-1) * pool_size + 1;
                col_end = col_start + pool_size - 1;
                
                % Extract pool region from A_conv
                pool_region = A_conv(row_start:row_end, col_start:col_end, f);
                
                % Find max location
                [max_row, max_col] = find(pool_region == max(pool_region(:)), 1);
                
                % Assign the corresponding gradient
                delta_conv(row_start + max_row - 1, col_start + max_col - 1, f) = delta_pool(i, j, f);
            end
        end
    end
    
    % ---- STEP 5: Compute Gradients for Convolutional Weights ----
    % dW_conv : (fH, fW, C_in, num_filters)
    dW_conv = zeros(size(W_conv));
    
    % db_conv : (1, num_filters)
    db_conv = zeros(1, size(W_conv, 4));
    
    [fh, fw, num_channels, num_filters] = size(W_conv); % fH = fh, fW = fw, C_in = num_channels
    
    for f = 1:num_filters
        for c = 1:num_channels
            % Convolve input with delta_conv (full cross-correlation)
            dW_conv(:,:,c,f) = conv2(X(:,:,c), rot90(delta_conv(:,:,f), 2), 'valid');
        end
        % Bias gradient: sum over spatial locations
        db_conv(f) = sum(delta_conv(:,:,f), 'all');
    end
    
    % ---- STEP 6: Update Weights ----
    W_fc = W_fc - alpha * dW_fc;  % (num_classes, fc_dim)
    b_fc = b_fc - alpha * db_fc;  % (num_classes, 1)
    W_conv = W_conv - alpha * dW_conv; % (fH, fW, C_in, num_filters)
    b_conv = b_conv - alpha * db_conv; % (1, num_filters)
end
