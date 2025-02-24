function prediction = forward_pass(image, filters, stride_conv, pool_size, stride_pool, fc_weights, fc_bias)
    % Step 1: Convolution Layer
    conv_out = convolutional_layer(image, filters, stride_conv);
    
    % Step 2: ReLU Activation
    relu_out = relu_activation(conv_out);
    
    % Step 3: Max Pooling Layer
    pooled_out = max_pooling(relu_out, pool_size, pool_size, stride_pool);
    
    % Step 4: Flatten Layer
    flattened_out = pooled_out(:); % Convert to 1D
    
    % Step 5: Fully Connected Layer
    fc_out = fully_connected_layers(flattened_out, fc_weights, fc_bias);
    
    % Step 6: Output Layer (Sigmoid Activation)
    prediction = output_layer(fc_out);
end
