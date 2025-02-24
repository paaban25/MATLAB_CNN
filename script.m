test_image = rand(28, 28, 1);

% Convolution filters: (filter height x filter width x num_channels x num_filters)
filters = rand(3, 3, 1, 8); % Example: 3x3 filter, 1 channel, 8 filters

% Fully connected layer: (M x N)
fc_weights = rand(10, 1352); % Example: 10 neurons, input size 100

% Bias vector: (M x 1)
fc_bias = rand(10, 1);

stride_conv = 1; % Convolution stride
pool_size = 2; % Pooling window size
stride_pool = 2; % Pooling stride

prediction = forward_pass(test_image, filters, stride_conv, pool_size, stride_pool, fc_weights, fc_bias);

disp("Prediction:");
disp(prediction);

% Check if values are between 0 and 1 (sigmoid output)
if all(prediction >= 0 & prediction <= 1)
    disp("✅ Prediction values are within valid range (0 to 1).");
else
    disp("❌ Error: Values out of range.");
end

conv_out = convolutional_layer(test_image,filters,stride_conv);
relu_out = relu_activation (conv_out);
pooled_out = max_pooling (relu_out,pool_size,pool_size,stride_pool);
flattened_out = pooled_out(:);
fc_out = fully_connected_layers(flattened_out,fc_weights,fc_bias);
final_output = output_layer(fc_out); % Apply sigmoid activation

disp (final_output);


% disp("Size of conv_out: "), disp(size(conv_out));
% disp("Size of relu_out: "), disp(size(relu_out));
% disp("Size of pooled_out: "), disp(size(pooled_out));
% disp("Size of flattened_out: "), disp(size(flattened_out));
% disp("Size of fc_out: "), disp(size(fc_out));


