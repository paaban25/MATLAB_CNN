% Set parameters
input_size = [8, 8, 3];  % Image size (h, w, num_channels)
filter_size = [3, 3, 3, 2]; % (fh, fw, num_channels, num_filters)
pool_size = 2;
num_classes = 5;
learning_rate = 0.01;

% Initialize random inputs
X = rand(input_size);  % Random input image
Y_true = zeros(num_classes, 1);
Y_true(randi(num_classes)) = 1; % Random one-hot encoded label

% Initialize random weights & biases
W_conv = rand(filter_size);  
b_conv = rand(1, 1, filter_size(4));

num_fc_inputs=18;

W_fc = randn(num_classes, num_fc_inputs);
b_fc = rand(num_classes, 1);

% Forward pass
A_conv = convolutional_layer(X, W_conv,b_conv, 1); % Convolution layer output
A_pool = max_pooling(A_conv, 2, 2, 2); % Using 2x2 pooling with stride 2

A_fc = reshape(A_pool, [], 1);             % Flattened input to FC layer

% Run backpropagation
[W_fc_new, b_fc_new, W_conv_new, b_conv_new] = backpropagation(X, Y_true, W_fc, b_fc, W_conv, b_conv, A_conv, A_pool, A_fc, learning_rate);

% Check if weights are updated
disp("FC Weights Updated: " + any(W_fc_new(:) ~= W_fc(:)));
disp("FC Bias Updated: " + any(b_fc_new(:) ~= b_fc(:)));
disp("Conv Weights Updated: " + any(W_conv_new(:) ~= W_conv(:)));
disp("Conv Bias Updated: " + any(b_conv_new(:) ~= b_conv(:)));
