function image_out = convolutional_layer(image, filter, b_conv, stride)
    % image: (h x w x num_channels)
    % filter: (fh x fw x num_channels x num_filters)
    % b_conv: (1 x num_filters) Bias term for each filter
    % stride: Step size for convolution
    % image_out: (ho x wo x num_filters)

    [h, w, num_channels_img] = size(image);
    [fh, fw, num_channels_filter, num_filters] = size(filter);
    
    % Validate that the stride is correct
    if mod((h - fh), stride) ~= 0 || mod((w - fw), stride) ~= 0
        error("Invalid stride value. Ensure (h - fh) and (w - fw) are divisible by stride.");
    end

    % Compute output dimensions
    ho = floor((h - fh) / stride) + 1;
    wo = floor((w - fw) / stride) + 1;
    
    % Validate number of channels
    if num_channels_img ~= num_channels_filter
        error("Number of input channels must match filter channels.");
    end

    % Initialize output feature map
    image_out = zeros(ho, wo, num_filters);

    % Convolution operation
    for f = 1:num_filters
        for i = 1:ho
            for j = 1:wo
                row_start = (i - 1) * stride + 1;
                row_end = row_start + fh - 1;
                col_start = (j - 1) * stride + 1;
                col_end = col_start + fw - 1;

                % Extract input patch
                region = image(row_start:row_end, col_start:col_end, :);
                
                % Perform convolution for this filter
                sum_value = 0;
                for c = 1:num_channels_img
                    sum_value = sum_value + element_multiplication(region(:,:,c), filter(:,:,c,f));
                end
                
                % Add bias term
                image_out(i, j, f) = sum_value + b_conv(f);
            end
        end
    end
end
