function output = max_pooling(input, ph, pw, stride)

    % Number of channels remains the same
    [h, w, num_filters] = size(input);

    % Compute output dimensions
    ho = floor((h - ph) / stride) + 1;
    wo = floor((w - pw) / stride) + 1;

    % Initialize output
    output = zeros(ho, wo, num_filters);

    for f = 1:num_filters  % Loop over each channel
        for i = 1:stride:h-ph+1  % Row index
            for j = 1:stride:w-pw+1  % Column index
                
                % Extract the pooling window
                region = input(i:i+ph-1, j:j+pw-1, f);

                % Apply max pooling
                output(ceil(i/stride), ceil(j/stride), f) = max(region(:));

            end
        end
    end
end
