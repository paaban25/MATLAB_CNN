function output = output_layer (input)
    % input will be of size (M X 1) where M is the number of neurons in the
    % previous fully connecetd layer.
    output = softmax (input);
end