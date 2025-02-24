function output = sigmoid (input);
    % Dimension of input and output will be same and element wise operation
    % is needed to be performed.
    output = 1 ./ (1 + exp(-input));
end