function a = softmax(z)
    exp_z = exp(z - max(z, [], 1)); % Numerical stability
    a = exp_z ./ sum(exp_z, 1);
end