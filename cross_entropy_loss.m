function loss = cross_entropy_loss(y_true, y_pred)
    % Ensure numerical stability by adding a small value (epsilon)
    epsilon = 1e-9;
    y_pred = max(y_pred, epsilon);  % Prevent log(0) issues
    
    % Compute categorical cross-entropy loss
    loss = -sum(y_true .* log(y_pred));
end
