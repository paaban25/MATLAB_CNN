function out = matrix_multiplication(a, b)
    [rowA, colA] = size(a);
    [rowB, colB] = size(b);
    
    if colA ~= rowB
        disp("Matrix Multiplication Not Possible");
        out = []; 
        return;
    end
    
    out = zeros(rowA, colB); 
    
    for i = 1:rowA
        for j = 1:colB
            accumulate = 0; 
            for k = 1:colA
                accumulate = accumulate + approximate_multiplier(a(i, k), b(k, j));
            end
            out(i, j) = accumulate; 
        end
    end
end
