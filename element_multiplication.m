function out = element_multiplication (a,b)

    [rowA, colA, depthA] = size(a);
    [rowB, colB, depthB] = size(b);

    if size(a) ~= size(b)
        disp("Matrix Multiplication Not Possible");
        out = []; 
        return;
    end

    out =0;

    for i=1:rowA
        for j=1:colB
            for k=1:depthA
                out = out + approximate_multiplier(a(i,j,k),b(i,j,k));
            end
        end
    end

end