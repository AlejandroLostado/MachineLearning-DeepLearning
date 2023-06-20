function [w, b] = svm_prim_nonsep2(data, labels, C)
% INPUT
% data: num-by-dim matrix. num is the number of data points,
% dim is the dimension of a point
% labels: num-by-1 vector, specifying the class that each point
% belongs to.
% either be +1 or be -1
% C: the tuning parameter
% OUTPUT
% w: dim-by-1 vector, the normal direction of hyperplane
% b: a scalar, the bias
        [num, dim] = size(data);
        cvx_begin
            variables w(dim) b xi(num);
            minimize(sum(w.^2) / 2 + C * sum(xi.^2));
            subject to
                labels .* (data * w + b) >= 1 - xi;
                xi >= 0;
        cvx_end
end