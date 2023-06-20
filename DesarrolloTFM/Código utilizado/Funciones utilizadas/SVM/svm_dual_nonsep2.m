function [w, b, alpha] = svm_dual_nonsep2(data, labels, C)
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
% alpha: num-by-1 vector, dual variables
        [num, ~] = size(data);
        H = (data * data') .* (labels * labels');
        cvx_begin
            variable alpha(num);
            maximize(sum(alpha) - alpha' * H * alpha/ 2 - sum(alpha.^2) / (4 * C));
            subject to
                alpha >= 0;
                labels' * alpha == 0
        cvx_end
        sv_ind = alpha > 1e-4;
        w = data' * (alpha .* labels);
        xi = alpha / (2 * C);
        b = mean(labels(sv_ind) .* (1 - xi(sv_ind)) - data(sv_ind, :) * w);
end