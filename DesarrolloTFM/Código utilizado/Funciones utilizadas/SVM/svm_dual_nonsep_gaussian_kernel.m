function [w, b, alpha] = svm_dual_nonsep_gaussian_kernel(data,labels, C, sigma)
% INPUT
% data: num-by-dim matrix. num is the number of data points,
% dim is the dimension of a point
% labels: num-by-1 vector, specifying the class that each point
% belongs to.
% either be +1 or be -1
% C: the tuning parameter
% sigma: the parameter of gaussian kernel
% OUTPUT
% b: a scalar, the bias
% alpha: num-by-1 vector, dual variables
    [num, ~] = size(data);
    K = zeros(num);
    kernel = @(x, y) exp(-norm(x - y)^2 / 2 / sigma^2) / sqrt(2 * pi)/sigma;
    for i = 1:num
        for j = i:num
            K(i, j) = kernel(data(i, :), data(j, :));
            K(j, i) = K(i, j);
        end
    end
    H = (labels * labels') .* K;

    cvx_begin
        variable alpha(num);
        maximize(sum(alpha) - alpha' * H * alpha / 2);
        subject to
            alpha >= 0;
            alpha <= C;
            labels' * alpha == 0;
    cvx_end
    w = data' * (alpha .* labels);
    ind = alpha > 1e-4 & alpha < C - 1e-4;
    b = mean(labels(ind) - K(ind, :) * (alpha .* labels));
end