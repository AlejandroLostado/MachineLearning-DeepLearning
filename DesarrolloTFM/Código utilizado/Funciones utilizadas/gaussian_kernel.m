function K = gaussian_kernel(X, Y, sigma)
    n = size(X, 1);
    m = size(Y, 1);
    K = zeros(n, m);
    for i = 1:n
        for j = 1:m
            K(i, j) = exp(-norm(X(i,:) - Y(j,:))^2 / 2 / sigma^2) / sqrt(2 * pi) / sigma;
        end
    end
end
