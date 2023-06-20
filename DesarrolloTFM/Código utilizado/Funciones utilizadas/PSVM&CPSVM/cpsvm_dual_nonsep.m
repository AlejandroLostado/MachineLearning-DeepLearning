function [alpha_val, gamma_val, beta_val, b]  = cpsvm_dual_nonsep(data, labels, C1,C2, eps, kerfPara)

[num, ~] = size(data);

K = kernelfun(data, kerfPara);
t0=cputime;
cvx_begin
    variable alpha_val(num);
    variable beta_val(num);
    variable gamma_val(num);

    maximize(sum(0.5 * alpha_val .* (labels + eps) - gamma_val) - 0.5 * sum(sum((labels .* alpha_val + beta_val - gamma_val)' * K * (labels .* alpha_val + beta_val - gamma_val))) - C2 * sum(labels' *K * sum((labels .* alpha_val) +  beta_val -  gamma_val)));
    subject to
        sum(labels .* alpha_val + beta_val - gamma_val) == -C2*sum(labels);
        0 <= alpha_val <= C1 / eps;
        beta_val >= 0;
        gamma_val >= 0;
cvx_end
tf=cputime-t0;
% Obtener los índices de los vectores de soporte
support_indices = (alpha_val > 0 & alpha_val < (C1 / eps)) | (beta_val > 0) | (gamma_val > 0);
b = mean(labels(support_indices) - K(support_indices, :) * (alpha_val .* labels + beta_val - gamma_val));
end