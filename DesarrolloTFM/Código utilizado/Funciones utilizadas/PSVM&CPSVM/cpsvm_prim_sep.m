function [w,b] = cpsvm_prim_sep(data, labels, C1, C2, epsilon)
    % INPUT
    % data: num-by-dim matrix. num is the number of data points,
    % dim is the dimension of a point
    % labels: num-by-1 vector, specifying the class that each point
    % belongs to.
    % either be +1 or be -1
    % C: the tuning parameter
    % epsilon: epsilon parameter
    % OUTPUT
    % w: num-by-1 vector, optimal weights
    % b: a scalar, the bias
    %Paper :Twin SVM for conditional probability estimation in binary and multiclass classification
    % Shao et al. 2023
    % model PSVM (3)
    
    [num, dim] = size(data);
%    t0=cputime;
    cvx_begin
        variable w(dim);
        variable b;
        variable xi(num);
        minimize(sum(w.^2) / 2  + C1 * sum(xi) / epsilon- C2 * sum(labels .* (data * w + b)));
        subject to
            labels .* (data * w + b - 0.5) >= 0.5 * epsilon - xi;
            data * w + b >= 0;
            data * w + b <= 1;
            xi >= 0;
    cvx_end
%     tf=cputime-t0;
%     Predict = sign(data*w + b-0.5);
%     % Calcular AUC y matriz de confusión
%     auc = AUCcalc(Predict, labels);
%     [~, Accu, Sens, Spec, cm] = medi_auc_accu(Predict, labels);
end