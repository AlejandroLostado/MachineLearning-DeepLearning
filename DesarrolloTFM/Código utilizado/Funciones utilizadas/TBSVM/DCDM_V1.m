% Dual coordinate descent method (DCDM) for solving the following
% Quadratic programming

%  min 1/2*alpha'*H*alpha+f'*alpha
%  s.a. lb<=alpha<=ub


function alpha = DCDM_V1(H,f,lb,ub,iter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   alpha = DCDM_V1(H,f,lb,ub,iter)
%
%   Input:
%        H      - Quadratic matrix in R^{n x n}(Require positive definite).
%        f      - vector
%        ub     - Upper bound vector in R^n
%        lb     - Lower bound vector in R^n
%        iter   - Maximum number of iterations

rng(2);
n=size(H,1);
PG=zeros(n,1);
alpha = lb+(ub-lb).*rand(n,1); 

k = 0;
Error=100;
eps=1e-5; % small enough value

%tic;
while Error>1.e-4 && k<iter 
    %rng(2);
    for i = randperm(n)
        G = H(i,:)*alpha+f(i);
        if abs(alpha(i)-lb(i)) < eps
            PG(i) = min(G,0);
        elseif abs(alpha(i)-ub(i)) < eps
            PG(i) = max(G,0);
        elseif lb(i)<alpha(i) && alpha(i)<ub(i)
            PG(i) = G;
        end
        if abs(PG(i)-0) > eps
            alpha(i) = min(max(alpha(i)-G/H(i,i),lb(i)),ub(i));
        end    
    end
    maxPG=max(PG);
    minPG=min(PG);
    Error=maxPG-minPG;
    k = k+1;
end
%t = toc/k;

