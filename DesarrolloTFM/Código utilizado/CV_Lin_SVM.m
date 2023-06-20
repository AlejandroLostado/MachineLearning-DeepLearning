% Cross Validation SVM soft margin (linear kernel)

clc
clear all
addpath(genpath('Funciones utilizadas'))
addpath(genpath('dataset_bin'))

% Conjuntos de datos utilizados (descomentar el que queramos):

% load('iris_binario.mat')
% load('exa_bmpm.mat')
% load('bupa_liverN.mat')
% load('heart_statlogN.mat')
% load('titanic.mat')
% load('ionosphereN.mat')
load('sonar.mat')

%X=normaliza(X);

[m,n]=size(X);
folds=10;
Cl=-7;
Ch=7;
FunPara.kerfPara.type = 'lin';

AUCMATRIX=zeros(1,Ch-Cl+1);
ACCUMATRIX=zeros(1,Ch-Cl+1);

for i=Cl:Ch
    i
    FunPara.c=2^i;
    for k=1:folds
        tst=perm(k:folds:m); % test
        trn=setdiff(1:m,tst); % training
        % Training data
        Ytr=Y(trn,:);
        Xtr=X(trn,:);
        % Test data
        Yt=Y(tst',:);
        Xt=X(tst',:);
%       prediction = SVM_soft_quadsolve(Xtr,Ytr,Xt,FunPara); % via Quadolve
        prediction = SVM_softcvx(Xtr,Ytr,Xt,FunPara);   % via cvx
        [AUC(k),Accu(k)]=medi_auc_accu(prediction,Yt);
    end
    AUCMATRIX(i-Cl+1)=mean(AUC);
    ACCUMATRIX(i-Cl+1)=mean(Accu);
end

