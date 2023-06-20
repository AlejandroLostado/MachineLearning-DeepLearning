% Cross Validation SVM soft margin (nonlinear kernel)

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
[m n]=size(X);
fInd=[1:n];%<--- return all features   
CV=10;
Cl=-7;
Ch=7;
AUCMATRIX=zeros(Ch-Cl+1,Ch-Cl+1);

for i=Cl:Ch
    C=2^i;
    for j = Cl:Ch
        sigma=2^j;
        gamma=1/(2*sigma*sigma);
        strlibsvm=strcat({' -c '}, {num2str(C)}, {' -g '}, {num2str(gamma)});
        for k=1:CV
            k
            tst=perm(k:10:m);
            trn=setdiff(1:m,tst);
            Ytr=Y(trn,:);
            Xtr=X(trn,fInd);
            Yt=Y(tst',:);
            Xt=X(tst',fInd);
            model = svmtrain(Ytr,Xtr,strlibsvm{1});
            prediction=svmpredict(Yt,Xt,model);
            AUC(k)=AUCcalc(prediction,Yt);
        end
        AUCMATRIX(i-Cl+1,j-Cl+1)=mean(AUC);
    end
end
    
