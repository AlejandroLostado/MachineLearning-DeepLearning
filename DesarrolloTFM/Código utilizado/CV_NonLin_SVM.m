% Cross Validation SVM soft margin (nonlinear kernel), by using: quadsolve

clc
clear all
addpath(genpath('dataset_bin'))
addpath(genpath('dataset_Imb'))
addpath(genpath('SVM'))

load 'sonar.mat'
%load('heart_statlogN.mat');
%load('bupa_liverN.mat');
%load('ionosphereN.mat');
%load('breastcancer')
%load('australian.mat');
%load('diabetes.mat');
%load('german_credit.mat');
%load('splice.mat');
%load x18data.mat; % Solar
%load x23data.mat; % Yeast4
%load yeast3.mat
%load('titanic.mat');
%load segment0_n.mat
%load('image_n.mat');
%load('waveformBin.mat');
%load 'phoneme.mat'
%load('ring_n.mat')

%X=normaliza(X);

[m,n]=size(X);
folds=10;
Cl=-7;
Ch=7;
FunPara.kerfPara.type = 'rbf';

AUCMATRIX=zeros(1,Ch-Cl+1);
ACCUMATRIX=zeros(1,Ch-Cl+1);

for i=Cl:Ch
    i
    FunPara.c=2^i;
    for j=Cl:Ch
        FunPara.kerfPara.pars = 2^j;
        for k=1:folds
            tst=perm(k:folds:m); % test
            trn=setdiff(1:m,tst); % training
            % Training data
            Ytr=Y(trn,:);
            Xtr=X(trn,:);
            % Test data
            Yt=Y(tst',:);
            Xt=X(tst',:);
            prediction = SVM_soft_quadsolve(Xtr,Ytr,Xt,FunPara); % via Quadolve
            [AUC(k),Accu(k)]=medi_auc_accu(prediction,Yt);
        end
        AUCMATRIX(i-Cl+1,j-Cl+1)=mean(AUC);
        ACCUMATRIX(i-Cl+1,j-Cl+1)=mean(Accu);
    end
end

