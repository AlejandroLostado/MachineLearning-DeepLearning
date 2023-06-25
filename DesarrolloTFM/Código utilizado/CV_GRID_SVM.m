% Cross Validation SVM soft margin (linear kernel)

clc
clear all
addpath(genpath('dataset_bin'))
addpath(genpath('dataset_Imb'))
addpath(genpath('SVM'))

%load 'sonar.mat'
load('heart_statlogN.mat');
%load('bupa_liverN.mat');
%load('ionosphereN.mat');
%load('breastcancer')
%load('australian.mat');
%load('diabetes.mat');
%load('german_credit.mat');
%load('splice.mat');
%load x18data.mat; % Flare-M
%load x23data.mat; % Yeast4
%load yeast3.mat
%load('titanic.mat');
%load segment0_n.mat
%load('image_n.mat');
%load('waveformBin.mat');
%load 'phoneme.mat'
%load('ring_n.mat')

%X=normaliza(X);
[m n]=size(X);
fInd=[1:n];%<--- return all features   
CV=10;
Cl=-7;
Ch=7;
AUCMATRIX=zeros(1,Ch-Cl+1);

for i=Cl:Ch
    C=2^i;
    strlibsvm=strcat({' -c '}, {num2str(C)});
    for k=1:CV
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
    AUCMATRIX(i-Cl+1)=mean(AUC);
end

