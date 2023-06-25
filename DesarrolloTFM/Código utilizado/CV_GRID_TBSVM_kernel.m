% Cross Validation TBSVM (linear kernel)

clc
clear all
addpath(genpath('dataset_bin'))
addpath(genpath('dataset_Imb'))
addpath(genpath('TBSVM'))

load 'exa_bmpm.mat'
%load 'sonar.mat'
%load('heart_statlogN.mat');
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
ACCUMATRIX=zeros(Ch-Cl+1,Ch-Cl+1);
AUCMATRIX=zeros(Ch-Cl+1,Ch-Cl+1);
FunPara.kerfPara.type = 'rbf';

for i=Cl:Ch
    i
    FunPara.c1=2^i;
    FunPara.c2=2^i;
    FunPara.c3=2^i;
    FunPara.c4=2^i;
    for j = Cl:Ch
        FunPara.kerfPara.pars = 2^j;
        for k=1:CV
            tst=perm(k:10:m);
            trn=setdiff(1:m,tst);
            Ya=Y(trn,:);
            Xa=X(trn,fInd);
            fin1=find(Ya==1);
            fin2=find(Ya==-1);
            DataTrain.A=Xa(fin1,:);
            DataTrain.B=Xa(fin2,:);
            Yt=Y(tst',:);
            Xt=X(tst',fInd);
            prediction=TWSVM(Xt,DataTrain,FunPara);
            [AUC(k),Accu(k)]=medi_auc_accu(prediction,Yt);
        end
        ACCUMATRIX(i-Cl+1,j-Cl+1)=mean(Accu);
        AUCMATRIX(i-Cl+1,j-Cl+1)=mean(AUC);
    end
end

