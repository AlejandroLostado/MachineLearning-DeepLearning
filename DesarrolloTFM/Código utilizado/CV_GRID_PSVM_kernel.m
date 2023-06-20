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

[m n]=size(X);
fInd=[1:n];%<--- return all features   
CV=10; % validacion cruzada cantidad de folds
kerfPara.type='rbf';

Cl=-8; %Cota inferior para C, Paper Shao et al usa -8
Ch=8;  % Cota superior para C, Paper Shao et al usa 8

Ceps_l=-8; %Cota inferior para eps, Paper Shao et al usa -8
Ceps_h=0; %Cota superior para eps, Paper Shao et al usa 0

pars_l=-6; %Cota inferior para eps, Paper Shao et al usa -6
pars_h=6; %Cota superior para eps, Paper Shao et al usa 6

AUCMATRIX=zeros(Ch-Cl+1, Ceps_h-Ceps_l+1);
ACCUMATRIX=zeros(Ch-Cl+1, Ceps_h-Ceps_l+1);

tic;  % Inicio del contador de tiempo
for i=Cl:Ch
    C=2^i;
    for j=Ceps_l: Ceps_h
        epsilon = 2^j; %cambiamos el nombre a epsilon pues 
        % la palabra eps está reservada en matlab par 2^-16
        for t=pars_l: pars_h
            kerfPara.pars=2^t;
            for k=1:CV
            tst=k:10:m;
            trn=setdiff(1:m,tst);
            Ytr=Y(trn,:);    % definimos las etiquetas de entrenamiento
            Xtr=X(trn,fInd); % definimos el conjunto de entrenamiento 
            Yt=Y(tst',:);    % definimos las etiquetas de test
            Xt=X(tst',fInd); % definimos el conjunto test
            [alpha_val, gamma_val, beta_val, b] = psvm_dual_nonsep(Xtr, Ytr, C, epsilon, kerfPara);
            Kt=kernelfun(Xtr,kerfPara,Xt);
            prediction = sign(Kt' * (alpha_val .* Ytr + beta_val - gamma_val) + b);
            [AUC(k),Accu(k)]=medi_auc_accu(prediction,Yt);
            end
        end
        AUCMATRIX(i-Cl+1,j-Ceps_l+1)=mean(AUC);
        ACCUMATRIX(i-Cl+1,j-Ceps_l+1)=mean(Accu);
    end
end

% Tiempo utilizado por el clasificador
tiempoTranscurrido = toc;  % Tiempo transcurrido desde tic hasta toc
disp(['El tiempo utilizado por el clasificador es: ' num2str(tiempoTranscurrido) ' segundos']);


% Especifica el nombre del archivo de Excel
filename = 'prueba.xlsx';

% Escribe la matriz AUCMATRIX en la hoja de cálculo 'AUC'
xlswrite(filename, AUCMATRIX', 'AUC');

% Escribe la matriz ACCUMATRIX en la hoja de cálculo 'Accuracy'
xlswrite(filename, ACCUMATRIX', 'Accuracy');