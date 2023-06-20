% Cross Validation CPSVM  (linear kernel)

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

Cl=-8; %Cota inferior para C, Paper Shao et al usa -8
Ch=8;  % Cota superior para C, Paper Shao et al usa 8

Ceps_l=-8; %Cota inferior para eps, Paper Shao et al usa -8
Ceps_h=0; %Cota superior para eps, Paper Shao et al usa 0
AUCMATRIX = zeros(Ch-Cl+1, Ceps_h-Ceps_l+1);
ACCUMATRIX = zeros(Ch-Cl+1, Ceps_h-Ceps_l+1);

tic;  % Inicio del contador de tiempo
for i=Cl:Ch
    C1=2^i;
%    for j=Cl:Ch
        C2=C1;
        for z=Ceps_l: Ceps_h
            epsilon = 2^z;
            % Inicializar variables para almacenar el máximo AUC y Accuracy
            maxAUC = -inf;
            maxAccuracy = -inf;
            for k=1:CV
                tst=k:10:m;
                trn=setdiff(1:m,tst);
                Ytr=Y(trn,:);    % definimos las etiquetas de entrenamiento
                Xtr=X(trn,fInd); % definimos el conjunto de entrenamiento 
                Yt=Y(tst',:);    % definimos las etiquetas de test
                Xt=X(tst',fInd); % definimos el conjunto test
                [w,b] = cpsvm_prim_sep(Xtr, Ytr, C1, C2, epsilon);
                prediction = sign(Xt*w + b-0.5);
                [AUC(k),Accu(k)]=medi_auc_accu(prediction,Yt);
            end
            AUC_media=mean(AUC);
            Accu_media=mean(Accu);
%            if AUC_media > maxAUC
%                maxAUC=AUC_media;
%            end
%            if Accu_media > maxAccuracy
%                maxAccuracy=Accu_media;
%            end
%        end
        AUCMATRIX(i-Cl+1,z-Ceps_l+1)=AUC_media;
        ACCUMATRIX(i-Cl+1,z-Ceps_l+1)=Accu_media;
    end
end

% Tiempo utilizado por el clasificador
tiempoTranscurrido = toc;  % Tiempo transcurrido desde tic hasta toc
disp(['El tiempo utilizado por el clasificador es: ' num2str(tiempoTranscurrido) ' segundos']);

% Especifica el nombre del archivo de Excel
filename = 'resultadoscpsvm.xlsx';

% Escribe la matriz AUCMATRIX en la hoja de cálculo 'AUC'
xlswrite(filename, AUCMATRIX', 'AUC');

% Escribe la matriz ACCUMATRIX en la hoja de cálculo 'Accuracy'
xlswrite(filename, ACCUMATRIX', 'Accuracy');