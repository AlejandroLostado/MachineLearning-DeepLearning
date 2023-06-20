clc
clear all
addpath(genpath('Funciones utilizadas'))
addpath(genpath('dataset_bin'))
load('iris_binario.mat')

K = 5; % número de pliegues de validación cruzada
indices = crossvalind('Kfold', size(X, 1), K); % dividir los datos en K subconjuntos


VP = zeros(K, 1);
FP = zeros(K, 1);
VN = zeros(K, 1);
FN = zeros(K, 1);

accuracy = zeros(K, 1);
sensitivity = zeros(K, 1);
specificity = zeros(K, 1);
precision = zeros(K, 1);
recall = zeros(K, 1);
F1_score = zeros(K, 1);

for i = 1:K
    test_idx = (indices == i); % índices de datos de prueba
    train_idx = ~test_idx; % índices de datos de entrenamiento
    
    X_train = X(train_idx, :); % datos de entrenamiento
    Y_train = Y(train_idx);
    
    X_test = X(test_idx, :); % datos de prueba
    Y_test = Y(test_idx);
    
    [w,b]=svm_prim_sep(X_train,Y_train); % entrenar el modelo con los datos de entrenamiento
    
    % Predecir etiquetas de datos de prueba
    predicciones = sign(X_test * w + b);
    
    % Calcular métricas de evaluación
    VP(i) = sum(predicciones == 1 & Y_test == 1);
    FP(i) = sum(predicciones == 1 & Y_test == -1);
    VN(i) = sum(predicciones == -1 & Y_test == -1);
    FN(i) = sum(predicciones == -1 & Y_test == 1);
    
    accuracy(i) = (VP(i)+VN(i))/(VP(i)+FP(i)+FN(i)+VN(i));
    sensitivity(i) = VP(i) / (VP(i) + FN(i));
    specificity(i) = VN(i)/ (VN(i) + FP(i));
    precision(i) = VP(i) / (VP(i) + FP(i));
    recall(i) = VP(i) / (VP(i) + FN(i));
    F1_score(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i));
end

% Imprimir resultados por pantalla
fprintf('Accuracy del SVM en forma primal tras la validación cruzada: %.2f%%\n', mean(accuracy)*100);
fprintf('Sensitivity del SVM en forma prima tras la validación cruzada: %.2f%%\n', mean(sensitivity)*100);
fprintf('Specificity del SVM en forma primal tras la validación cruzada: %.2f%%\n', mean(specificity)*100);
fprintf('Precision del SVM en forma primal tras la validación cruzada: %.2f%%\n', mean(precision)*100);
fprintf('Recall del SVM en forma primal tras la validación cruzada: %.2f%%\n', mean(recall)*100);
fprintf('F1 score del SVM en forma primal tras la validación cruzada: %.2f%%\n', mean(F1_score)*100);
