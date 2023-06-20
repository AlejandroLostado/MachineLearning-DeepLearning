clc
clear all
addpath(genpath('Funciones utilizadas'))
addpath(genpath('dataset_bin'))
load('bmpm.mat')

% Obtener índices de puntos con etiqueta 1 y -1
ind1 = find(data(:, 3) == 1);
indm1 = find(data(:, 3) == -1);

% Trazar los puntos con etiqueta 1 como triángulos rojos
scatter(data(ind1, 1), data(ind1, 2), 50, 'r', 'filled', 'Marker', 'v');
hold on;

% Trazar los puntos con etiqueta -1 como círculos azules
scatter(data(indm1, 1), data(indm1, 2), 50, 'b', 'filled', 'Marker', 'o');

% Obtener los parámetros del hiperplano separador
[w, b, alpha] = svm_dual_nonsep_gaussian_kernel(data(:,1:2), data(:,3), 10000, 1);

% Definir los límites del gráfico
xlim([min(data(:,1))-1, max(data(:,1))+1]);
ylim([min(data(:,2))-1, max(data(:,2))+1]);

% Generar una malla de puntos para cubrir todo el área del gráfico
xrange = linspace(min(data(:,1))-1, max(data(:,1))+1, 200);
yrange = linspace(min(data(:,2))-1, max(data(:,2))+1, 200);
[X, Y] = meshgrid(xrange, yrange);
xy = [X(:) Y(:)];

% Evaluar la función de decisión en los puntos de la malla
f = (alpha .* data(:,3))' * gaussian_kernel(data(:,1:2), xy, 1) + b;

% Graficar la línea de decisión
contour(X, Y, reshape(f, size(X)), [0 0], 'k', 'LineWidth', 2);

% Agregar título y leyenda
title('Gráfico de separación de clases con kernel gaussiano y SVM dual');
legend('Clase 1', 'Clase -1', 'Hiperplano decisión');

% Predicción de las etiquetas en el conjunto de datos de entrenamiento
y_pred = sign((alpha .* data(:,3))' * gaussian_kernel(data(:,1:2), data(:,1:2), 1) + b);

% Convertir y_pred a un vector columna
y_pred = reshape(y_pred, [], 1);

% Cálculo de las métricas de evaluación
tp = sum(data(:,3) == 1 & y_pred == 1); % Verdaderos positivos
tn = sum(data(:,3) == -1 & y_pred == -1); % Verdaderos negativos
fp = sum(data(:,3) == -1 & y_pred == 1); % Falsos positivos
fn = sum(data(:,3) == 1 & y_pred == -1); % Falsos negativos


accuracy = (tp+tn)/(tp+tn+fp+fn);
precision = tp/(tp+fp);
recall = tp/(tp+fn);
f1_score = 2*precision*recall/(precision+recall);
sensitivity = tp/(tp+fn);
specificity = tn/ (tn + fp);

% Imprimir resultados por pantalla
fprintf('Accuracy del SVM en forma dual con kernel gaussiano: %.2f%%\n', mean(accuracy)*100);
fprintf('Sensitivity del SVM en forma dual con kernel gaussiano: %.2f%%\n', mean(sensitivity)*100);
fprintf('Specificity del SVM en forma dual con kernel gaussiano: %.2f%%\n', mean(specificity)*100);
fprintf('Precision del SVM en forma dual con kernel gaussiano: %.2f%%\n', mean(precision)*100);
fprintf('Recall del SVM en forma dual con kernel gaussiano: %.2f%%\n', mean(recall)*100);
fprintf('F1 score del SVM en forma dual con kernel gaussiano: %.2f%%\n', mean(f1_score)*100);