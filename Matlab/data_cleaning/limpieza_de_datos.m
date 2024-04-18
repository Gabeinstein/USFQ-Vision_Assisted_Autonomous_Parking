%%% Proyecto Sistemas de Comunicaciones - Primavera 2024
%%% Gabriel Oña, Jose Montahuano y Emilia Casares

%%% Código para unir los datasets generados desde rosbag(.bag) a un solo
%%% Dataset (.csv) junto con el valor esperado y la correccion de (inf -> 0)

%%% ------- En la Red Neuronal --------
%%% Retro -> 0
%%% Paralelo -> 1
%%% Diagonal -> 2
%%% Frente -> 3

Retro = readmatrix("datasets\diagonal\dataset_diagonal.csv");
Paralelo = readmatrix("datasets\paralelo\dataset_paralelo.csv");
Diagonal = readmatrix("datasets\diagonal\dataset_diagonal.csv");
Frente = readmatrix("datasets\frente\dataset_frente.csv");

Retro(:,end + 1) = 0;
Paralelo(:, end + 1) = 1;
Diagonal(:, end + 1) = 2;
Frente(:, end + 1) = 3;

Retro(Retro == Inf) = 0;
Paralelo(Paralelo == Inf) = 0;
Diagonal(Diagonal == Inf) = 0;
Frente(Frente == Inf) = 0;

dataset_completo = cat(1,Retro,Paralelo,Diagonal,Frente);

writematrix(dataset_completo, 'datasets\dataset_completo.csv');

