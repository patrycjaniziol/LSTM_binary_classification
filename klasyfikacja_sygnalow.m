
%% za³adowanie danych treningowych
load('TrainData');
load('TrainDataLabels');
XXTrain = num2cell(TrainData,2);
YYTrain = categorical(TrainDataLabels);
%%
load('TestData');
load('TestDataLabels');
XXTest = num2cell(TestData,2);
YYTest = categorical(TestDataLabels);

%% Wizualizacja danych treningowych oraz testowych na wykresie
[x,y] = size(XXTrain);
figure

for k=1:x
    plot(XXTrain{k},'g')
hold on;
end

[x,y] = size(XXTest);
hold on
for k=1:x
    plot(XXTrain{k},'r')
hold on;
end
hold on
title('Dane treningowe - kolor zielony, dane testowe - kolor czerwony      ')
clear x; clear y; clear k;
%% okreœlenie wartoœci parametrów
inputSize = 1;
numHiddenUnits =300;
numClasses = 2;
%% zaimplementowanie warstw sieci neuronowej
layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
    ]
%% okreœlenie wartoœci parametrów
maxEpochs = 60;
miniBatchSize = 126;
InitialLearnRate = 0.001;
SequenceLength = 256;
GradientThreshold = Inf;
%% ustawienie opcji treningu
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'cpu', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', InitialLearnRate, ...
    'SequenceLength', SequenceLength, ...
    'GradientThreshold', GradientThreshold, ...
    'Plots','training-progress', ...
    'Verbose',false);
%% trening sieci neuronowej
net = trainNetwork(XXTrain,YYTrain,layers,options);

%% klasyfikacja danych treningowych
[trainPred, trainScores] = classify(net,XXTrain, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
%% stworzenie macierzy pomy³ek dla danych treningowych
figure
plotconfusion(YYTrain,trainPred,'[training]    ')
acc_train = sum(trainPred == YYTrain)./numel(YYTrain)
pause(4)
%% klasyfikacja danych testowych
YYPred = classify(net,XXTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
%% stworzenie macierzy pomy³ek dla danych testowych
figure
plotconfusion(YYTest,YYPred,'[testing]  ')
acc_test = sum(YYPred == YYTest)./numel(YYTest)
pause(4)
%%