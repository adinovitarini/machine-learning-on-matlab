clear all;clc
[XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;
[XTest,YTest,anglesTest] = digitTest4DArrayData;
dataFolder = fullfile(toolboxdir('nnet'),'nndemos','nndatasets','DigitDataset');
imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
figure(1);
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
%%
labelCount = countEachLabel(imds)
%%
labelCount = countEachLabel(imds)
%%
img = readimage(imds,1);
size(img)
%%
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
%%
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%%
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
%% Training 
net = trainNetwork(imdsTrain,layers,options);
%%
[YPred,YPredScores] = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

%%
figure(2);
title('Confusion Matrix on Test Dataset')
confusionchart(imdsValidation.Labels,YPred,'Normalization',"row-normalized");

%%
chosenClass = "6";
classIdx = find(net.Layers(end).Classes == chosenClass);

numImgsToShow = 9;

[sortedScores,imgIdx] = findMaxActivatingImages(imdsValidation,chosenClass,YPredScores,numImgsToShow);

figure
plotImages(imdsValidation,imgIdx,sortedScores,YPred,numImgsToShow)



%% Function 
function downloadExampleFoodImagesData(url,dataDir)
% Download the Example Food Image data set, containing 978 images of
% different types of food split into 9 classes.

% Copyright 2019 The MathWorks, Inc.

fileName = "ExampleFoodImageDataset.zip";
fileFullPath = fullfile(dataDir,fileName);

% Download the .zip file into a temporary directory.
if ~exist(fileFullPath,"file")
    fprintf("Downloading MathWorks Example Food Image dataset...\n");
    fprintf("This can take several minutes to download...\n");
    websave(fileFullPath,url);
    fprintf("Download finished...\n");
else
    fprintf("Skipping download, file already exists...\n");
end
exampleFolderFullPath = fullfile(dataDir,"pizza");
if ~exist(exampleFolderFullPath,"dir")
    fprintf("Unzipping file...\n");
    unzip(fileFullPath,dataDir);
    fprintf("Unzipping finished...\n");
else
    fprintf("Skipping unzipping, file already unzipped...\n");
end
fprintf("Done.\n");

end

function [sortedScores,imgIdx] = findMaxActivatingImages(imds,className,predictedScores,numImgsToShow)
% Find the predicted scores of the chosen class on all the images of the chosen class
% (e.g. predicted scores for sushi on all the images of sushi)
[scoresForChosenClass,imgsOfClassIdxs] = findScoresForChosenClass(imds,className,predictedScores);

% Sort the scores in descending order
[sortedScores,idx] = sort(scoresForChosenClass,'descend');

% Return the indices of only the first few
imgIdx = imgsOfClassIdxs(idx(1:numImgsToShow));

end

function [sortedScores,imgIdx] = findMinActivatingImages(imds,className,predictedScores,numImgsToShow)
% Find the predicted scores of the chosen class on all the images of the chosen class
% (e.g. predicted scores for sushi on all the images of sushi)
[scoresForChosenClass,imgsOfClassIdxs] = findScoresForChosenClass(imds,className,predictedScores);

% Sort the scores in ascending order
[sortedScores,idx] = sort(scoresForChosenClass,'ascend');

% Return the indices of only the first few
imgIdx = imgsOfClassIdxs(idx(1:numImgsToShow));

end

function [scoresForChosenClass,imgsOfClassIdxs] = findScoresForChosenClass(imds,className,predictedScores)
uniqueClasses = unique(imds.Labels);
chosenClassIdx = string(uniqueClasses) == className;
imgsOfClassIdxs = find(imds.Labels == className);
scoresForChosenClass = predictedScores(imgsOfClassIdxs,chosenClassIdx);
end

function plotImages(imds,imgIdx,sortedScores,predictedClasses,numImgsToShow)

for i=1:numImgsToShow
    score = sortedScores(i);
    sortedImgIdx = imgIdx(i);
    predClass = predictedClasses(sortedImgIdx); 
    correctClass = imds.Labels(sortedImgIdx);
        
    imgPath = imds.Files{sortedImgIdx};
    
    if predClass == correctClass
        color = "\color{green}";
    else
        color = "\color{red}";
    end
    
    predClassTitle = strrep(string(predClass),'_',' ');
    correctClassTitle = strrep(string(correctClass),'_',' ');
    
    subplot(3,ceil(numImgsToShow./3),i)
    imshow(imread(imgPath));
    title("Predicted: " + color + predClassTitle + "\newline\color{black}Score: " + num2str(score) + "\newlineGround truth: " + correctClassTitle);
end

end

function plotGradCAM(img,gradcamMap,alpha)

subplot(1,2,1)
imshow(img);

h = subplot(1,2,2);
imshow(img)
hold on;
imagesc(gradcamMap,'AlphaData',alpha);

originalSize2 = get(h,'Position');

colormap jet
colorbar

set(h,'Position',originalSize2);
hold off;
end
