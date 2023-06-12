% This is an implementation of the flower image segmentation for the "daffodil" 
% flower from the oxford flowers dataset.

% This is my own implementation of the CNN without use of any other
% existing networks.

% The dataset used for this task is organized in two directories: one for the images ('.\daffodilSeg\ImagesRsz256')
% and one for their corresponding labels ('.\daffodilSeg\LabelsRsz256'). The labels are stored as grayscale images
% where pixel intensity corresponds to the class label of the corresponding pixel in the image.

% An imageDatastore and a pixelLabelDatastore are created for efficient loading and preprocessing of image and
% pixel label data, respectively. The class names and their corresponding pixel label IDs are defined before creating
% the datastores.

% Declare the directories where the images and their corresponding labels are stored
imageDir = '.\daffodilSeg\ImagesRsz256';
labelDir = '.\daffodilSeg\LabelsRsz256';

% Create an image datastore, which is a repository for the image data
imds = imageDatastore(imageDir);

% Define the class names and corresponding pixel label IDs
classNames = ["flower","background"];
labelID   = [1 3];  % Pixel label IDs: flower=1, background=3, ignoring other labels

% Create a Pixel Label Datastore, which is a repository for pixel label data
pxds = pixelLabelDatastore(labelDir, classNames, labelID);

% Split the data into training, validation, and test sets
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds);

% Combine the image and pixel label datastores for validation and training
dsVal = combine(imdsVal,pxdsVal);
dsTrain = combine(imdsTrain, pxdsTrain);

% Define the network parameters
numClasses = 2; % flower and background
numFilters = 64;
filterSize = 3;

layers = [
    % Downsampling layers
    imageInputLayer([256 256 3])
    convolution2dLayer(filterSize,32,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,64,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,128,'Padding',1)
    reluLayer()
    
   

    % Upsampling layers
    transposedConv2dLayer(4,64,'Stride',2,'Cropping',1)
    reluLayer()
    transposedConv2dLayer(4,64,'Stride',2,'Cropping',1)
    reluLayer()
    convolution2dLayer(1,numClasses)
    softmaxLayer()
    pixelClassificationLayer()  %unweighted
];

% Data augmentation is applied to the training set to improve the 
% network's ability to generalize and prevent overfitting.
% Define the range for random pixel translation during data augmentation
xTrans = [-10 10];
yTrans = [-10 10];
dsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data,xTrans,yTrans));

% Specify the training options
opts = trainingOptions('sgdm', ... % Use stochastic Gradient Descent with momentum
    'InitialLearnRate',1e-3, ...  % Initial learning rate
    'MaxEpochs',100, ...  % Maximum number of epochs to train
    'MiniBatchSize',4, ...  % Number of observations per mini-batch
    'Plots',"training-progress", ...  % Plot training progress
    "Verbose", false, ...  % Do not display training progress in command window
    "ExecutionEnvironment","multi-gpu", ...  % Use multiple GPUs if available
    "ValidationData",dsVal, ...  % Data for validation
    "ValidationFrequency",15, ..., % Frequency of validation
    "Shuffle", "every-epoch"); %shuffle training images every epoch

% Train the network with the specified training options
net = trainNetwork(dsTrain, layers, opts);

% Save the trained network
save('segmentnet.mat', 'net')

% Perform semantic segmentation on the test images using the trained network
pxdsResults = semanticseg(imdsTest,net,"WriteLocation","out");

% Overlay the segmentation results on the first image in the test set
overlayOut = labeloverlay(readimage(imdsTest,1),readimage(pxdsResults,1));
figure
imshow(overlayOut);
title('overlayOut')

% Evaluate the semantic segmentation results
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest);

% Plot the confusion matrix
figure
cm = confusionchart(metrics.ConfusionMatrix.Variables, ...
classNames, Normalization='row-normalized');
cm.Title = 'Normalized Confusion Matrix (%)';

% Compute and plot the mean Intersection over Union (IoU) for each image
imageIoU = metrics.ImageMetrics.MeanIoU;
figure
histogram(imageIoU)
title('Image Mean IoU')

% Overlay the segmentation results on some test images with the true labels and predicted labels side by side
numTestImages = 5;  % number of test images to display
for i = 1:numTestImages
    % Read the test image
    I = readimage(imdsTest, i);
    
    % Read the true labels
    trueLabels = readimage(pxdsTest, i);
    
    % Perform semantic segmentation to get the predicted labels
    predictedLabels = semanticseg(I, net);
    
    % Overlay the true labels on the test image
    overlayTrue = labeloverlay(I, trueLabels, 'Transparency', 0.6);
    
    % Overlay the predicted labels on the test image
    overlayPredicted = labeloverlay(I, predictedLabels, 'Transparency', 0.6);
    
    % Display the test image and the overlays
    figure
    subplot(1,3,1)
    imshow(I)
    title('Test Image')
    
    subplot(1,3,2)
    imshow(overlayTrue)
    title('True Labels')
    
    subplot(1,3,3)
    imshow(overlayPredicted)
    title('Predicted Labels')
end

% Using Matlab's partitionCamVidData function to divide the data into training, validation, and testing sets.
% https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html

function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds)
% Partition CamVid data by randomly selecting 60% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
numTrain = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 20% of the images for validation
numVal = round(0.20 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use 20% of the images for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = 1:numel(pxds.ClassNames);

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end

% Useing Matlab's The augmentImageAndLabel function for 
% augmentation for increasing the size of the training dataset 
% It applies random reflection and translation to the images 
% and the corresponding pixel label images. 
% https://uk.mathworks.com/help/vision/ug/semantic-segmentation-using-deep-learning.html

function data = augmentImageAndLabel(data, xTrans, yTrans)
% Augment images and pixel label images using random reflection and
% translation.

for i = 1:size(data,1)
    
    tform = randomAffine2d(...
        'XReflection',true,...
        'XTranslation', xTrans, ...
        'YTranslation', yTrans);
    
    % Center the view at the center of image in the output space while
    % allowing translation to move the output image out of view.
    rout = affineOutputView(size(data{i,1}), tform, 'BoundsStyle', 'centerOutput');
    
    % Warp the image and pixel labels using the same transform.
    data{i,1} = imwarp(data{i,1}, tform, 'OutputView', rout);
    data{i,2} = imwarp(data{i,2}, tform, 'OutputView', rout);
    
end
end