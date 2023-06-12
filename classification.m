% This is a implementation of flower image classification on the oxford-17 flowers dataset
% using a transfer learning approach where I utilize the pre-trained ResNet-50 network. 

% The dataset used for this task is stored in the './17flowers' directory, 
% with each subfolder representing a different class with each class contatining 80 images.
% The labels for the images are determined from the names of those subfolders.

% The data is split into training and validation sets in a 70:30 ratio. 
% The training set is used to fine-tune the ResNet-50 model, 
% and the validation set is used to evaluate the performance of the network.

% Matlab's Transfer Learning tutorial was used for guuidance during this implementation.
% https://uk.mathworks.com/help/deeplearning/ug/transfer-learning-using-pretrained-network.html
 
% Declare the directory where the images are stored
imageDir = '.\17flowers';

% Create an image datastore, which is a repository for the image data
imds = imageDatastore(imageDir, ...
 'IncludeSubfolders',true, ...  % Include subfolders in the datastore
 'LabelSource','foldernames');  % Use the folder names as source of labels

% Split the images into training and validation datasets (70% training, 30% validation)
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

% Load the pre-trained ResNet-50 network
net = resnet50;


%For the ResNet-50 model, the default inputSize is [224, 224, 3]
% This means that the network is designed to take in images of size 224x224

% Get the input size of the network
inputSize = net.Layers(1).InputSize;

% Create a layer graph from the pre-trained network
lgraph = layerGraph(net); 

% Get the number of classes (number of unique labels)
numClasses = numel(categories(imdsTrain.Labels));

% Define a new learnable fully connected layer with the same number of neurons as the number of classes
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...  % Name of the layer
    'WeightLearnRateFactor',10, ...  % Learning rate factor for the weights
    'BiasLearnRateFactor',10);  % Learning rate factor for the biases
    
% Replace the last fully connected layer 'fc1000' in the network with the new learnable layer
lgraph = replaceLayer(lgraph,'fc1000',newLearnableLayer);

% Define a new classification output layer
newClassLayer = classificationLayer('Name','new_classoutput');

% Replace the last classification output layer in the network with the new layer
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);

% Data augmentation is applied to the training set to improve the 
% network's ability to generalize and prevent overfitting.

% Define the range for random pixel translation during data augmentation
pixelRange = [-30 30];
% Define an image augmenter for data augmentation during training
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...  % Randomly reflect images along the vertical axis
    'RandXTranslation',pixelRange, ...  % Randomly translate images along the horizontal axis
    'RandYTranslation',pixelRange);  % Randomly translate images along the vertical axis

%inputSize(1:2) corresponds to the desired size of the images, which is [224, 224] 
% for ResNet-50. The augmentedImageDatastore function creates an augmented image 
% datastore that automatically resizes all images to this size when they are read 
% during the training of the network.

%The image resizing is performed using bicubic interpolation, 
% which is the default resizing method in Matlabs's imresize function. 
% This method works by fitting a smooth curve through the surrounding pixels 
% of each input pixel to calculate the output pixel value, 
% which helps to maintain the visual quality of the image after resizing.

% Create an augmented Image Datastore for training
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

% Specify the training options
options = trainingOptions('sgdm', ...  % Use stochastic Gradient Descent with momentum
    'MiniBatchSize',10, ...  % Number of observations per mini-batch
    'MaxEpochs',6, ...  % Maximum number of epochs to train
    'InitialLearnRate',1e-4, ...  % Initial learning rate
    'Shuffle','every-epoch', ...  % Shuffle data every epoch
    'ValidationData',augimdsTrain, ...  % Data for validation
    'ValidationFrequency',3, ...  % Frequency of validation
    'Verbose',false, ...  % Do not display training progress in command window
    'Plots','training-progress');  % Plot training progress

% Train the network with the specified training options
netTransfer = trainNetwork(augimdsTrain,lgraph,options);

% Save the trained network
save('classnet.mat', 'netTransfer');

% Create an augmented Image Datastore for validation
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Classify the validation images using the trained network
[YPred,scores] = classify(netTransfer,augimdsValidation);

% Randomly select 4 images from the validation set for display
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

% Get the ground truth labels from the validation set
YValidation = imdsValidation.Labels;

% Calculate the classification accuracy
accuracy = mean(YPred == YValidation)

% Generate the confusion matrix
confMat = confusionmat(YValidation, YPred)

% Display the confusion matrix
figure
confusionchart(confMat, categories(YValidation))
