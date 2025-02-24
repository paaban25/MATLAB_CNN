clc; clear; close all;

%% 1. Load the MNIST Dataset
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% 2. Split Dataset (80% Train, 20% Validation)
[trainData, valData] = splitEachLabel(imds, 0.8, 'randomized');
% trainData.Files and valData.Files contains images of all Labels

%% 3. Get Labels & Convert to Categorical
trainLabels = categorical(trainData.Labels);
valLabels = categorical(valData.Labels);
% trainLabels and valLabels contains the labels of the corresponding image.

%% 4. Preprocess Images (Resize & Augment)
inputSize = [28 28]; % Target size for CNN
augTrain = augmentedImageDatastore(inputSize, trainData);
augVal = augmentedImageDatastore(inputSize, valData);

%% 5. Define CNN Parameters
inputSize = [28 28 1]; % 28x28 Grayscale images
numClasses = numel(unique(trainLabels)); % Number of digits (0-9)

% Initialize Weights and Biases (Random)
filterSize = [3 3]; % 3x3 convolutional filters
numFilters = 8; % Number of filters
learningRate = 0.01; % Learning rate

convWeights = randn(filterSize(1), filterSize(2), inputSize(3), numFilters) * 0.01;
convBiases = zeros(1, 1, numFilters);

fcWeights = randn(10, numFilters * 26 * 26/4) * 0.01;
fcBiases = zeros(10, 1);

ph=2;
pw=2;
stride_pooling =2;
stride_conv = 1;
%% 6. Training CNN (One Image at a Time)
numEpochs = 1;

for epoch = 1:numEpochs
    fprintf('Epoch %d/%d\n', epoch, numEpochs);

    % Reset the datastore for each epoch
    trainData.reset;

    trainItems = size(augTrain.Files);
    numtrainItems = trainItems(1);
    for i=1:numtrainItems
        XImg = readimage(trainData, i); % Read the actual image
        XImg = im2gray(XImg); % Convert to grayscale if needed
        XImg = double(XImg) / 255.0; % Normalize the pixel values
        label = trainLabels(i);


        YLabel = onehotencode(label, 2, 'ClassNames', categories(trainLabels));
        YLabel = YLabel';

        % Forward pass (Using your convolution function)
        convOutput = convolutional_layer(XImg, convWeights, convBiases, stride_conv); % Using your function
        reluOutput = relu_activation(convOutput); % ReLU Activation
        maxPoolingOutput = max_pooling (reluOutput,ph,pw,stride_pooling);
        flattenedOutput = maxPoolingOutput(:);


        % Use approximate multiplication for the fully connected layer
        fcOutput = fully_connected_layers(flattenedOutput,fcWeights,fcBiases); 
        Output = output_layer(fcOutput); % Softmax Activation

        % Compute Loss (Cross-Entropy)
        % loss = -sum(YLabel .* log(softmaxOutput));
        loss = cross_entropy_loss (YLabel,Output);

        [fcWeights,fcBiases,convWeights,convBiases] = backpropagation(XImg,YLabel,fcWeights,fcBiases,convWeights,convBiases,convOutput,maxPoolingOutput,flattenedOutput,learningRate);

    end
    fprintf('Loss at epoch %d: %.4f\n', epoch, loss);
end



%% 7. Validation
correct = 0; total = 0;
valItems = size(augVal.Files);
numValItems = valItems(1);

for j=1: numValItems
    XImg_val = readimage(valData, j); % Read the actual image
    XImg_val = im2gray(XImg_val); % Convert to grayscale if needed
    XImg_val = double(XImg_val) / 255.0; % Normalize the pixel values
    label_val = valLabels(j);

    YLabel_val = onehotencode(label_val, 2, 'ClassNames', categories(valLabels));
    YLabel_val = YLabel_val'


    % Forward pass (Using your convolution function)
        convOutput_val = convolutional_layer(XImg_val, convWeights, convBiases, stride_conv); % Using your function
        reluOutput_val = relu_activation(convOutput_val); % ReLU Activation
        maxPoolingOutput_val = max_pooling (reluOutput_val,ph,pw,stride_pooling);
        flattenedOutput_val = maxPoolingOutput_val(:);
        
        
        % Use approximate multiplication for the fully connected layer
        fcOutput_val = fully_connected_layers(flattenedOutput_val,fcWeights,fcBiases); 
        Output_val = output_layer(fcOutput_val); % Softmax Activation

        [~, predLabel] = max(Output_val);

        % Compute Accuracy
        correct = correct + (predLabel == YLabel_val);
        total = total + 1;

end

fprintf('Validation Accuracy: %.2f%%\n', (correct / total) * 100);