function [accuracy, sensitivity, specificity, fs, fs_names] = classify_with_feature_selection2(x,y,numOfFolds,C,feature_names)
% input: x ,y data out ..., fs selected features needs table creation,
% C, penalty for eature selection
% numOfFolds, number of folds
% feature_names: names of the used features
% output: accuracy, sensitivity, specificity, fs, fs is logical index matric for
% chosen features from sequential feature selection, each row correspond to
% one fold, columns correspond to the chosen features
% fs_names, the corresponding names for the chosen features

c = cvpartition(y,'k',numOfFolds, 'Stratify',true);% Crossvalidation partitions
opts = statset('display','off');%iter

classifier = @(train_data,train_labels,test_data,test_labels)classf(train_data,train_labels,test_data,test_labels,C);

% Standardizing
x=zscore(x);

accuracy = zeros(1,numOfFolds); sensitivity = zeros(1,numOfFolds); specificity = zeros(1,numOfFolds);
fs = false(numOfFolds, size(x, 2));
fs_names = cell(numOfFolds,1);
parfor j=1:numOfFolds
    
    train_data = x(c.training(j),:);
    test_data = x(c.test(j),:);
    train_labels = y(c.training(j));
    test_labels = y(c.test(j));
    [fs(j,:),~] = sequentialfs(classifier,train_data, train_labels,'cv', 3, 'mcreps', 1, 'options', opts, 'direction', 'forward');
    fs_names{j} = strjoin(feature_names(fs(j,:)),', '); 
%     opts = struct('Optimizer','bayesopt','ShowPlots',false,...
%     'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);
%     svm = fitcsvm(train_data,train_labels,'KernelFunction','polynomial','PolynomialOrder', 2,'Cost',[0 C; 1 0],'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);

    svm = fitcsvm(train_data(:,fs(j,:)),train_labels,'KernelFunction','linear','Cost',[0 C(1); C(2) 0],'KernelScale','auto');% ,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
    predicted_labels = predict(svm, test_data(:,fs(j,:)));
    conf = confusionmat(test_labels, predicted_labels);
%     if length(conf)==1 %case where all test labels are from same class
%         if all(test_labels)
%             conf = [conf,0;0,0];
%         else
%             conf = [0,0;0,conf];
%         end
%     end
    accuracy(j) = (conf(1,1)+conf(2,2))*100/sum(conf, 'All');
    sums = sum(conf, 2);
    sensitivity(j) = conf(2,2)*100/sums(2); %with Kinova accuarcy
    specificity(j) = conf(1,1)*100/sums(1);% without kinova accuracy
end
