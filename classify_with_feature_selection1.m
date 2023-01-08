function [accuracy, sensitivity, specificity, fs] = classify_with_feature_selection1(x,y,numOfFolds,C)
% input: x ,y data out ..., fs selected features needs table creation
% C, penalty for feature selection & classification
% output: accuracy, sensitivity, specificity, 
% fs, is logical index vector for chosen features from sequential
% feature selection

c = cvpartition(y,'k',numOfFolds, 'Stratify',false);% crossvalidation partitions
opts = statset('display','off');

classifier = @(train_data,train_labels,test_data,test_labels)classf(train_data,train_labels,test_data,test_labels,C);
% standardize data
x=zscore(x);
[fs,~] = sequentialfs(classifier,x,y,'cv',c,'mcreps',1,'options',opts,'direction', 'forward');

total_conf = zeros(2,2); %total confusion matrix : sum of confusion matrix of each fold
for j=1:numOfFolds
    
    train_data = x(c.training(j),fs);
    test_data = x(c.test(j),fs);
    train_labels = y(c.training(j));
    test_labels = y(c.test(j));
%     opts = struct('Optimizer','bayesopt','ShowPlots',false,...
%     'AcquisitionFunctionName','expected-improvement-plus','Verbose',0);
%     svm = fitcsvm(train_data,train_labels,'KernelFunction','polynomial','PolynomialOrder', 5,'Cost',[0 C; 1 0],'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);

    svm = fitcsvm(train_data,train_labels,'KernelFunction','rbf','Cost',[0 C(1); C(2) 0],'KernelScale','auto');% ,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
    predicted_labels = predict(svm, test_data);
    conf = confusionmat(test_labels, predicted_labels);
%     if length(conf)==1 %case where all test labels are from same class
%         if all(test_labels)
%             conf = [conf,0;0,0];
%         else
%             conf = [0,0;0,conf];
%         end
%     end
    total_conf = total_conf + conf;
end

%mean accuracy sensitivity and specificity
accuracy = (total_conf(1,1)+total_conf(2,2))*100/sum(total_conf, 'All');
sums = sum(total_conf, 2);
sensitivity = total_conf(2,2)*100/sums(2); %with Kinova accuracy
specificity = total_conf(1,1)*100/sums(1);% without kinova accuracy
end