function score = classf(train_data,train_labels,test_data,test_labels,C)
classifier = fitcsvm(train_data,train_labels,'KernelFunction','linear');%'polynomial','PolynomialOrder',5); 
% classifier = fitcsvm(train_data,train_labels,'KernelFunction','polynomial','PolynomialOrder',2, 'Standardize',true);
% classifier = fitcknn(train_data,train_labels,'NumNeighbors',3);
res = predict(classifier , test_data);

% score = 0;
% for i = 1:length(test_labels)
%     if res(i) ~= test_labels(i)
%         if test_labels(i) == 0%improve specificity
%             score = score + C(1);
%         else
%             score = score + c(2);
%         end
%     end
miss = logical(test_labels(res ~= test_labels));
% Specificity penalty
spec = C(1)* sum(~miss);
% Sensitivity penalty
sens = C(2)*sum(miss);
score = sens + spec;


end