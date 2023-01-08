%% Section 1 : setting some parameters for classification
clear 
% Labels used for classification can be either for device usage or movement success(score) 
% var device_or_score: should be equal to 1 or 2:
%'1' histograms and classification on device Kinova-O540 usage (with or without Kinova)
%'2' histograms and classification on score (movement success vs failure)
device_or_score = 1; 
% var score_condition: value is any from [1, 2, 3]
% '1' histograms and classification on all data (with and without device),
% '2' ... on data with device usage,
% '3' ... on data without device usage,
% NOTE: Ignore setting this variable if classifying on device usage (i.e.
% set its value only if device_or_score = 2
score_condition = 1;

% Applying chosen parameters
device_score = ["Device", "Score"];
device_score_choice = convertCharsToStrings(device_score{device_or_score});

mov_names ={'movA';'movB';'movC';'movD';'movE';'movF';'movX'};
condition = {'Without', 'With'};

% Following if/else sets some params depending on previous variables choices
if device_or_score == 2 % Classification on Score
    if score_condition == 1 %All combined
        % var loop_array: if == 1 we extract features from data using
        % the device, if == 2, we use data without device, if == [1, 2] we
        % use both
        loop_array = 1:2; %meaning 'All' uses With and without
    elseif score_condition == 2 %With device
        loop_array = 2;
    else % Without device
        loop_array = 1;
    end      
else % Classification on deivce (device usage or not), we use data with and without the deice
    loop_array = 1:2;
end
% 'data' structure that contains all participants data, 3 fields named:
% 'ID', 'with' and 'without', inside each of them is another struct 
% containing all movements per participant and the score (aka
% movement success vs failure). Each movement name is a field of this
% struct, whose value is a struct made of 2 fields : 'data' and 'score'.
% 'data' field is a cell array, each cell contains a vector correponding to one
% sample of the magnitude of the accelerometer measurmenets. While 'score'
% field is made of a cell array containig the corresponding score of each sample

% load the provided "sample_data.mat" file which shows how 'data' variable
% looks like. Notice, this is not real data, So the code will run but the 
% results are meaningless. All the field values are equal,
% however the code can run with different number of data samples per
% movement and each data sample can be of different size (different time 
% length for each data sample)
load('dummy_data.mat','data');

%% Section 2 : feature extraction
% var features: struct that stores the computed features, with fields corresponding to the all
% movements, each field contains a matrix of the data, with number of rows
% corresponding to number of samples, and the number of columns coressponding
% to the number of features (here 7)
% var labels: struct that stores the labels of the movements, contains 2 fields,
% each field containes a struct similar to 'features', which has labels for each
% movement. The field 'Score' contains labels of success vs failure, the
% field 'Device' contains labels of device usage(with) or no usage (without)
% Notice! Labels Y.Label: 0 == without Kinova & 1 ==with Kinova, Y.Score:
% 0 == failure & 1 == success

feature_names = {'Mean','Std','Entropy','Min','Max', 'Median', 'Energy', 'Normalized Energy', 'Time'}; %used later
features = cell2struct(cell(length(mov_names),1),mov_names);

labels = struct();
labels.Score = cell2struct(cell(length(mov_names),1),mov_names);
labels.Device = cell2struct(cell(length(mov_names),1),mov_names);
for j = 1:length(data) % Iterate over all participants
    for k = loop_array %1 without, 2 with, [1,2] both
        if ~isempty(data(j).(condition{k})) % if data is found
            for m=1:length(mov_names) % Iterate over all movements data
                for l=1:length(data(j).(condition{k}).(mov_names{m}).data)
                    if ~isempty(data(j).(condition{k}).(mov_names{m}).data{l})          
                        tmp = data(j).(condition{k}).(mov_names{m}).data{l};
                        x(1)= mean(tmp);
                        x(2)= std(tmp);
                        x(3)= entropy_(tmp);
                        x(4)= min(tmp);
                        x(5)= max(tmp);
                        x(6) = median(tmp);
                        x(7) = sum(tmp.^2);
                        x(8) = rms(tmp)^2;
                        x(9) = length(tmp); %sampling freq = 1s, so length corresponds to seconds
                        features.(mov_names{m}) = [features.(mov_names{m}); x];
                        labels.Device.(mov_names{m}) = [labels.Device.(mov_names{m}); k-1]; %0 without or 1 with Kinova
                        labels.Score.(mov_names{m}) = [labels.Score.(mov_names{m}); data(j).(condition{k}).(mov_names{m}).score{l} ];
                    end
                end
            end
        end
    end
end

%% Section 3 : feature selection - 1
% Note this section corresponds to the results of section 3.1 and 3.2 in the paper
% where feature selection is done using 5-fold cross-validation repeated 
% 100 MC reps, and then the performance of the chosen features are 
% validated using 5-fold cross validation 

numOfFolds = 5 ; % number of folds for crossvalidaiton
varTypes = {'string','double','double','double'};
chosen_features = struct();% containes features chosen by sequential feature
% selection for each movement and for each fold
% var performance: struct containing accuracy, sensitivity and specificty
% for the movements chosen in mov2analyze
performance = struct();
% Choosing a subset of movements to analyze from all movements stored in
% the struct 'X', and the struct 'data'
mov2analyze = ["movD", "movE"];
% var C: penalties to use depending on the type of classification, C(1)
% penalty for label ==0, C(2) penalty for label == 1, Note that 0 and 1
% depend on which type of classification was chosen in the first section
% whether on Score or device usage (see feature selection section), this
% variable is used to adjust for data imbalance if needed and its value is adjusted
% later depending on the number of samples for labels 0 & 1
C = [1, 1];   
% var T: table used to store results
T = table('Size',[length(mov2analyze),4],'RowNames',mov2analyze, 'VariableNames', {'Chosen Features','Accuracy','Sensitivity','Specificity'}, 'VariableTypes',varTypes);

for j = 1:length(mov2analyze) % Iterate over chosen movements to analyze
    fprintf(['\n','--- ',mov2analyze{j},' ---','\n'])
%     C(1) = length(labels.(device_score_choice).(mov2analyze{j}))/ sum(labels.(device_score_choice).(mov2analyze{1}) == 0) ;
%     C(2) = length(labels.(device_score_choice).(mov2analyze{j}))/ sum(labels.(device_score_choice).(mov2analyze{1})) ;
    x=features.(mov2analyze(j));
    y=labels.(device_score_choice).(mov2analyze(j));
    if length(y)>2
        % Classifying and getting performance and chosen features
        [accuracy, sensitivity, specificity, fs] = classify_with_feature_selection1(x,y,numOfFolds,C);

        %saving chosen features
        chosen_features.(mov2analyze(j)) = fs;

        performance.(mov2analyze(j)).Accuracy = accuracy;
        performance.(mov2analyze(j)).Sensitivity = sensitivity;
        performance.(mov2analyze(j)).Specificity = specificity;
        performance.(mov2analyze(j)).Features = fs;
        if any(isnan([accuracy, sensitivity, specificity])) % if no results
            T(movs2analyse(j),2:4) = repelem({000},3);
        else
            T(mov2analyze(j),1) = join(feature_names(fs));
            T(mov2analyze(j),2) = {accuracy};
            T(mov2analyze(j),3) = {sensitivity};
            T(mov2analyze(j),4) = {specificity};
        end
        fprintf(['\n','Accuracy = ',num2str(accuracy), '%%; Sensitivity = ',num2str(sensitivity), '%%; Specificity = ',num2str(specificity),'%%;\n'])
    else
        % number of smaples too small
        fprintf("Insufficient Samples, i.e. number of samples <=2")
    end
end

%% Section 4 : feature selection - 2 
% Note this section corresponds to the results of section 3.3 in the paper
% where 3-fold cross-validation is performed, and the feature selection is
% done for each fold separately (with 100 MC reps)
disp('Used features in order :')
disp(feature_names)
numOfFolds = 3;%use 3 with "with Kinova" subgroup
chosen_features = struct();% containes features chosen by sequential feature
% selection for each movement and for each fold
% var performance: struct containing accuracy, sensitivity and specificty
% for the movements chosen in mov2analyze
performance = struct();
% Choosing a subset of movements to analyze from all movements stored in
% the struct 'X', and the struct 'data'
mov2analyze = ["movD", "movE"];
% var C: penalties to use depending on the type of classification, C(1)
% penalty for label ==0, C(2) penalty for label == 1, Note that 0 and 1
% depend on which type of classification was chosen in the first section
% whether on Score or device usage (see feature selection section), this
% variable is used to adjust for data imbalance if needed and its value is
% adjusted later depending on the number of samples for labels 0 & 1
C = [1, 1];   
% var T: table used to store results
T=table('Size',[numOfFolds*length(mov2analyze), 6],'VariableTypes',{'double', 'double','string', 'double', 'double', 'double'},'VariableNames',{'Mov','fold','Features','Accuracy','Sensitivity','Specificity'});

for j = 1:length(mov2analyze) % Iterate over chosen movements to analyze
    fprintf(['\n','--- ',mov2analyze{j},' ---','\n'])
%     C(1) = length(labels.(device_score_choice).(mov2analyze{j}))/ sum(labels.(device_score_choice).(mov2analyze{1}) == 0) ;
%     C(2) = length(labels.(device_score_choice).(mov2analyze{j}))/ sum(labels.(device_score_choice).(mov2analyze{1})) ;
    x=features.(mov2analyze(j));
    y=labels.(device_score_choice).(mov2analyze(j));
    if length(y)>2
        % Classifying and getting performance and chosen features
        [accuracy, sensitivity, specificity, fs, fs_names] = classify_with_feature_selection2(x,y,numOfFolds,C, feature_names);

        %saving chosen features
        chosen_features.(mov2analyze(j)) = fs;

        performance.(mov2analyze(j)).Accuracy = accuracy;
        performance.(mov2analyze(j)).Sensitivity = sensitivity;
        performance.(mov2analyze(j)).Specificity = specificity;
        performance.(mov2analyze(j)).Features = fs;
        T((j-1)*numOfFolds+1:j*numOfFolds,1) = num2cell(j*ones(numOfFolds,1));
        T((j-1)*numOfFolds+1:j*numOfFolds,2) = num2cell([1:numOfFolds]');
        T((j-1)*numOfFolds+1:j*numOfFolds,3) = fs_names;
        T((j-1)*numOfFolds+1:j*numOfFolds,4) = num2cell(round(accuracy',2));
        T((j-1)*numOfFolds+1:j*numOfFolds,5) = num2cell(round(sensitivity',2)); 
        T((j-1)*numOfFolds+1:j*numOfFolds,6) = num2cell(round(specificity',2));

        fprintf(['\n','Accuracy = ',num2str(accuracy), '%%; Sensitivity = ',num2str(sensitivity), '%%; Specificity = ',num2str(specificity),'%%;\n'])
    else
        % number of smaples too small
        fprintf("Insufficient Samples, i.e. number of samples <=2")
    end
end

%% Section 5 : Features histogram
% Note chosen features must contain only 1 vector per movement, i.e. you
% can use it after running section 3 (not 4) of the code
close all
mov2analyze = ["movD", "movE"];
display_names = ["Movement 1","Movement 2"]; % name of the chosen movements for figure titles
histogram_device_score = "Score" ;% display results for chosen features for
% device or score. Note, meaningless if set to "Device" when device_or_score = 2, and  
% score_condition = 2 or 3, because all labels will be only zeros or ones

for j = 1:length(mov2analyze) %movments loop
    x=features.(mov2analyze(j));
    y=labels.(histogram_device_score).(mov2analyze(j));
    h(j) = figure();
    used_features = find(chosen_features.(mov2analyze{j}));% if all features this should be [1 2 3 ... length(features)]
    t = tiledlayout(length(used_features),1,'TileSpacing','Compact','Padding','Compact','Parent',h(j));
    title(t,join([display_names(j),'histograms']),'FontSize',15)
    for k=used_features % j = features loop
        x_tmp=x(:,k);
        nexttile;
        %hs.Position = hs.Position + [-0.1 -0.1 0.1 0.1];
        h_without = histogram(x_tmp(y==0),20 ,'Normalization', 'Probability');
        hold on
        h_with = histogram(x_tmp(y==1), 20, 'Normalization', 'Probability');
        set(gca,'FontSize',12);
        if histogram_device_score == "Device"
            legend('No-KINOVA', 'KINOVA')
        else
            legend('Score = 0', 'Score =  1')
        end
        title(feature_names{k})
    end
    title_ = join([display_names(j),"chosen features histograms"]," ");
    set(h(j),'color','w');
    
end

%% Section 6 : success ratio plots
% NOTE! do not use this feature after running section 4, use with section 3

% load(join(['Results/',combined_or_not,'/', device_and_score_condition, '/', 'chosen_features'],''))
close all
%mov D & E correspond to movements '1' and '2' in the paper
mov2analyze = ["movD", "movE"];
display_names = ["Movement 1","Movement 2"];

for ii = 1:length(mov2analyze)
    fs = find(chosen_features.(mov2analyze{ii}));
    x_ = features.(mov2analyze(ii));
    y_ = labels.Score.(mov2analyze(ii));  
    feature_count = length(fs);
    hx = figure();
    set(hx,'color','w');
    
    t=tiledlayout(feature_count,1,'TileSpacing','Compact','Padding','Compact','Parent',hx);
    name = strcat("Success rates of ", display_names(ii), " using ", join(feature_names(fs),"-"));
    title(t,name,'FontSize',15)
    % j = features loop
    for j=1:feature_count
        nexttile;
        for l=0:1    % "without" and "with" loop
            hold on
            idx = labels.Device.(mov2analyze(ii))==l;
            x = x_(idx,:);
            y = y_(idx);
            x_tmp = x(:,fs(j));
            feature_values = linspace(min(x_(:,fs(j))), max(x_(:,fs(j))), 100);  %length(feature_values) = 100
            ratio = zeros(length(feature_values),1);
            success = zeros(length(feature_values),1);
            count = zeros(length(feature_values),1);
            for k = 1:length(feature_values)
                res = x_tmp<=feature_values(k);
                count_tmp = sum(res);
                success_tmp = sum(y(res)==1);
                count(k) = count_tmp;
                success(k) = success_tmp;
                ratio(k) = 100*success_tmp/count_tmp;
            end
            
            %hs.Position = hs.Position + [-0.1 -0.1 0.1 0.1];
            plot(feature_values, ratio, 'LineWidth',2);
            xlim([-inf,inf])
            ylim([0,100])
            xlabel(feature_names{fs(j)})
            ylabel("Success rate %")
            set(gca,'FontSize',12);
            
        end
        legend('Without Kinova','With Kinova','location','southeast')
        
    end

end