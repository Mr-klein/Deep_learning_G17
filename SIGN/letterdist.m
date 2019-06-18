clc
clear all
close all

%% get data
% train set
trainfilename = 'sign_mnist_train.csv';
traindata = getData(trainfilename);

% test set
testfilename = 'sign_mnist_test.csv';
testdata = getData(testfilename);

% labels
labels = {'A';'B';'C';'D';'E';'F';'G';'H';'I';'J';'K';'L';'M';'N';'O';'P';'Q';'R';'S';'T';'U';'V';'W';'X';'Y';'Z'};

%% count number of letter occurences
% train set
trainlabels = traindata(:,1);
trainbin = countData(trainlabels);

% test set
testlabels = testdata(:,1);
testbin = countData(testlabels);

%% sort data
% train set
[strainbin, strainlabels] = sortData(trainbin, labels, 'descend');

% test set
[stestbin, stestlabels] = sortData(testbin, labels, 'descend');

%% calculations
% train set
trainmean = mean(strainbin);          % mean
trainstd = std(strainbin);            % standard deviation
trainbinn = strainbin/sum(strainbin); % normalized for plotting
trainmeann = mean(trainbinn);         % normalized mean
trainstdn = std(trainbinn);           % normalized standard deviation

% test set
testmean = mean(stestbin);            % mean
teststd = std(stestbin);              % standard deviation
testbinn = stestbin/sum(stestbin);    % normalized for plotting
testmeann = mean(testbinn);           % normalized mean
teststdn = std(testbinn);             % normalized standard deviation

%% plot
% train set
figure(1); clf(1);
bar(0:1:25,trainbinn,'b'); hold on
plot(xlim,[trainmeann trainmeann], 'r--')
plot(xlim,[trainstdn trainstdn], 'g--')
set(gca,'xtick',[0:25],'xticklabel',strainlabels,'FontSize',15)
title(['Training set distribution of ',num2str(size(trainlabels,1)),' samples in descending order'])
xlabel('Letters in descending order')
ylabel('Occurrences of letters [%]')
legend('data distribution',['data mean = ',num2str(trainmeann)],['data std = ',num2str(trainstdn)])

% test set
figure(2); clf(2);
bar(0:1:25,testbinn,'b'); hold on
plot(xlim,[testmeann testmeann], 'r--')
plot(xlim,[teststdn teststdn], 'g--')
set(gca,'xtick',[0:25],'xticklabel',stestlabels,'FontSize',15)
title(['Test set distribution of ',num2str(size(testlabels,1)),' samples in descending order'])
xlabel('Letters in descending order')
ylabel('Occurrences of letters [%]')
legend('data distribution',['data mean = ',num2str(testmeann)],['data std = ',num2str(teststdn)])

%% FUNCTIONS
% retrieve data from csv files
function data = getData(filename)
    data = csvread(filename,1);
end

% sort data in ascending or decending order
function [sorted_data, new_labels] = sortData(data,labels,option)
    [sorted_data, new_indices] = sort(data,option);
    new_labels = labels(new_indices);
end

% count the number of occurences in data
function bin = countData(data)
    bin = zeros(26,1);
    for i = 0:25
        if i == 9 || i == 25
            continue
        else
            bin(i+1,1) = sum(data(:) == i);
        end
    end
end