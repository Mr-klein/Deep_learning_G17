clc
clear all
close all

%%
fs = 0:1:3; % fs = [[6,12,18,24],[12,24,36,48],[24,48,72,96],[48,96,144,192]]
lr = 0:1:2; % Lr = [0.01, 0.005, 0.001]
mo = 0:1:3; % Momentum = [0.8, 0.85, 0.9, 0.95]

results = [];
for k = 1:size(mo,2)
    for j = 1:size(lr,2)
        for i = 1:size(fs,2)
            filename = "../results/network_optimizer_results/resultfs"+fs(i)+"lr"+lr(j)+"mo"+mo(k)+".txt";
            results = [results, [i;j;k;textread(filename)]];
        end
    end
end