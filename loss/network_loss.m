clc
clear all
close all

%% obtain data
filename = 'lossfs3lr1mo3.txt';
fileID = fopen(filename,'r');
formatSpec = '%f';
loss = (fscanf(fileID,formatSpec));
sloss = (smoothdata(loss,'movmean',12));

%%
x = (1:1:size(loss,1))*100/3400;

%%
figure(1); clf(1);
% plot(x,loss,'b');
hold on;
plot(x,sloss,'r','LineWidth',2);
%line([3400 3400], get(gca, 'ylim'));
set(gca, 'YScale', 'log','FontSize',12);
title('Loss Curve for 50 epochs')
ylabel('Loss (log scale)')
xlabel('# Epochs')
grid on