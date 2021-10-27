len = 21;
%% loading for 5 methods without attack
% blue = readstrdata(importexcel(3:3+len-1,1), len);
% red = readstrdata(importexcel(109:109+len-1,1), len);
% yellow = readstrdata(importexcel(56:56+len-1,1), len);
% purple = readstrdata(importexcel(214:214+len-1,1), len);
% green = readstrdata(importexcel(161:161+len-1,1), len);
%% loading for 6 methods
col = 10; % 5 for signflip, 10 for gaussian
blue = readstrdata(importexcel(3:3+len-1,1), len);
red = readstrdata(importexcel(89:89+len-1,col), len);
yellow = readstrdata(importexcel(56:56+len-1,col), len);
purple = readstrdata(importexcel(122:122+len-1,col), len); 
% warn: col=5: row start 123; col=10: row start 122
green = readstrdata(importexcel(156:156+len-1,col), len);
brightblue = readstrdata(importexcel(189:189+len-1,col), len);
%% loading for 5 lambdas
% blue = readstrdata(importexcel(3:3+len-1,15), len);
% red = readstrdata(importexcel(56:56+len-1,15), len);
% yellow = readstrdata(importexcel(109:109+len-1,15), len);
% purple = readstrdata(importexcel(162:162+len-1,15), len);
% green = readstrdata(importexcel(215:215+len-1,15), len);

%% create figures
figure('InvertHardcopy','off','PaperUnits','centimeters','Color',[1 1 1],...
    'OuterPosition',[1 1 640 360]);

% Create axes
axes1 = axes('Units','pixels');
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot1 = plot(blue(:,1),[blue(:,2),red(:,2),yellow(:,2),purple(:,2),green(:,2),...
    brightblue(:,2)],'MarkerIndices',1:2:21,...
    'MarkerSize',10,...
    'LineWidth',2);
%% labels for 5 methods without attack
% set(plot1(1),'DisplayName','Ideal SGD','Marker','diamond');
% set(plot1(2),'DisplayName','RSA','Marker','v');
% set(plot1(3),'DisplayName','Stochastic ADMM','Marker','^');
% set(plot1(4),'DisplayName','Geometric median','Marker','>');
% set(plot1(5),'DisplayName','Median','Marker','<');
%% labels for 6 methods
set(plot1(1),'DisplayName','Ideal SGD','Marker','diamond');
set(plot1(2),'DisplayName','RSA','Marker','v');
set(plot1(3),'DisplayName','Stochastic ADMM','Marker','^');
set(plot1(4),'DisplayName','Geometric median','Marker','>');
set(plot1(5),'DisplayName','Median','Marker','<');
set(plot1(6),'DisplayName','SGD','MarkerSize',15,'Marker','pentagram',...
    'Color',[0.301 0.745 0.933]);
%% labels for 5 lambdas
% set(plot1(1),'DisplayName','\lambda = 0.01','Marker','diamond');
% set(plot1(2),'DisplayName','\lambda = 0.1','Marker','v');
% set(plot1(3),'DisplayName','\lambda = 0.5','Marker','^');
% set(plot1(4),'DisplayName','\lambda = 1','Marker','>');
% set(plot1(5),'DisplayName','\lambda = 2','Marker','<');
%% other attributes
xlabel('Iteration');
ylabel({'Top-1 Accuracy'},'HorizontalAlignment','center');
xlim(axes1,[0 2e4]);
ylim(axes1,[0 0.65]);%[0.35 0.65] without sgd, [0 0.65] with sgd
box(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontSize',20,'OuterPosition',[1 1 640 360],'XTick',...
    [0 4e3 8e3 12e3 16e3 20e3],'YTick',0:0.2:0.6,'Position',...
    [64 58 554.75, 291]);
% Create legend
legend1 = legend(axes1,'show');
set(legend1,'Location','southeast','FontSize',22);


function S = importexcel(row,col)
% Import the data
[~, ~, S] = xlsread('/Users/li/Desktop/li/USTC/Ling/ADMM-byzantine/admm_journal/ADMM实验/ADMM实验/实验原始数据/实验数据.xlsx','covertype数据集');
S = S(row,col);

S = string(S);
S(ismissing(S)) = '';
end

function vec = readstrdata(S,len)
vec = zeros(len,2);
for i = 1:len
    tmp = strsplit(S(i),' ');
    vec(i,:) = tmp(1:2);
end
end