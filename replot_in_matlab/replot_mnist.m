len = 11;
%% loading for 5 methods
% blue = readstrdata(importexcel(3:3+len-1,1), len);
% red = readstrdata(importexcel(110:110+len-1,1), len);
% yellow = readstrdata(importexcel(57:57+len-1,1), len);
% purple = readstrdata(importexcel(216:216+len-1,1), len);
% green = readstrdata(importexcel(163:163+len-1,1), len);

%% loading for 5 lambdas
blue = readstrdata(importexcel(3:3+len-1,19), len);
red = readstrdata(importexcel(57:57+len-1,19), len);
yellow = readstrdata(importexcel(110:110+len-1,19), len);
purple = readstrdata(importexcel(156:156+len-1,19), len);
green = readstrdata(importexcel(209:209+len-1,19), len);

%% create figures
figure('InvertHardcopy','off','PaperUnits','centimeters','Color',[1 1 1],...
    'OuterPosition',[1 1 640 360]);

% Create axes
axes1 = axes('Units','pixels');
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot1 = plot(blue(:,1),[blue(:,2),red(:,2),yellow(:,2),purple(:,2),green(:,2)],...
    'MarkerIndices',[1 2 3 4 5 6 7 8 9 10 11],...
    'MarkerSize',10,...
    'LineWidth',2);
%% labels for 5 methods
% set(plot1(1),'DisplayName','Ideal SGD','Marker','diamond');
% set(plot1(2),'DisplayName','RSA','Marker','v');
% set(plot1(3),'DisplayName','Stochastic ADMM','Marker','^');
% set(plot1(4),'DisplayName','Geometric median','Marker','>');
% set(plot1(5),'DisplayName','Median','Marker','<');
%% labels for 5 lambdas
set(plot1(1),'DisplayName','\lambda = 0.005','Marker','diamond');
set(plot1(2),'DisplayName','\lambda = 0.1','Marker','v');
set(plot1(3),'DisplayName','\lambda = 0.5','Marker','^');
set(plot1(4),'DisplayName','\lambda = 1','Marker','>');
set(plot1(5),'DisplayName','\lambda = 2','Marker','<');
%% other attributes
xlabel('Iteration');
ylabel({'Top-1 Accuracy'},'HorizontalAlignment','center');
xlim(axes1,[0 1000]);
ylim(axes1,[0 1]);
box(axes1,'on');
% Set the remaining axes properties
set(axes1,'FontSize',20,'OuterPosition',[1 1 640 320],'XTick',...
    [0 200 400 600 800 1000],'Position',...
    [64 58 554.75, 291]);
% Create legend
legend1 = legend(axes1,'show');
set(legend1,'Location','southeast','FontSize',22);


function S = importexcel(row,col)
% Import the data
[~, ~, S] = xlsread('/Users/li/Desktop/li/USTC/Ling/ADMM-byzantine/admm_journal/ADMM实验/ADMM实验/实验原始数据/实验数据.xlsx','mnist数据集');
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