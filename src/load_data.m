clear;
close all;
%% 初始化变量。
filename = 'H:\直流电弧研究\GreeData\GreeData\Graph\2017-03-30Gree-Hongshi\20170330GeneratorGreeWarm_CompressorWH14.FCI.REP';
num_repeat = 42;
offset = 2000;
amplify = 5.3041;
%% 每个文本行的格式:
%   列1: 双精度值 (%f)
%	列2: 双精度值 (%f)
%   列3: 双精度值 (%f)
%	列4: 双精度值 (%f)
% 有关详细信息，请参阅 TEXTSCAN 文档。
formatSpec = '%f%f%f%f%[^\n\r]';
delimiter = '\t';
%% 打开文本文件，读取数据
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN,  'ReturnOnError', false);
fclose(fileID);
%% 输出电流值
data = table(dataArray{1:2}, 'VariableNames', {'current','integral'});
data = (data.current - offset) / amplify;
data = data(1:num_repeat:end);
clearvars filename delimiter formatSpec fileID dataArray ans amplify num_repeat offset;%清除临时变量