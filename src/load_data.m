clear;
close all;
%% ��ʼ��������
filename = 'H:\ֱ���绡�о�\GreeData\GreeData\Graph\2017-03-30Gree-Hongshi\20170330GeneratorGreeWarm_CompressorWH14.FCI.REP';
num_repeat = 42;
offset = 2000;
amplify = 5.3041;
%% ÿ���ı��еĸ�ʽ:
%   ��1: ˫����ֵ (%f)
%	��2: ˫����ֵ (%f)
%   ��3: ˫����ֵ (%f)
%	��4: ˫����ֵ (%f)
% �й���ϸ��Ϣ������� TEXTSCAN �ĵ���
formatSpec = '%f%f%f%f%[^\n\r]';
delimiter = '\t';
%% ���ı��ļ�����ȡ����
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN,  'ReturnOnError', false);
fclose(fileID);
%% �������ֵ
data = table(dataArray{1:2}, 'VariableNames', {'current','integral'});
data = (data.current - offset) / amplify;
data = data(1:num_repeat:end);
clearvars filename delimiter formatSpec fileID dataArray ans amplify num_repeat offset;%�����ʱ����