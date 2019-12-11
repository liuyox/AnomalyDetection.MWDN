clear;
close all;
%%
dir_list = ["../data/negative", "../data/positive"];
for d = 1:2
    data_path = char(dir_list(d));
    list = dir([data_path ,'/*.txt']);
    num_files = length(list);
    for i=1:num_files
        current = [];
        filename = fullfile(list(i).folder,list(i).name);
        fileID = fopen(filename,'r');
        if fileID == -1
            continue
        end
        try
            while feof(fileID) ~= 1
                str = fgetl(fileID);
                str = deblank(str);
                data = regexp(str,'\t','split');
                if strcmp(data(1),'CurrentSamplingRate')
                    simple_rate = str2num(cell2mat(data(2)));
                elseif strcmp(data(1),'CurrentZero')
                    offset = str2num(cell2mat(data(2)));
                elseif strcmp(data(1),'CurrentAmplify')
                    amplify = str2num(cell2mat(data(2)));
                elseif strcmp(data(1),'C')
                    current = [current,(str2num(cell2mat(data(2))) - offset) / amplify];
                end
            end
        catch
            fclose(fileID);
            fprintf('%s error\n',filename);
            continue
        end
        fclose(fileID);

        current = current(50:end);
        window_size = 48;
        num_examples = floor(length(current)/window_size);
        data = zeros(num_examples,window_size+1);
        for j=1:num_examples
            data(j,1:window_size) = current((j-1)*window_size+1:j*window_size);
        end
        data(:,end) = d - 1;
        filename = fullfile('../data', 'data.txt');
        dlmwrite(filename,data,'delimiter','\t','-append');
    end
end