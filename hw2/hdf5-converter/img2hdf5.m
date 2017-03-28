% 取得所有圖片目錄
dirs = dir('./CroppedYale/yaleB*');

training_data = [];
training_label = [];

testing_data = [];
testing_label = [];

for m = 1:numel(dirs)
    % 取得該目錄的所有圖片資訊
    imageFiles = dir(strcat('./CroppedYale/', dirs(m).name, '/*.pgm'));
    
    for n = 1:numel(imageFiles)
        % 略過含有 Ambient 的圖片
        if 1 == contains(imageFiles(n).name, 'Ambient', 'IgnoreCase', true)
            continue;
        end
        
        % 圖片路徑
        path = strcat('./CroppedYale/', dirs(m).name, '/', imageFiles(n).name);
        
        % 讀入圖片後大小轉成 192*168
        img = imresize(imread(path), [192, 168]);

        if n <= 35
            training_data = cat(3, training_data, img);
            training_label = cat(2, training_label, m);
        else
            testing_data = cat(3, testing_data, img);
            testing_label = cat(2, testing_label, m);
        end
    end
end

delete 'data/training.hdf5';
delete 'data/testing.hdf5';

training_data = permute(reshape(double(training_data), [192, 168, 1, size(training_data, 3)]), [2, 1, 3, 4]);
testing_data = permute(reshape(double(testing_data), [192, 168, 1, size(testing_data, 3)]), [2, 1, 3, 4]);

h5create('data/training.hdf5', '/data', size(training_data), 'Datatype', 'double');
h5write('data/training.hdf5', '/data', training_data);

h5create('data/training.hdf5', '/label', size(training_label), 'Datatype', 'double');
h5write('data/training.hdf5', '/label', training_label);

h5create('data/testing.hdf5', '/data', size(testing_data), 'Datatype', 'double');
h5write('data/testing.hdf5', '/data', testing_data);

h5create('data/testing.hdf5', '/label', size(testing_label), 'Datatype', 'double');
h5write('data/testing.hdf5', '/label', testing_label);
