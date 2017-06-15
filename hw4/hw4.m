dirs = dir('./CroppedYale/yaleB*');

training_data = zeros(1330, 192*168);
training_label = ones(1330, 1);

testing_data = zeros(1084, 192*168);
testing_label = ones(1084, 1);

p = 1;
q = 1;
for m = 1:numel(dirs)
    imageFiles = dir(strcat('./CroppedYale/', dirs(m).name, '/*.pgm'));
    
    for n = 1:numel(imageFiles)
        if 1 == contains(imageFiles(n).name, 'Ambient', 'IgnoreCase', true)
            continue;
        end
        
        path = strcat('./CroppedYale/', dirs(m).name, '/', imageFiles(n).name);
        origin = reshape(double(imresize(imread(path), [192, 168])), 1, []);
        
        if n <= 35
            training_data(p, :) = origin;
            
            if 1 == m
                training_label(p) = 0;
            end
            
            p = p + 1;
        else
            testing_data(q, :) = origin;
            
            if 1 == m
                testing_label(q) = 0;
            end
            
            q = q + 1;
        end
    end
end

[vec, ~, val, ~, ~] = pca(training_data);

training_set = sparse(training_data * vec(:, 1:30));
testing_set = sparse(testing_data * vec(:, 1:30));

%libsvmwrite('training.txt', training_label, training_set);
%libsvmwrite('testing.txt', testing_label, testing_set);

training_model = svmtrain(training_label, training_set, '-t 0 -d 5 -c 0.03125 -g 0.0078125');
predict = svmpredict(testing_label, testing_set, training_model);