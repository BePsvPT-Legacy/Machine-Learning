tStart = tic;

% 取得所有圖片目錄
dirs = dir('./CroppedYale/yaleB*');

images = cell(length(dirs));

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
        
        % 讀入圖片後大小轉成 192*168 並轉成 signed integer
        images{m, n} = int64(imresize(imread(path), [192, 168]));
    end
end

% 取得 rows 和 cols 數目
[rows, cols] = size(images);

% 初始化正確數目
sadCorrect = 0;
ssdCorrect = 0;

% 總共圖片數目
count = 0;

for m = 1:rows
    disp(strcat('current people: ', int2str(m)));
    
    for n = 36:cols
        % 略過空值
        if 1 == isempty(images{m, n})
            continue
        end
        
        sadMin = intmax('int64');
        ssdMin = intmax('int64');
        
        % 找出 m row 最小值
        for q = 1:35
            temp = images{m, n} - images{m, q};

            sadDiff = sum(sum(abs(temp)));
            ssdDiff = sum(sum(temp.^2));

            if sadDiff < sadMin
                sadMin = sadDiff;
            end

            if ssdDiff < ssdMin
                ssdMin = ssdDiff;
            end
        end
        
        sad = 1;
        ssd = 1;

        for p = 1:rows
            if p == m
                continue
            elseif 0 == (sad | ssd)
                break
            end
            
            for q = 1:35
                if 0 == (sad | ssd)
                    break
                end

                temp = images{m, n} - images{p, q};

                % 如果有比 m row 還小的值出現，則可直接停止尋找
                if 1 == sad && sum(sum(abs(temp))) < sadMin
                    sad = 0;
                end

                if 1 == ssd && sum(sum(temp.^2)) < ssdMin
                    ssd = 0;
                end
            end
        end
        
        sadCorrect = sadCorrect + sad;
        ssdCorrect = ssdCorrect + ssd;
        
        count = count + 1;
    end
end

disp(strcat('sad: ', num2str(sadCorrect / count * 100), '%'));
disp(strcat('ssd: ', num2str(ssdCorrect / count * 100), '%'));

toc(tStart);
