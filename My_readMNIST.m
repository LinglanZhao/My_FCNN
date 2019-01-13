% readMNIST by Siddharth Hegde
%
% Description:
% Read digits and labels from raw MNIST data files
% File format as specified on http://yann.lecun.com/exdb/mnist/
% Note: The 4 pixel padding around the digits will be remove
%       Pixel values will be normalised to the [0...1] range
%
% Usage:
% [imgs labels] = readMNIST(imgFile, labelFile, readDigits, offset)
%
% Parameters:
% imgFile = name of the image file
% labelFile = name of the label file
% readDigits = number of digits to be read
% offset = skips the first offset number of digits before reading starts
%
% Returns:
% imgs = 20 x 20 x readDigits sized matrix of digits
% labels = readDigits x 1 matrix containing labels for each digit
%
function [X Y] = My_readMNIST(imgFile, labelFile, readDigits, offset)
    % [imgs labels] = readMNIST(imgFile, labelFile, readDigits, offset)
    % 定义多分类标签：
    label_0 = [1;0;0;0;0;0;0;0;0;0];
    label_1 = [0;1;0;0;0;0;0;0;0;0];
    label_2 = [0;0;1;0;0;0;0;0;0;0];
    label_3 = [0;0;0;1;0;0;0;0;0;0];
    label_4 = [0;0;0;0;1;0;0;0;0;0];
    label_5 = [0;0;0;0;0;1;0;0;0;0];
    label_6 = [0;0;0;0;0;0;1;0;0;0];
    label_7 = [0;0;0;0;0;0;0;1;0;0];
    label_8 = [0;0;0;0;0;0;0;0;1;0];
    label_9 = [0;0;0;0;0;0;0;0;0;1];
    % Read digits
    fid = fopen(imgFile, 'r', 'b');
    header = fread(fid, 1, 'int32');
    if header ~= 2051
        error('Invalid image file header');
    end
    count = fread(fid, 1, 'int32');
    if count < readDigits+offset
        error('Trying to read too many digits');
    end
    
    h = fread(fid, 1, 'int32');
    w = fread(fid, 1, 'int32');
    
    if offset > 0
        fseek(fid, w*h*offset, 'cof');
    end
    
    imgs = zeros([h w readDigits]);
    
    for i=1:readDigits
        for y=1:h
            imgs(y,:,i) = fread(fid, w, 'uint8');
        end
    end
    
    fclose(fid);

    % Read digit labels
    fid = fopen(labelFile, 'r', 'b');
    header = fread(fid, 1, 'int32');
    if header ~= 2049
        error('Invalid label file header');
    end
    count = fread(fid, 1, 'int32');
    if count < readDigits+offset
        error('Trying to read too many digits');
    end
    
    if offset > 0
        fseek(fid, offset, 'cof');
    end
    
    labels = fread(fid, readDigits, 'uint8');
    fclose(fid);
    
    % Calc avg digit and count
    imgs = trimDigits(imgs, 4);
    imgs = normalizePixValue(imgs);
    %[avg num stddev] = getDigitStats(imgs, labels);
    
    % 修改部分
    %[imgs labels] = readMNIST(imgFile, labelFile, readDigits, offset)
    X = [];
    Y = [];
    for i = 1 :  readDigits
        x1 = imgs(:,:,i);
        X  = [X,x1(:)]; % reshape(imgs(:,:,i), [400, 1]); 按列读取
        y1 = labels(i);
        % One Hot encodings:
        switch y1
                case {0}
                 Y=[Y,label_0];
                case {1}
                 Y=[Y,label_1];
                case {2}
                 Y=[Y,label_2];
                case {3}
                 Y=[Y,label_3];
                case {4}
                 Y=[Y,label_4];
                case {5}
                 Y=[Y,label_5];
                case {6}
                 Y=[Y,label_6];
                case {7}
                 Y=[Y,label_7];
                case {8}
                 Y=[Y,label_8];
                case {9}
                 Y=[Y,label_9];
            otherwise
                 Y=[Y,label_0];
        end
    end
    
end

function digits = trimDigits(digitsIn, border)
    dSize = size(digitsIn);
    digits = zeros([dSize(1)-(border*2) dSize(2)-(border*2) dSize(3)]);
    for i=1:dSize(3)
        digits(:,:,i) = digitsIn(border+1:dSize(1)-border, border+1:dSize(2)-border, i);
    end
end

function digits = normalizePixValue(digits)
    digits = double(digits);
    for i=1:size(digits, 3)
        digits(:,:,i) = digits(:,:,i)./255.0;
    end
end
