function [Yp, P] = My_fcnnPredict(X, W, b, activation_function, Y)
%%% 实现全连接神经网络的前向传播预测以及检验
% 输入：
% X为待检测的数据
% Y为待检测数据的真实标签
% W,b为的神经网络各层的权值和偏置（W,b为1*L的cell结构，L为网络层数）
% activation_function激活函数为取值'ReLU'或者'tanh'
% 输出：
% Yp为预测输出
% P为识别准确率——如果该函数有输入Y则返回识别准确率:[0,1]范围的小数；如果无Y输入，则该函数只实现前向预测不检验，P返回-1
    if activation_function ~= 'ReLU' & activation_function ~= 'tanh'
       error('Undefined Operation');
    end
    flag = 0;
    if nargin <= 3
        Y = [];
        flag = 1; % P应该返回-1
    end
    L = size(W,2); % L为网络层数
    m = size(X,2); % m为样本数

    % 前向预测
    % 第一层：
    Z{1} = W{1}*X + repmat(b{1},1,m);
    if activation_function == 'tanh'
        A{1} = tanh(Z{1});
    else
        A{1} = ReLU(Z{1});
    end
    % 中间层：
    for i = 2 : L-1
        Z{i} = W{i}*A{i-1} + repmat(b{i},1,m);
        if activation_function == 'tanh'
            A{i} = tanh(Z{i});
        else
            A{i} = ReLU(Z{i});
        end
    end
    % 输出层：
    Z{L} = W{L}*A{L-1} + repmat(b{L},1,m);
    C = size(Z{L},1); % C为分类数
    if C >= 2 %多分类问题
        A{L} = Soft_max(Z{L}); %输出层的激活输出 激活函数为：soft max
    else      %二分类问题
        A{L} = Sigmoid(Z{L});  %输出层的激活输出 激活函数为：sigmoid function
    end
    Yp = A{L};

    % 计算识别准确率
    if flag == 0 
%         %硬判决
%         Q = A{L};
%         Q(Q >= 0.5) = 1;
%         Q(Q < 0.5) = 0;
%         P = 1 - length(find(sum(abs(Q-Y),1)~=0))/m;
        % 软判决
        [val_Yp,index_Yp] = max(A{L});
        [val_Y,index_Y] = max(Y);
         P = length(find(index_Yp == index_Y))/m;
    else
        P = -1;
    end
end

%%%%%%%%%%  子函数模块  %%%%%%%%%%
%%% ReLU
function Y = ReLU(X)
    X(X < 0) = 0;
    Y = X;
end

%%% 单位阶跃函数（ReLU的导函数）
function Y = Unitstep(X)
    X(X >= 0) = 1;
    X(X < 0)  = 0;
    Y = X;
end

%%% Sigmoid 
function y=Sigmoid(x)
y=1./(1+exp(-x));
end

%%% Soft max
function A = Soft_max(Z)
    [m,n] = size(Z);
    A = zeros(m,n);
    temp = exp(Z);
    A = temp./repmat( sum(temp),m,1);
    %%% 防止Z值过大exp后A出现inf的情况
    if length(find(isnan(A)==1)) ~= 0
        bug_position = isnan(A);
        A(bug_position) = 1;
    end
end

%%% Cost Function
function J = CostFuncion(A,Y) 
[no,m] = size(A);
s = 10^-8;
    if no == 1 %二分类的情况
        L = -(Y.*log(A+s)+(1-Y).*log(1-A+s)); %Loss function 
        J = sum(L,2)/m; %Cost function
    else     %多分类的情况
        L = -sum(Y.*log(A+s));
        J = sum(L,2)/m;
    end
end
