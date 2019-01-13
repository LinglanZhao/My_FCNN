function [Yp, P] = My_fcnnPredict(X, W, b, activation_function, Y)
%%% ʵ��ȫ�����������ǰ�򴫲�Ԥ���Լ�����
% ���룺
% XΪ����������
% YΪ��������ݵ���ʵ��ǩ
% W,bΪ������������Ȩֵ��ƫ�ã�W,bΪ1*L��cell�ṹ��LΪ���������
% activation_function�����Ϊȡֵ'ReLU'����'tanh'
% �����
% YpΪԤ�����
% PΪʶ��׼ȷ�ʡ�������ú���������Y�򷵻�ʶ��׼ȷ��:[0,1]��Χ��С���������Y���룬��ú���ֻʵ��ǰ��Ԥ�ⲻ���飬P����-1
    if activation_function ~= 'ReLU' & activation_function ~= 'tanh'
       error('Undefined Operation');
    end
    flag = 0;
    if nargin <= 3
        Y = [];
        flag = 1; % PӦ�÷���-1
    end
    L = size(W,2); % LΪ�������
    m = size(X,2); % mΪ������

    % ǰ��Ԥ��
    % ��һ�㣺
    Z{1} = W{1}*X + repmat(b{1},1,m);
    if activation_function == 'tanh'
        A{1} = tanh(Z{1});
    else
        A{1} = ReLU(Z{1});
    end
    % �м�㣺
    for i = 2 : L-1
        Z{i} = W{i}*A{i-1} + repmat(b{i},1,m);
        if activation_function == 'tanh'
            A{i} = tanh(Z{i});
        else
            A{i} = ReLU(Z{i});
        end
    end
    % ����㣺
    Z{L} = W{L}*A{L-1} + repmat(b{L},1,m);
    C = size(Z{L},1); % CΪ������
    if C >= 2 %���������
        A{L} = Soft_max(Z{L}); %�����ļ������ �����Ϊ��soft max
    else      %����������
        A{L} = Sigmoid(Z{L});  %�����ļ������ �����Ϊ��sigmoid function
    end
    Yp = A{L};

    % ����ʶ��׼ȷ��
    if flag == 0 
%         %Ӳ�о�
%         Q = A{L};
%         Q(Q >= 0.5) = 1;
%         Q(Q < 0.5) = 0;
%         P = 1 - length(find(sum(abs(Q-Y),1)~=0))/m;
        % ���о�
        [val_Yp,index_Yp] = max(A{L});
        [val_Y,index_Y] = max(Y);
         P = length(find(index_Yp == index_Y))/m;
    else
        P = -1;
    end
end

%%%%%%%%%%  �Ӻ���ģ��  %%%%%%%%%%
%%% ReLU
function Y = ReLU(X)
    X(X < 0) = 0;
    Y = X;
end

%%% ��λ��Ծ������ReLU�ĵ�������
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
    %%% ��ֹZֵ����exp��A����inf�����
    if length(find(isnan(A)==1)) ~= 0
        bug_position = isnan(A);
        A(bug_position) = 1;
    end
end

%%% Cost Function
function J = CostFuncion(A,Y) 
[no,m] = size(A);
s = 10^-8;
    if no == 1 %����������
        L = -(Y.*log(A+s)+(1-Y).*log(1-A+s)); %Loss function 
        J = sum(L,2)/m; %Cost function
    else     %���������
        L = -sum(Y.*log(A+s));
        J = sum(L,2)/m;
    end
end
