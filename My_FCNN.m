function [W, b] = My_FCNN (X, Y, learning_rate, activation_function, num_epoch, num_units, keep_prob, batch_size, update_algorithm, print_flag, plot_flag, save_flag)
    %%%%%% My Fully Connected Neural Network
    %%%%%% 函数功能描述：
    % 输入训练数据X和对应标签Y以及其他网络参数，训练输出全连接神经网络各层神经元的权值矩阵W和偏置向量b（cell结构）
    % 网络中的非线性激活函数采用ReLU或者tanh; 目标函数采用softmax的cross-entropy loss; 正则化采用(inverted)dropout
    %%%%%% 参数介绍：
    % X:输入样本矩阵(D,m) 每一列为一个样本的特征向量
    % Y:输入样本标签矩阵(C,m) 每一列上仅在该样本对应的类别处标记为1，其余为0
    % learning_rate: 学习率/步长
    % activation_function: 非线性激活函数'ReLU'或'tanh'
    % num_epoch: 遍历全部样本次数，每次遍历一遍所有样本都会对batch_num个Mini-batch分别计算梯度下降并更新权值
    % num_units: 向量，网络各层（隐层+输出层）单元数
    % keep_prob: 向量，dropout regularization的参数，每一层的保留概率，每个元素对应这一层神经元输出的保留概率
    % batch_size: mini-batch大小，一般设为2的幂次，若为inf则不进行mini-batch分组
    % update_algorithm: 用于选择标准梯度(SGD)下降或者momentum/RMSprop/Adam优化算法之一:'SGD','Momentum',Nesterov_Momentum,'RMSprop'和'Adam' 
    % print_flag: 取值true/false， 是/否在迭代过程中输出交互熵和训练错误率
    % plot_flag: 取值true/false， 是/否在迭代完成后绘制交互熵和训练错误率变化曲线
    % save_flag: 取值true/false， 是/否在训练完成后保存神经网络各层神经元的权值矩阵W和偏置向量b（cell结构）
    
    %%% 读取样本数和特征向量维数：
    [n0,m] = size(X);  % n0为特征向量维数，m为测试集大小即样本数
    [C,m1] = size(Y);  % C为多分类数
    L = length(num_units); % L为神经网络层数（隐层+输出层）
    if (m ~= m1)
        error('样本数量与标签个数不匹配');
    end
    if L ~= length(keep_prob)
        error('保留概率向量与网络层数不匹配');
    end
    
    %%% 优化算法：
    % Mini-batch分组
    [Xm,Ym,batch_num] = My_Minibatch (X,Y,batch_size);
    % momentum/RMSprop/Adam优化算法
    % 参数设置如下：
    kk = 0; % 迭代次数初始化
    B1 = 0.9; % 'momentum'
    B2 = 0.999; s = 10^(-8); % 'RMSprop'

    %%% 初始化向量集(He initialization)：
    % W,b{i}表示第i层的权值和偏置 i=1,2,...L
    % X,Y{j}表示第j个mini-batch分组的数据和标签 j=1,2...batch_num
    % Z,A{i}表示第i层的加权输入和激活函数输出
    % Xavier initialization uses a scaling factor for the weights  W[l]  of sqrt(1./layers_dims[l-1]) where He initialization use sqrt(2./layers_dims[l-1])
    % 由于matlab下标为正整数，第一层单独写开，其他层用for循环初始化
    % 第一层神经元权值矩阵，每行为一个神经元的所有权值；方差归一化避免梯度消失/爆炸的情况
    if activation_function == 'ReLU'
        W{1} = randn(num_units(1),n0)*sqrt(2/n0); % He initialization
    else
        W{1} = randn(num_units(1),n0)*sqrt(1/n0); % Xavier initialization
    end
    b{1} = zeros(num_units(1),1); % 第一层神经元的偏置
    % 其余层：
    for i = 2 : L
        % 初始化其余层神经元权值矩阵 
        if activation_function == 'ReLU'
            W{i} = randn(num_units(i),num_units(i-1))*sqrt(2/num_units(i-1)); % He initialization
        else
            W{i} = randn(num_units(i),num_units(i-1))*sqrt(1/num_units(i-1)); % Xavier initialization
        end
        b{i} = zeros(num_units(i),1); % 初始化其余层神经元的偏置
    end
    % Adam算法矩阵初始化：
    for i = 1 : L
        Vdw{i} = zeros(size(W{i})); %'momentum'
        Vdb{i} = zeros(size(b{i}));
        Sdw{i} = zeros(size(W{i})); %'RMSprop'
        Sdb{i} = zeros(size(b{i}));
    end   
    % 各mini-baich分组的样本数：
    for i = 1 : batch_num
        sample_num(i) = size(Xm{i},2);  % sample_num(j)为第j个mini-baich的样本数
    end
    cost_record = []; %记录每次迭代的cost function的值
    Trainerror_record = []; %记录每次迭代的训练误差

    for t = 1 : num_epoch %epoch: one pass through the training set
        % 每次遍历一遍全部训练样本
        for k = 1 : batch_num %iteration: computing on a single mini-batch
            % 每次对应一组mini-batch计算梯度下降
            %%% Forward Propagation:
            % 第一层：
            Z{1} = W{1}*Xm{k} + repmat(b{1},1,sample_num(k));
            A{1} = non_linearity(activation_function, Z{1});
            mask{1} = (rand(size(Z{1})) <= keep_prob(1)); % Inverted dropout
            A{1} = ((A{1}.*mask{1})/keep_prob(1)); % assuring that the result of the cost will still have the same expected value as without drop-out
            % 中间层：
            for i = 2 : L-1 
                Z{i} = W{i}*A{i-1} + repmat(b{i},1,sample_num(k));
                A{i} = non_linearity(activation_function, Z{i});
                mask{i} = (rand(size(Z{i})) <= keep_prob(i));
                A{i} = ((A{i}.*mask{i})/keep_prob(i)); 
            end
            % 输出层：
            Z{L} = W{L}*A{L-1} + repmat(b{L},1,sample_num(k));
            if C >= 2 %多分类问题
                A{L} = Soft_max(Z{L}); %输出层的激活输出 激活函数为：soft max
            else      %二分类问题
                A{L} = Sigmoid(Z{L}) ;  %输出层的激活输出 激活函数为：sigmoid function
            end
            
            %%% 该次迭代相关的数据记录：
            J = CostFuncion(A{L},Ym{k}); %本次前向传播的cost function: cross-entropy cost
            % 硬判决
            % Q = round(A{L}); 
            % err = length(find(sum(abs(Q-Ym{k}),1)~=0))/sample_num(k);
            % 软判决
            [val_Yp,index_Yp] = max(A{L});
            [val_Y,index_Y] = max(Ym{k});
            err = length(find(index_Yp ~= index_Y))/sample_num(k);
            Trainerror_record = [Trainerror_record, err]; %当前检测错误的概率
            cost_record = [cost_record, J]; %当前Cost function的值      

            %%% Backward Propagation:
            % 输出层：
            dZ{L} = A{L}-Ym{k}; % 'dX'表示J:cost function（或者对应的L:Loss function）对变量（矩阵）'X'求偏导
            dW{L} = (dZ{L}*A{L-1}')/sample_num(k);
            db{L} = sum(dZ{L},2)/sample_num(k);
            dA{L-1} = W{L}'*dZ{L};
            dA{L-1} = ((dA{L-1}.*mask{L-1})/keep_prob(L-1)); 
            % 中间层：
            for i = L-1 : -1 : 2
                dZ{i} = dA{i}.*d_non_linearity(activation_function, Z{i});
                dW{i} = (dZ{i}*A{i-1}')/sample_num(k);
                db{i} = sum(dZ{i},2)/sample_num(k);
                dA{i-1} = W{i}'*dZ{i};
                dA{i-1} = ((dA{i-1}.*mask{i-1})/keep_prob(i-1)); 
            end
            % 第一层：
            dZ{1} = dA{1}.*d_non_linearity(activation_function, Z{1});
            dW{1} = (dZ{1}*Xm{k}')/sample_num(k);
            db{1} = sum(dZ{1},2)/sample_num(k);

            kk = kk + 1;
            %%% 权值更新：
            switch update_algorithm
                case {'SGD'}
                    for i = 1 : L
                        % 利用标准梯度下降进行权值更新
                        W{i} = W{i} - learning_rate*dW{i};
                        b{i} = b{i} - learning_rate*db{i};
                    end
                case {'Momentum'}
                    for i = 1 : L
                        % 利用'momentum'算法进行权值更新:
                        Vdw{i} = (B1*Vdw{i}+(1-B1)*dW{i});
                        Vdb{i} = (B1*Vdb{i}+(1-B1)*db{i});
                        W{i} = W{i} - learning_rate*Vdw{i};
                        b{i} = b{i} - learning_rate*Vdb{i};
                    end
                case {'Nesterov_Momentum'}
                    for i = 1 : L
                        % 利用'Nesterov_Momentum'算法进行权值更新:
                        Vdw_old{i} = Vdw{i};
                        Vdb_old{i} = Vdb{i};
                        Vdw{i} = B1*Vdw{i} - learning_rate*dW{i};
                        Vdb{i} = B1*Vdb{i} - learning_rate*db{i};
                        W{i} = W{i} + (-B1*Vdw_old{i} + (1+B1)*Vdw{i});
                        b{i} = b{i} + (-B1*Vdb_old{i} + (1+B1)*Vdb{i});
                    end
                case {'RMSprop'}
                    for i = 1 : L
                        % 利用'RMSprop'算法进行权值更新:
                        Sdw{i} = (B2*Sdw{i}+(1-B2)*dW{i}.^2);
                        Sdb{i} = (B2*Sdb{i}+(1-B2)*db{i}.^2);
                        W{i} = W{i} - (learning_rate*dW{i}./(sqrt(Sdw{i})+s));
                        b{i} = b{i} - (learning_rate*db{i}./(sqrt(Sdb{i})+s));
                    end
                case {'Adam'}
                    for i = 1 : L
                        % 利用Adam算法进行权值更新
                        Vdw{i} = (B1*Vdw{i}+(1-B1)*dW{i});
                        Vdb{i} = (B1*Vdb{i}+(1-B1)*db{i});
                        Sdw{i} = (B2*Sdw{i}+(1-B2)*dW{i}.^2);
                        Sdb{i} = (B2*Sdb{i}+(1-B2)*db{i}.^2);
                        Vdw_c{i} = Vdw{i}/(1-B1^kk);
                        Vdb_c{i} = Vdb{i}/(1-B1^kk);
                        Sdw_c{i} = Sdw{i}/(1-B2^kk);
                        Sdb_c{i} = Sdb{i}/(1-B2^kk);
                        W{i} = W{i} - (learning_rate*Vdw_c{i}./(sqrt(Sdw_c{i})+s));
                        b{i} = b{i} - (learning_rate*Vdb_c{i}./(sqrt(Sdb_c{i})+s));
                    end
                otherwise
                    error('Undefined Operation');
            end 
            if (mod(t,10) == 1)&(k == 1)&print_flag
                disp(strcat('epoch=',num2str(t),',batch_size=', num2str(batch_size), ': Cost=', num2str(J), ';Training error=', num2str(err)))
            end
        end
    end
    disp('迭代完成')
    
    %%%%%% 绘图
    % Cost_Function/Training_Error versus. #iteration
    if plot_flag
        set(0,'defaultfigurecolor','w'); % 图片设置为白底
        figure(1)
        plot(cost_record);
        xlabel('#iteration(batch size = 512)');ylabel('Training Lost'); title('MNIST Multi-Layer Perceptron with dropout');grid on;
        figure(2)
        plot(1-Trainerror_record);
        xlabel('#iteration(batch size = 512)');ylabel('Training accuracy');title('MNIST Multi-Layer Perceptron with dropout');grid on;
    end
    
    %%%%%% 保存数据
    % 神经网络各层神经元的权值矩阵W和偏置向量b（cell结构）
    if save_flag
        save('Myfcnn_show.mat','W','b');
        disp('权值矩阵W和偏置向量b已成功保存')
    end
%     save('Loss_SGD.mat','cost_record');
%     save('Loss_Momentum.mat','cost_record');
%     save('Loss_RMSprop.mat','cost_record');
%     save('Loss_Adam.mat','cost_record');
end

%%%%%%%%%%  子函数模块  %%%%%%%%%%
%%% 非线性激活函数 forward 
function Y = non_linearity(activation_function, X)
    switch activation_function
        case {'ReLU'}
             X(X < 0) = 0;
             Y = X;
        case {'tanh'}
             Y = tanh(X);
        otherwise
             error('Undefined non_linearity');
    end
end

%%% 非线性激活函数的导数 backward 
function Y = d_non_linearity(activation_function, X)
    switch activation_function
        case {'ReLU'}
             Y = (X>=0);
        case {'tanh'}
             Y = (1-tanh(X).^2);
        otherwise
             error('Undefined non_linearity');
    end
end

%%% Sigmoid 
function y=Sigmoid(x)
y=1./(1 + exp(-x));
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

%%%%%% 生成Mini-batch分组
function [Xm,Ym,batch_num] = My_Minibatch (X,Y,batch_size)
%%% Mini-batch分组 
% X为输入数据集；Y为数据集对应的标签；size为mini-batch大小
% Xm，Ym为分组输出 cell结构；batch_num为分组个数
% size一般取64.128.256,512等
    [n0,m] = size(X);  % n0为特征向量维数，m为测试集大小即样本数
    [C,m1] = size(Y); % C为多分类数
    if m1 ~= m
        error('样本数量与标签个数不匹配');
    end
    if batch_size >= m
        Xm{1} = X;
        Ym{1} = Y;
        batch_num = 1;
    else
        num = floor(m/batch_size);
        res = mod(m,batch_size);
        % 确定分组个数
        if res == 0
            batch_num = num;
        else
            batch_num = num + 1;
        end
        for i = 1 : num
            Xm{i} = X(:,((i-1)*batch_size+1) : i*batch_size);
            Ym{i} = Y(:,((i-1)*batch_size+1) : i*batch_size);
        end
        if res ~= 0
            Xm{batch_num} = X(:,(num*batch_size+1) : m);
            Ym{batch_num} = Y(:,(num*batch_size+1) : m);
        end
    end
end