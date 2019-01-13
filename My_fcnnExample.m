%%% 使用My_FCNN.m函数的一个例子：
%%% 读取MNIST数据集（training set）：
num_training = 60000; 
disp('开始读取MNIST数据集...')
[X,Y] = My_readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte',num_training, 0);
disp('训练数据读取完成')
% MNIST_dataset =load('MNIST_dataset.mat');
% X0 = MNIST_dataset.X;
% Y0 = MNIST_dataset.Y;

%%%%%% 利用神经网络训练数据 生成各层神经元的权值W和偏置b
%%%%%% 神经网络参数设定：
%%% hyper paramters：
C = size(Y, 1);  % C为多分类数
num_epoch = 5; % 最大迭代次数（即遍历全部样本次数） 每次遍历一遍所有样本都会对所有Mini-batch分别计算梯度下降并更新权值
% 如果为gradient descent或momentum时num_epoch应更大
learning_rate = 0.005; % 学习率
activation_function = 'ReLU'; % 选择非线性激活函数
num_units = [120, 84, C]; % 网络各层（隐层+输出层）单元数，参考LeNet-5 全连接层的结构
%%% 正则化参数：
% dropout regularization：
keep_prob = [0.9, 0.9, 1]; % 保留概率向量，每个元素对应这一层神经元输出的保留概率
%%% 优化算法：
% Mini-batch梯度下降算法
% Mini-batch分组：
batch_size = 512;  % mini-batch大小为256 
% gradient descent或momentum/RMSprop/Adam优化算法
update_algorithm = 'Adam'; % 利用Adam优化算法进行权值更新
print_flag = true;
plot_flag = true;
save_flag = false; 

%%% 利用网络进行训练：
[W, b] = My_FCNN (X, Y, learning_rate, activation_function, num_epoch, num_units, keep_prob, batch_size, update_algorithm, print_flag, plot_flag, save_flag);

% %%%%%% 或者读入已经训练好的网络权值和偏置
% MyValues =load('MyNN_values.mat');
% W = MyValues.W;
% b = MyValues.b;

%%%%%% 在training set上检测训练误差
[Yp, P] = My_fcnnPredict(X, W, b, activation_function, Y); % 前向预测
STR=strcat('在训练集',num2str(num_training),'个训练样本中，识别准确率为： ',num2str(P)); 
disp(STR)

%%%%%% 在test set上进行验证
num_test = 10000;
[X_test,Y_test] = My_readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte',num_test, 0); % 读取测试集数据
[Yp_t, P_t] = My_fcnnPredict(X_test, W, b, activation_function, Y_test); 
STR_t=strcat('在测试集',num2str(num_test),'个测试样本中，识别准确率为： ',num2str(P_t));
disp(STR_t)