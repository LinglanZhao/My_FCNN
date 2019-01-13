%%% ʹ��My_FCNN.m������һ�����ӣ�
%%% ��ȡMNIST���ݼ���training set����
num_training = 60000; 
disp('��ʼ��ȡMNIST���ݼ�...')
[X,Y] = My_readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte',num_training, 0);
disp('ѵ�����ݶ�ȡ���')
% MNIST_dataset =load('MNIST_dataset.mat');
% X0 = MNIST_dataset.X;
% Y0 = MNIST_dataset.Y;

%%%%%% ����������ѵ������ ���ɸ�����Ԫ��ȨֵW��ƫ��b
%%%%%% ����������趨��
%%% hyper paramters��
C = size(Y, 1);  % CΪ�������
num_epoch = 5; % ������������������ȫ������������ ÿ�α���һ�������������������Mini-batch�ֱ�����ݶ��½�������Ȩֵ
% ���Ϊgradient descent��momentumʱnum_epochӦ����
learning_rate = 0.005; % ѧϰ��
activation_function = 'ReLU'; % ѡ������Լ����
num_units = [120, 84, C]; % ������㣨����+����㣩��Ԫ�����ο�LeNet-5 ȫ���Ӳ�Ľṹ
%%% ���򻯲�����
% dropout regularization��
keep_prob = [0.9, 0.9, 1]; % ��������������ÿ��Ԫ�ض�Ӧ��һ����Ԫ����ı�������
%%% �Ż��㷨��
% Mini-batch�ݶ��½��㷨
% Mini-batch���飺
batch_size = 512;  % mini-batch��СΪ256 
% gradient descent��momentum/RMSprop/Adam�Ż��㷨
update_algorithm = 'Adam'; % ����Adam�Ż��㷨����Ȩֵ����
print_flag = true;
plot_flag = true;
save_flag = false; 

%%% �����������ѵ����
[W, b] = My_FCNN (X, Y, learning_rate, activation_function, num_epoch, num_units, keep_prob, batch_size, update_algorithm, print_flag, plot_flag, save_flag);

% %%%%%% ���߶����Ѿ�ѵ���õ�����Ȩֵ��ƫ��
% MyValues =load('MyNN_values.mat');
% W = MyValues.W;
% b = MyValues.b;

%%%%%% ��training set�ϼ��ѵ�����
[Yp, P] = My_fcnnPredict(X, W, b, activation_function, Y); % ǰ��Ԥ��
STR=strcat('��ѵ����',num2str(num_training),'��ѵ�������У�ʶ��׼ȷ��Ϊ�� ',num2str(P)); 
disp(STR)

%%%%%% ��test set�Ͻ�����֤
num_test = 10000;
[X_test,Y_test] = My_readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte',num_test, 0); % ��ȡ���Լ�����
[Yp_t, P_t] = My_fcnnPredict(X_test, W, b, activation_function, Y_test); 
STR_t=strcat('�ڲ��Լ�',num2str(num_test),'�����������У�ʶ��׼ȷ��Ϊ�� ',num2str(P_t));
disp(STR_t)