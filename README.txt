# A Matlab Implementation of Fully Connected Neural Network (MLP) from Scratch
## My Fully Connected Neural Network
### Description:
����ѵ������X�Ͷ�Ӧ��ǩY�Լ��������������ѵ�����ȫ���������������Ԫ��Ȩֵ����W��ƫ������b��cell�ṹ��
�����еķ����Լ��������ReLU����tanh; Ŀ�꺯������softmax��cross-entropy loss; ���򻯲���(inverted)dropout
### Parameters:
X:������������(D,m) ÿһ��Ϊһ����������������
Y:����������ǩ����(C,m) ÿһ���Ͻ��ڸ�������Ӧ����𴦱��Ϊ1������Ϊ0
learning_rate: ѧϰ��/����
activation_function: �����Լ����'ReLU'��'tanh'
num_epoch: ����ȫ������������ÿ�α���һ���������������batch_num��Mini-batch�ֱ�����ݶ��½�������Ȩֵ
num_units: ������������㣨����+����㣩��Ԫ��
keep_prob: ������dropout regularization�Ĳ�����ÿһ��ı������ʣ�ÿ��Ԫ�ض�Ӧ��һ����Ԫ����ı�������
batch_size: mini-batch��С��һ����Ϊ2���ݴΣ���Ϊinf�򲻽���mini-batch����
update_algorithm: ����ѡ���׼�ݶ�(SGD)�½�����momentum/RMSprop/Adam�Ż��㷨֮һ:'SGD','Momentum',Nesterov_Momentum,'RMSprop'��'Adam' 
print_flag: ȡֵtrue/false�� ��/���ڵ�����������������غ�ѵ��������
plot_flag: ȡֵtrue/false�� ��/���ڵ�����ɺ���ƽ����غ�ѵ�������ʱ仯����
save_flag: ȡֵtrue/false�� ��/����ѵ����ɺ󱣴������������Ԫ��Ȩֵ����W��ƫ������b��cell�ṹ��
### Usage:
```
[W, b] = My_FCNN (X, Y, learning_rate, activation_function, num_epoch, num_units, keep_prob, batch_size, update_algorithm, print_flag, plot_flag, save_flag)  
```
## Author: Linglan Zhao
## Date: 2019.01.13