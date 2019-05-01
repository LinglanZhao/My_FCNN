# A Matlab Implementation of Fully Connected Neural Network (Multilayer Perceptron) from Scratch
---
## My Fully Connected Neural Network
### Description:
输入训练数据X和对应标签Y以及其他网络参数，训练输出全连接神经网络各层神经元的权值矩阵W和偏置向量b（cell结构）
网络中的非线性激活函数采用ReLU或者tanh; 目标函数采用softmax的cross-entropy loss; 正则化采用(inverted)dropout
### Parameters:
* X:输入样本矩阵(D,m) 每一列为一个样本的特征向量    
* Y:输入样本标签矩阵(C,m) 每一列上仅在该样本对应的类别处标记为1，其余为0  
* learning\_rate: 学习率/步长  
* activation\_function: 非线性激活函数'ReLU'或'tanh'  
* num\_epoch: 遍历全部样本次数，每次遍历一遍所有样本都会对batch\_num个Mini-batch分别计算梯度下降并更新权值  
* num\_units: 向量，网络各层（隐层+输出层）单元数  
* keep\_prob: 向量，dropout regularization的参数，每一层的保留概率，每个元素对应这一层神经元输出的保留概率  
* batch\_size: mini-batch大小，一般设为2的幂次，若为inf则不进行mini-batch分组  
* update\_algorithm: 用于选择标准梯度(SGD)下降或者momentum/RMSprop/Adam优化算法:
   * 'SGD'
   * 'Momentum'
   * 'Nesterov_Momentum'
   * 'RMSprop'
   * 'Adam'
* print\_flag: 取值true/false， 是/否在迭代过程中输出交互熵和训练错误率  
* plot\_flag: 取值true/false， 是/否在迭代完成后绘制交互熵和训练错误率变化曲线  
* save\_flag: 取值true/false， 是/否在训练完成后保存神经网络各层神经元的权值矩阵W和偏置向量b（cell结构）  
### Usage:
```
[W, b] = My_FCNN (X, Y, learning_rate, activation_function, num_epoch, num_units, keep_prob, batch_size, update_algorithm, print_flag, plot_flag, save_flag)  
```
### Example:
![MLP](https://github.com/LinglanZhao/My_FCNN/blob/master/optim.png)
## Author: 
* Linglan Zhao
## Update: 
* Implemented before 2018.
* Updated in 2019.05.01. 