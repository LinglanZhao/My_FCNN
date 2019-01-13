function [W, b] = My_FCNN (X, Y, learning_rate, activation_function, num_epoch, num_units, keep_prob, batch_size, update_algorithm, print_flag, plot_flag, save_flag)
    %%%%%% My Fully Connected Neural Network
    %%%%%% ��������������
    % ����ѵ������X�Ͷ�Ӧ��ǩY�Լ��������������ѵ�����ȫ���������������Ԫ��Ȩֵ����W��ƫ������b��cell�ṹ��
    % �����еķ����Լ��������ReLU����tanh; Ŀ�꺯������softmax��cross-entropy loss; ���򻯲���(inverted)dropout
    %%%%%% �������ܣ�
    % X:������������(D,m) ÿһ��Ϊһ����������������
    % Y:����������ǩ����(C,m) ÿһ���Ͻ��ڸ�������Ӧ����𴦱��Ϊ1������Ϊ0
    % learning_rate: ѧϰ��/����
    % activation_function: �����Լ����'ReLU'��'tanh'
    % num_epoch: ����ȫ������������ÿ�α���һ���������������batch_num��Mini-batch�ֱ�����ݶ��½�������Ȩֵ
    % num_units: ������������㣨����+����㣩��Ԫ��
    % keep_prob: ������dropout regularization�Ĳ�����ÿһ��ı������ʣ�ÿ��Ԫ�ض�Ӧ��һ����Ԫ����ı�������
    % batch_size: mini-batch��С��һ����Ϊ2���ݴΣ���Ϊinf�򲻽���mini-batch����
    % update_algorithm: ����ѡ���׼�ݶ�(SGD)�½�����momentum/RMSprop/Adam�Ż��㷨֮һ:'SGD','Momentum',Nesterov_Momentum,'RMSprop'��'Adam' 
    % print_flag: ȡֵtrue/false�� ��/���ڵ�����������������غ�ѵ��������
    % plot_flag: ȡֵtrue/false�� ��/���ڵ�����ɺ���ƽ����غ�ѵ�������ʱ仯����
    % save_flag: ȡֵtrue/false�� ��/����ѵ����ɺ󱣴������������Ԫ��Ȩֵ����W��ƫ������b��cell�ṹ��
    
    %%% ��ȡ����������������ά����
    [n0,m] = size(X);  % n0Ϊ��������ά����mΪ���Լ���С��������
    [C,m1] = size(Y);  % CΪ�������
    L = length(num_units); % LΪ���������������+����㣩
    if (m ~= m1)
        error('�����������ǩ������ƥ��');
    end
    if L ~= length(keep_prob)
        error('�����������������������ƥ��');
    end
    
    %%% �Ż��㷨��
    % Mini-batch����
    [Xm,Ym,batch_num] = My_Minibatch (X,Y,batch_size);
    % momentum/RMSprop/Adam�Ż��㷨
    % �����������£�
    kk = 0; % ����������ʼ��
    B1 = 0.9; % 'momentum'
    B2 = 0.999; s = 10^(-8); % 'RMSprop'

    %%% ��ʼ��������(He initialization)��
    % W,b{i}��ʾ��i���Ȩֵ��ƫ�� i=1,2,...L
    % X,Y{j}��ʾ��j��mini-batch��������ݺͱ�ǩ j=1,2...batch_num
    % Z,A{i}��ʾ��i��ļ�Ȩ����ͼ�������
    % Xavier initialization uses a scaling factor for the weights  W[l]  of sqrt(1./layers_dims[l-1]) where He initialization use sqrt(2./layers_dims[l-1])
    % ����matlab�±�Ϊ����������һ�㵥��д������������forѭ����ʼ��
    % ��һ����ԪȨֵ����ÿ��Ϊһ����Ԫ������Ȩֵ�������һ�������ݶ���ʧ/��ը�����
    if activation_function == 'ReLU'
        W{1} = randn(num_units(1),n0)*sqrt(2/n0); % He initialization
    else
        W{1} = randn(num_units(1),n0)*sqrt(1/n0); % Xavier initialization
    end
    b{1} = zeros(num_units(1),1); % ��һ����Ԫ��ƫ��
    % ����㣺
    for i = 2 : L
        % ��ʼ���������ԪȨֵ���� 
        if activation_function == 'ReLU'
            W{i} = randn(num_units(i),num_units(i-1))*sqrt(2/num_units(i-1)); % He initialization
        else
            W{i} = randn(num_units(i),num_units(i-1))*sqrt(1/num_units(i-1)); % Xavier initialization
        end
        b{i} = zeros(num_units(i),1); % ��ʼ���������Ԫ��ƫ��
    end
    % Adam�㷨�����ʼ����
    for i = 1 : L
        Vdw{i} = zeros(size(W{i})); %'momentum'
        Vdb{i} = zeros(size(b{i}));
        Sdw{i} = zeros(size(W{i})); %'RMSprop'
        Sdb{i} = zeros(size(b{i}));
    end   
    % ��mini-baich�������������
    for i = 1 : batch_num
        sample_num(i) = size(Xm{i},2);  % sample_num(j)Ϊ��j��mini-baich��������
    end
    cost_record = []; %��¼ÿ�ε�����cost function��ֵ
    Trainerror_record = []; %��¼ÿ�ε�����ѵ�����

    for t = 1 : num_epoch %epoch: one pass through the training set
        % ÿ�α���һ��ȫ��ѵ������
        for k = 1 : batch_num %iteration: computing on a single mini-batch
            % ÿ�ζ�Ӧһ��mini-batch�����ݶ��½�
            %%% Forward Propagation:
            % ��һ�㣺
            Z{1} = W{1}*Xm{k} + repmat(b{1},1,sample_num(k));
            A{1} = non_linearity(activation_function, Z{1});
            mask{1} = (rand(size(Z{1})) <= keep_prob(1)); % Inverted dropout
            A{1} = ((A{1}.*mask{1})/keep_prob(1)); % assuring that the result of the cost will still have the same expected value as without drop-out
            % �м�㣺
            for i = 2 : L-1 
                Z{i} = W{i}*A{i-1} + repmat(b{i},1,sample_num(k));
                A{i} = non_linearity(activation_function, Z{i});
                mask{i} = (rand(size(Z{i})) <= keep_prob(i));
                A{i} = ((A{i}.*mask{i})/keep_prob(i)); 
            end
            % ����㣺
            Z{L} = W{L}*A{L-1} + repmat(b{L},1,sample_num(k));
            if C >= 2 %���������
                A{L} = Soft_max(Z{L}); %�����ļ������ �����Ϊ��soft max
            else      %����������
                A{L} = Sigmoid(Z{L}) ;  %�����ļ������ �����Ϊ��sigmoid function
            end
            
            %%% �ôε�����ص����ݼ�¼��
            J = CostFuncion(A{L},Ym{k}); %����ǰ�򴫲���cost function: cross-entropy cost
            % Ӳ�о�
            % Q = round(A{L}); 
            % err = length(find(sum(abs(Q-Ym{k}),1)~=0))/sample_num(k);
            % ���о�
            [val_Yp,index_Yp] = max(A{L});
            [val_Y,index_Y] = max(Ym{k});
            err = length(find(index_Yp ~= index_Y))/sample_num(k);
            Trainerror_record = [Trainerror_record, err]; %��ǰ������ĸ���
            cost_record = [cost_record, J]; %��ǰCost function��ֵ      

            %%% Backward Propagation:
            % ����㣺
            dZ{L} = A{L}-Ym{k}; % 'dX'��ʾJ:cost function�����߶�Ӧ��L:Loss function���Ա���������'X'��ƫ��
            dW{L} = (dZ{L}*A{L-1}')/sample_num(k);
            db{L} = sum(dZ{L},2)/sample_num(k);
            dA{L-1} = W{L}'*dZ{L};
            dA{L-1} = ((dA{L-1}.*mask{L-1})/keep_prob(L-1)); 
            % �м�㣺
            for i = L-1 : -1 : 2
                dZ{i} = dA{i}.*d_non_linearity(activation_function, Z{i});
                dW{i} = (dZ{i}*A{i-1}')/sample_num(k);
                db{i} = sum(dZ{i},2)/sample_num(k);
                dA{i-1} = W{i}'*dZ{i};
                dA{i-1} = ((dA{i-1}.*mask{i-1})/keep_prob(i-1)); 
            end
            % ��һ�㣺
            dZ{1} = dA{1}.*d_non_linearity(activation_function, Z{1});
            dW{1} = (dZ{1}*Xm{k}')/sample_num(k);
            db{1} = sum(dZ{1},2)/sample_num(k);

            kk = kk + 1;
            %%% Ȩֵ���£�
            switch update_algorithm
                case {'SGD'}
                    for i = 1 : L
                        % ���ñ�׼�ݶ��½�����Ȩֵ����
                        W{i} = W{i} - learning_rate*dW{i};
                        b{i} = b{i} - learning_rate*db{i};
                    end
                case {'Momentum'}
                    for i = 1 : L
                        % ����'momentum'�㷨����Ȩֵ����:
                        Vdw{i} = (B1*Vdw{i}+(1-B1)*dW{i});
                        Vdb{i} = (B1*Vdb{i}+(1-B1)*db{i});
                        W{i} = W{i} - learning_rate*Vdw{i};
                        b{i} = b{i} - learning_rate*Vdb{i};
                    end
                case {'Nesterov_Momentum'}
                    for i = 1 : L
                        % ����'Nesterov_Momentum'�㷨����Ȩֵ����:
                        Vdw_old{i} = Vdw{i};
                        Vdb_old{i} = Vdb{i};
                        Vdw{i} = B1*Vdw{i} - learning_rate*dW{i};
                        Vdb{i} = B1*Vdb{i} - learning_rate*db{i};
                        W{i} = W{i} + (-B1*Vdw_old{i} + (1+B1)*Vdw{i});
                        b{i} = b{i} + (-B1*Vdb_old{i} + (1+B1)*Vdb{i});
                    end
                case {'RMSprop'}
                    for i = 1 : L
                        % ����'RMSprop'�㷨����Ȩֵ����:
                        Sdw{i} = (B2*Sdw{i}+(1-B2)*dW{i}.^2);
                        Sdb{i} = (B2*Sdb{i}+(1-B2)*db{i}.^2);
                        W{i} = W{i} - (learning_rate*dW{i}./(sqrt(Sdw{i})+s));
                        b{i} = b{i} - (learning_rate*db{i}./(sqrt(Sdb{i})+s));
                    end
                case {'Adam'}
                    for i = 1 : L
                        % ����Adam�㷨����Ȩֵ����
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
    disp('�������')
    
    %%%%%% ��ͼ
    % Cost_Function/Training_Error versus. #iteration
    if plot_flag
        set(0,'defaultfigurecolor','w'); % ͼƬ����Ϊ�׵�
        figure(1)
        plot(cost_record);
        xlabel('#iteration(batch size = 512)');ylabel('Training Lost'); title('MNIST Multi-Layer Perceptron with dropout');grid on;
        figure(2)
        plot(1-Trainerror_record);
        xlabel('#iteration(batch size = 512)');ylabel('Training accuracy');title('MNIST Multi-Layer Perceptron with dropout');grid on;
    end
    
    %%%%%% ��������
    % �����������Ԫ��Ȩֵ����W��ƫ������b��cell�ṹ��
    if save_flag
        save('Myfcnn_show.mat','W','b');
        disp('Ȩֵ����W��ƫ������b�ѳɹ�����')
    end
%     save('Loss_SGD.mat','cost_record');
%     save('Loss_Momentum.mat','cost_record');
%     save('Loss_RMSprop.mat','cost_record');
%     save('Loss_Adam.mat','cost_record');
end

%%%%%%%%%%  �Ӻ���ģ��  %%%%%%%%%%
%%% �����Լ���� forward 
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

%%% �����Լ�����ĵ��� backward 
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

%%%%%% ����Mini-batch����
function [Xm,Ym,batch_num] = My_Minibatch (X,Y,batch_size)
%%% Mini-batch���� 
% XΪ�������ݼ���YΪ���ݼ���Ӧ�ı�ǩ��sizeΪmini-batch��С
% Xm��YmΪ������� cell�ṹ��batch_numΪ�������
% sizeһ��ȡ64.128.256,512��
    [n0,m] = size(X);  % n0Ϊ��������ά����mΪ���Լ���С��������
    [C,m1] = size(Y); % CΪ�������
    if m1 ~= m
        error('�����������ǩ������ƥ��');
    end
    if batch_size >= m
        Xm{1} = X;
        Ym{1} = Y;
        batch_num = 1;
    else
        num = floor(m/batch_size);
        res = mod(m,batch_size);
        % ȷ���������
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