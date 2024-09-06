%data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
%data = load('shuttle.dat','-ascii'); function_type = 'c';  %data1 = data(1:700,:);
data = load('california.dat','-ascii'); function_type = 'f'; data1 = data(1:15480,:);
% addpath('../LSSVMlab')
data2=data(15481:end,:);



data=data1;
X = data(:,1:end-1);
Y = data(:,end);

% binarize the labels for shuttle data (comment these lines for
% california!)
Y(Y == 1) = 1;
Y(Y ~= 1) = -1;

testX = data2(:,1:end-1);
testY = data2(:, end);

%%

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 4;
% function_type = 'c'; %'c' - classification, 'f' - regression  
kernel_type = 'lin_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);