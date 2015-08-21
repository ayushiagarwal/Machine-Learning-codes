% Read the input data set
dataset=readtable('dataset1.csv');
dataset1= readtable('dataset2.csv');
%Learning rates
n=[0.001,.01,0.1,0.5];
%dataset 1
r = dataset{:,4}; 
x=dataset{:,1:3};
%dataset 2
r1 = dataset1{:,4}; 
x1=dataset1{:,1:3};
legendtext=[]
%for batch learning :Pass the input features,target output and learning rate
% Run for different lerning rates

for i=1:size(n,2)
 legendtext=['Learning rate ',num2str(n(i))]
figure
[new_wt_b,err_b,iteration_b]=logistic_regression_batch(x,r,n(i));
plot(iteration_b,err_b,'b');
xlabel('iterations');
ylabel('error');
legend(legendtext)
title('batch learning for datasets');

hold on
[new_wt_b1,err_b1,iteration_b1]=logistic_regression_batch(x1,r1,n(i));
plot(iteration_b1,err_b1,'r');
clearvars net_wt_b1,err_b1,iteration_b1;

end
hold off
 %for online learning :plot different datasets for different learning rates

 for i=1:size(n,2)
 legendtext=['Learing rate ',num2str(n(i))]
 figure
 [new_wt_o,err_o,iteration_o]=logistic_regression_online(x,r,n(i));
 plot(iteration_o,err_o,'b');
 xlabel('iterations');
 ylabel('error');
 legend(legendtext)
 title('online learning for datasets');
 hold on

[new_wt_o1,err_o1,iteration_o1]=logistic_regression_online(x1,r1,n(i));
plot(iteration_o1,err_o1,'r');

 end
hold off
