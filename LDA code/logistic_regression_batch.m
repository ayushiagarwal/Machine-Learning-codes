%Function for defining logistic regression for batch learning:
%This particular function will calculate new weights over one complete
%epoch based on the error function and learning rate.

%Input for the function is input features(x1,x2,x0) and target output r and
%learning rate(eta).Output from the function is new weights,error for
%each epoch annd max no of iterations
function [ w,global_error,iteration_f ] = logistic_regression_batch( x,r,l_rate )

global_error= [];
iteration_f=[];
[f_rsize,f_csize] = size(x);
w = zeros(f_csize,1);
dw = zeros(f_csize,1);
%error=zeros(f_rsize,1);
convergence=0;
for j=1:f_csize,
   w(j) = -.01+(.01+.01)*rand;
end

iteration=0; 
 while((convergence~=1)&&(iteration<=1000)), % when convergence has reached or max no of iterations has reached
     iteration=iteration +1;
     g_err=0;
for j=1:f_csize,
     dw(j) = 0; 
end
for t=1:f_rsize,
   s = 0;
   for j=1:f_csize,
       s = s + w(j)*x(t,j);
    end
   y= 1/(1+exp(-s));
   error=r(t)-y;
   
   for j = 1:f_csize,
       dw(j) = dw(j) + (r(t) - y)*x(t,j) ;
   end
   g_err=g_err+sqrt(error*error);
 % g_err=max(error);
end

for j = 1:f_csize,
   w(j) = w(j)+ l_rate*dw(j); % calculate new weights for new epoch
end
global_error=[global_error;(g_err/f_rsize)];
iteration_f=[iteration_f;iteration];
% when error becomes zero convergence has reached
if(g_err==0),
       convergence=1;
end
 end
end