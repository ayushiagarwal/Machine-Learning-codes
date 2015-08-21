%Function for defining logistic regression for online learning:
%This particular function will calculate new weights over every  single
%iteration and calculate new weights on each iteration based on error and
%learning rate

%Input for the function is input features(x1,x2,x0) and target output r and
%learning rate(eta).Output from the function is new weights,error for
%each epoch annd max no of iterations
function [ w,global_error,iteration_f] = logistic_regression_online( x,r,l_rate )
global_error= [];
iteration_f=[];
[f_rsize,f_csize] = size(x);
w = zeros(f_csize,1);
dw = zeros(f_csize,1);
convergence=0;
for j=1:f_csize,
   w(j) = -.01+(.01+.01)*rand;
end

iteration=0; 
 while((convergence~=1)&& (iteration<=1000)),%run the loop till convergence has reached or max no iterations has reached
     g_err=0;
     iteration=iteration +1;
for j=1:f_csize,
     dw(j) = 0; 
end
 for t=1:f_rsize,

   s = 0;
   for j=1:f_csize,
       s = s + w(j)*x(t,j);
    end
   y= 1/(1+exp(-s));
   error=r(t)-y; %Calculate error
   for j = 1:f_csize,
       dw(j) = dw(j) + (r(t) - y)*x(t,j) ;
        w(j) = w(j)+ l_rate*dw(j); %calculate new weights for every iteration
   end
   g_err=g_err+sqrt(error*error);
end


global_error=[global_error;(g_err/f_rsize)];
iteration_f=[iteration_f;iteration];
%When global error becomes zero convergence is achieved
if(g_err==0),
       convergence=1;
end
 end
end