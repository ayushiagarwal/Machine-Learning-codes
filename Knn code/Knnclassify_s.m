load fisheriris
%--------------------------------------------------------------------%
% Program to perform 10 cross validation and report accuracy using k
% nearest neighbor classification method 
%--------------------------------------------------------------------%
class = species;
features = meas;

k = input('Enter k: ');

% Classify and perform 10 fold cross validation using classifier knnclassify_f
cp = cvpartition(class,'k',10); 

class_type=@(XTRAIN,YTRAIN,XTEST)knn_classify(XTEST,XTRAIN,YTRAIN,k);
err_rate = crossval('mcr',features,class,'predfun',class_type,'partition',cp)
accuracy = 1 - err_rate




