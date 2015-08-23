function knn = knnclassify_f(sample,train,group,k)
%----------------------------------------------------------%
% Function to classify data using nearest neighbor method
%----------------------------------------------------------%

% Determine size of the matrix for further calculations
[s_rowsize,s_colsize] = size(sample);
[t_rowsize,t_colsize] = size(train);
[g_rowsize,g_colsize] = size(group);

% Initial Validations:
% a) If input arguments passed are less than 3
if nargin < 3
    error(message('Argument size mismatch'));
% b) If arguments are less than 4, default k = 1
elseif nargin < 4
    k = 1;
% c) If the input value for k is non-numeric
elseif ~isnumeric(k)
    error(message('k should be a numeric value'));
% d) If the input value for k is in decimals
elseif k < 1
    error(message('k is less than 1'));
end

% If the number of columns of sample and training do not match
if (t_colsize ~= s_colsize)
    error(message('Column size of sample and training should match'));
end

% If the number of rows of training and class do not match
if (t_rowsize ~= g_rowsize)
    error(message('Row size of training and class should match'));
end


for i=1:s_rowsize,
% Metric used to calculate distance: Euclidean 
% Euclidean disance would calculate the pairwise distance 
% between two sets of observations
     distance(i,:)= pdist2(sample(i,:),train,'euclidean');

% Sort the distance in ascending order and create a subset of the
% nearest neighbors on the basis of k
     [s_distance,indices]=sort(distance(i,:));
      grp_idx = indices(:,1:k);
% Find the class corresponding to each nearest neighbors from the
% group using index grp_idx 
    class = group(grp_idx);
% If more than one class is identified for a sample, choose the 
% class that repeats the most, or if its a tie-pick the class that is 
% closest to the test set
    [unique_classify,~,ic]=unique(class);
    knn(i,g_colsize)= unique_classify(mode(ic));
end



