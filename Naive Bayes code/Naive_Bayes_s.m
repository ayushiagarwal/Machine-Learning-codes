%-----------------------------
% Load Fruits.CSV file
%-----------------------------
fruits=readtable('fruits.csv');
%Reading no of features from user
n=input('Enter the no. of features: ');
features=cell(n,1);
%Reading the n number of features names from user
for i=1:n
    features{i,1}=input('Enter the feature name present in the data set: ');
end
type = fruits{:,1}; 
training=fruits{:,features};

%Matrix to calculate total number of samples per class using binary features 
class_count=fruits(:,2:3);
%Determine the total number of samples per class
count_c=sum(class_count{:,:},2);
%Create a table of classes and their respective counts to pass in  function call
classes=table(type,count_c);

% Call to Naive Bayes Classifier to classify the type of class
class_type=Naive_Bayes_f(training,classes);