%------------------------------------
% Build a Naive Bayes Classifier
%-------------------------------------
function classification = Naive_Bayes_f(train,classes)

%Calculation of  Total count of all the classes
total_classes=sum(classes{:,2},1);
%Determine the size of matrices for further calculation
classes_rowsize=size(classes,1);
train_colsize=size(train,2);
%Declaring matrices
p_total_likely=ones(classes_rowsize,1);
p_likely=zeros(classes_rowsize,train_colsize);
num_posterior = zeros(classes_rowsize,1);

for i=1:classes_rowsize,
    %Calculating Prior Probability
    p_prior=classes{i,2}/total_classes;
    %Calculating probability of likelihood for each feature
    for j=1:train_colsize,
     p_likely(i,j)=train(i,j)/classes{i,2};
    %Calculting Total Likelihood Probability for all given features 
     p_total_likely(i)=p_total_likely(i)*p_likely(i,j);
    end
    %Calculating numerator of Posterior Probability as evidence always
    %remains constant for classification
    num_posterior(i) = (p_prior*p_total_likely(i));
end
%Classifying classes on the basis of maximum probability
[num_posterior,indices] = max(num_posterior);
 classification = classes{indices,1}
end