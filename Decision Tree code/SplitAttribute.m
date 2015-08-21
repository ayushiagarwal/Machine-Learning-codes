function total_entropy = SplitAttribute( sub_features,total)

%--------------------------------------------------------------%
% Function will split each attribute of a feature and call
% the entropy function to calculate entropy
%--------------------------------------------------------------%
[sf_rsize,sf_csize]=size(sub_features);
sortedA = sortrows(sub_features,2);  
[~,~,uniqueIndex] = unique(sortedA(:,2));  

cellA = mat2cell(sortedA,...                      
                    accumarray(uniqueIndex(:),1),sf_csize);
 
[c_rsize,c_csize] = size(cellA);
 c=zeros(size(cellA));   
 
entropy_att=zeros(size(cellA)); 
entropy_feature=zeros(size(cellA)); 

for k=1:c_rsize,
    c=cell2mat(cellA(k));    
    entropy_att(k) = Entropy_calculation(c,total);
end
total_entropy=sum(entropy_att);

