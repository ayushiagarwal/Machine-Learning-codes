function [ entropy ] = Entropy_calculation( attributes,total )

%--------------------------------------------------------------------%
% Function will calculate entropy for each attribute 
%--------------------------------------------------------------------%
[s_rattributes,s_cattributes] =size(attributes);  
[unique_c,~,idx]=unique(attributes(:,1));
[sr_unique,sc_unique]=size(unique_c);

 counts=zeros(sr_unique,1);
 probability=zeros(sr_unique,1);

 counts = accumarray(idx(:),1,[],@sum);
 t_features=sum(counts);
 probability=counts/t_features;
 entropy=((t_features/total)*(-(dot(probability,log2(probability)))));

end


