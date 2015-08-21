%----------------------------------------------------------------%
% Script to build a decision tree 
%----------------------------------------------------------------%

% Read the input file
dataset=readtable('fruits_decisionTree.csv');

% Declarations
type = dataset{:,1}; 
features=dataset{:,:};
dec_tree=[];
label=cell(1,31);
cols={'Banana','Orange','Other','Green','Long','Orange','Short','Sour','Sweet','Yellow'};
main_entropy = 0;
rule=[];
% Convert categorical values to numerical
class = categorical(type);
class_count=zeros(size(class));
num_features=categorical(features);
numeric_features = double(num_features);

% Count the number of classes
class_count = countcats(class);

% Determine size of different matrices
[class_rsize,class_csize] = size(class_count);
[f_rsize,f_csize] = size(numeric_features);
subf_mat = zeros(f_rsize,2);
information_gain=zeros(f_csize-1,1);

count_remaining_features = f_csize - 1;  
total = sum(class_count);

% Main Entropy calculation for classification
for i=1:class_rsize,
  prob_class=class_count(i)/total  ;
  main_entropy = main_entropy + (-(prob_class*log2(prob_class)));
end

% Calculate the entropy for each feature provided
for j=2:f_csize,
subf_mat=[numeric_features(:,1),numeric_features(:,j)];
entropy=SplitAttribute(subf_mat,total);
information_gain(j-1)=main_entropy-entropy;  
end

% Determine the feature with high information gain to consider as root node
u=1;
[root,ind] = max(information_gain);

% Create a dynamic matrix dec_tree to capture the nodes for tree-plot at
% each level
level1 = u;            
dec_tree=[dec_tree,0];
label(u)={'Root'};

main_index = ind+1;
count_remaining_features = count_remaining_features - 1;

% Sort the rows by the first column
sortedB = sortrows(numeric_features,ind+1);  
% Find indices of unique values
 [~,~,uniqueIndex] = unique(sortedB(:,ind+1)) ; 

 % Matrix determines the 
 cellB = mat2cell(sortedB,...                       
                     accumarray(uniqueIndex(:),1),f_csize);
  
 [b_rsize,b_csize]=size(cellB);
 d=zeros(size(cellB));

 %------------------------------------------------------------------------------%
 % Logic Implemented: Root node is the feature with the highest information gain
 % a) Split the feature with multiple attributes
 % b) For every attribute and remaining features, calculate the entropy 
 % c) Until entropy becomes zero or remaining features is less than 1,
 % determine leaf nodes
 % d) If entropy is not zero at the last feature consider max. frequency of
 % class
 % e) Plot decision tree by maintaining levels at attribute split
 % f) Maintain lables at each node of decision tree
 %------------------------------------------------------------------------------%
for v=1:b_rsize,
 indexes =[];
 indexes=[indexes,main_index];
 d=cell2mat(cellB(v));
 [d_rsize,d_csize]=size(d);
 u=u+1;
 level2 = u;
 dec_tree=[dec_tree,level1];
 label(u)={cols{d(1,main_index)}};

 % Calculate the entropy for the attribute 
 Entropy_MF = Entropy_calculation(d,d_rsize);
 i_gain=zeros(count_remaining_features,1);

% Check if features lead to a leaf node based on root split
 a = unique(d(:,1));
 if(size(a)==1)
     leaf=unique(d(:,1));
     u=u+1;
     level3=u;
     dec_tree=[dec_tree,level2];
     label(u)={cols{leaf}};
     row=[leaf,d(1,main_index),0,0];
     rule=[rule;row];
 else
 for z=2:d_csize,
     if (ismember(z,indexes)),
     continue
     else    
 subf_mat=[d(:,1),d(:,z)];
 entropy=SplitAttribute(subf_mat,d_rsize);
 i_gain(z-1)=Entropy_MF-entropy; 
     end
 end
 
%  Leaf node for the attribute of the root feature is determined
 [root,ind] = max(i_gain);
 indexes=[indexes,ind+1];
 count_remaining_features = count_remaining_features - 1;
 
 sortedC = sortrows(d,ind+1); 
 [~,~,uniqueIndex] = unique(sortedC(:,ind+1)) ; 
 cellC = mat2cell(sortedC,accumarray(uniqueIndex(:),1),d_csize); 
 [e_rsize,e_csize]=size(cellC);
 
 for h=1:e_rsize,
 e=cell2mat(cellC(h));
 u=u+1;
 level3=u;
 dec_tree=[dec_tree,level2];
 label(u)={cols{e(1,ind+1)}};
 [e_rsize,e_csize]=size(e);
 Entropy_MF = Entropy_calculation(e,e_rsize);
 a= unique(e(:,1));

 % If one unique 
 if(size(a)==1)
      leaf=unique(e(:,1));
      u=u+1;
      level4 = u;
      dec_tree=[dec_tree,level3];
      label(u)={cols{leaf}};
      row=[leaf,d(1,main_index),e(1,ind+1),0];
     rule=[rule;row];
 else
  for g=2:e_csize,
     if (ismember(g,indexes)),
     continue,
     else

 sortedD = sortrows(e,g);                   
 [~,~,uniqueIndex] = unique(sortedD(:,g)) ; 
 
 cellD = mat2cell(sortedD,...                       
                     accumarray(uniqueIndex(:),1),e_csize);
  
 [f_rsize,f_csize]=size(cellD);
 for q=1:f_rsize
      f=cell2mat(cellD(q));
      a= unique(f(:,1));
      u=u+1;
      level4 = u;
      dec_tree=[dec_tree,level3];
      label(u)={cols{f(1,g)}};
 if(size(a)==1)
     leaf=unique(f(:,1));
      u=u+1;
      level5=u;
      dec_tree=[dec_tree,level4];
      label(u)={cols{leaf}};
      row=[leaf,d(1,main_index),e(1,ind+1),f(1,g)];
     rule=[rule;row];
 else
      leaf=mode(f(:,1));
      u=u+1;
      level5=u;
      dec_tree=[dec_tree,level4];
      label(u)={cols{leaf}};
      row=[leaf,d(1,main_index),e(1,ind+1),f(1,g)];
     rule=[rule;row];
      
 end

 end
  clearvars level3,level4,level5;
end
  end
 end
 end
 end
 clear indexes
end

% Display labels in the decision tree
 treeplot(dec_tree);
 [xs,ys,h,s] = treelayout(dec_tree);
 
 for o = 2:numel(dec_tree)
	x = xs(o);
	y = ys(o);
    
	p_x = xs(dec_tree(o));
	p_y = ys(dec_tree(o));

	mid_x = (x + p_x)/2;
	mid_y = (y + p_y)/2;

	text(mid_x,mid_y,label{o});
    
 end
 
%Rule prediction:
%Features should be in numerical values as defined in matrix

n=input('Enter the features');

output=predict(rule,n);
disp(output);

