function output=predict(rule,features)
 [r_rsize,r_csize]=size(rule);

 %Predict classes based on user input
 for i=1:r_rsize

         if(features(:,:)==rule(i,2:end));
         leaf=rule(i,1);
         
         if(leaf==1)
             output='Banana';
         elseif(leaf==2)
                 output='Orange';
         else(leaf==3)
                 output='Other';
         
         end 
         
       
         end
 end
