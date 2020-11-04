Heterogenous Engagement Auxillary Deepwalk
===
This code implements the model discussed in Heterogenous Engagement Auxillary Deepwalk. Video engagement is important in online advertisements where there is no physical interaction with a user. Videos shown in the same channel are likely to have a similar reviewer rating, hence we use graph embeddings to identify YouTube advertisements that are fraud or genuinely popular. 

Requirements
---
This code is based on the Deepwalk code found at:
https://github.com/apoorvavinod/DeepWalk_implementaion

One Class SVM 
---
Train the model
matlab -r oneclass_svm(inputfile, outputfile, num_users, th)
- inputfile is the click sequence
- outputfile is after removing outlier sequences
- num_users is number of users for training oneclass svm (slow for large number)
- th is the outlier threshold in oneclass svm
