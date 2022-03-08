Fake YouTube Views Detection
===
This code implements the model discussed in the paper _Heterogenous Engagement Auxillary Deepwalk_. Video engagement is important in online advertisements where there is no physical interaction with a user. Videos shown in the same channel are likely to have a similar reviewer rating, hence we use graph embeddings to identify YouTube advertisements that are fraud or genuinely popular. 

Requirements
---
This code is based on the Deepwalk code found at:
https://github.com/apoorvavinod/DeepWalk_implementaion

One Class SVM 
---
Train the model<br>
*matlab -r oneclass_svm(inputfile, outputfile, num_users, th)*
- inputfile is the click sequence
- outputfile is after removing outlier sequences
- num_users is number of users for training oneclass svm (slow for large number)
- th is the outlier threshold in oneclass svm



Deep Walk Embeddings
---
Create the network<br>
*matlab -r create_network(inputfile, num_neighbours)*
- inputfile is the training sequences
- num_neighbours is max edges for each user
- output is in network.mat<br><br>

Generate the embeddings<br>
*python DeepWalk.py --d 64 --walks 500 --len 10 --window 3 -e -i network.mat -emb embeddings.txt*
- d is length of embedding
- walks is number of random walks
- Len is length of each walk
- window is skip-chain window in each walk
- e save embeddings to file
- i is inputfile name
- emb is output embedding file name<br>

Testing
---
Predict accuracy of test users<br>
*matlab -r network_regression(embeddings, networkfile, trainfile, testfile, num_neighbours)*
- embeddings is generated from DeepWalk
- networkfile is input to DeepWalk
- trainfile is sequences for users in network
- testfile is sequences for new users
- num_neighbours is max edges for test user using highest covariance


Paper Link : https://www.sciencedirect.com/science/article/pii/S0925231221013382?via%3Dihub


