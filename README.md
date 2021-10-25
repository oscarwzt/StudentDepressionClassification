# StudentDepressionClassification
Here we have data of 2000 students in the following categories: depression (1 means depressed, 0 means not depressed), the number of friends, the number of Facebook friends, extraversion, neuroticism, stress, cortisol.

The goal of this project is to predict whether a student is depressed based on the other 6 variables. In other words, depression is the target variable, and the other 6 are predictor variables.

First, I ran a PCA on the predictors because based on the correlation matrix, many variables are collinear. The other reason is to reduce dimension. From analyzing the screeplot, I determined that there were 2 principal components (PC). From the loadings matrix, the first PC can be summarized as "challenges" (neuroticism, stress, cortisol). The second PC can be summarized as "support" (the number of friends, the number of Facebook friends). 

![challengeVSsupport](https://user-images.githubusercontent.com/71715227/138627024-14cb7fdd-7217-40d7-9f81-f7407a324353.png)

As we can see, there seem to be 4 clusters in the plot above. Although it is quite obvious, I used the Silhouette method to make it less arbitary. The method confirmed that there are 4 clusters, and then I plot the clusters:

![cluster](https://user-images.githubusercontent.com/71715227/138627691-a8d5cea6-9d0e-49e2-a149-8e339eb97104.png)

