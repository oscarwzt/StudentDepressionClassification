#StudentDepressionClassification

[depressionData.csv](https://github.com/oscarwzt/StudentDepressionClassification/files/7407089/depressionData.csv)
Here we have data of 2000 students in the following categories: depression (1 means depressed, 0 means not depressed), the number of friends, the number of Facebook friends, extraversion, neuroticism, stress, cortisol.

The goal of this project is to predict whether a student is depressed based on the other 6 variables. In other words, depression is the target variable, and the other 6 are predictor variables.

First, I ran a PCA on the predictors because based on the correlation matrix, many variables are collinear. The other reason is to reduce dimension. From analyzing the screeplot, I determined that there were 2 principal components (PC). From the loadings matrix, the first PC can be summarized as "challenges" (neuroticism, stress, cortisol). The second PC can be summarized as "support" (the number of friends, the number of Facebook friends). 

![supportVSchallenge](https://user-images.githubusercontent.com/71715227/138629290-edbf4703-aa00-4d98-8420-1d46392e7b87.png)

As we can see, there seem to be 4 clusters in the plot above. Although it is quite obvious, I used the Silhouette method to make it less arbitary. The method confirmed that there are 4 clusters, and then I plot the clusters:

![cluster](https://user-images.githubusercontent.com/71715227/138629220-245d143e-2f85-4946-bbbd-fd30249e3419.png)

At first glance, it seems like those who are in the 4th quadrant are more likely to be depressed. In the following plot, green means not depressed, and blue means depressed. 

![depressedVSnot](https://user-images.githubusercontent.com/71715227/138629368-18d920f9-60bb-4b28-ab55-d77a50e551c3.png)

The plot confirmed the speculation: those with high challenges and low support are likely to suffer from dimension.
Now, to make a classification, I tested SVM, decision tree, and logistic regression classfication algorithms. The train and test dataset was a 2:1 split. Out of those three algorithms, SVM performed the best, so I built the model based on it. The model achieved an average accuracy of 92.5% from 5-fold cross-validation.
