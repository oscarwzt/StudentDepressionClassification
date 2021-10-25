# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:37:36 2021

@author: Oscar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from sklearn.metrics import mean_squared_error as MSE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
#Principle Component Regression
#%% loading data
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
data = pd.read_csv('middleSchoolData.csv')
cleanedData = data.dropna()
predictors = cleanedData.drop
x = cleanedData.drop(cleanedData.columns[[0, 1, 3]], axis = 1)
y = cleanedData['acceptances']/cleanedData['school_size']

catName = x.columns
#charterSchoolData = data.iloc[]
a = np.mean(data['per_pupil_spending'])

xcs = csData.drop(csData.columns[1], axis = 1)
#%% PCA
from scipy.stats import zscore
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
xZscored = x.apply(zscore)
yCentered = y-np.mean(y)#y.apply(zscore)
r = xZscored.corr()
#plt.imshow(r) 
#plt.colorbar()
pca = PCA().fit(xZscored)
eigenValues = pca.explained_variance_
eigenVectors =  pca.components_
xReduced = pca.fit_transform(xZscored)
ratio = pca.explained_variance_ratio_
#plt.bar(x = range(1, len(ratio)+1), height = ratio)
temp = np.linspace(0,21,21)
plt.bar(temp, eigenVectors[3,:])
temp1 = eigenVectors[:,0]
temp2 = eigenVectors[0]

temp = np.matmul(xZscored, eigenVectors)
#%% regress
from sklearn.model_selection import RepeatedKFold
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

regress = linear_model.LinearRegression()

xPC = xReduced[:,:3]
tempxpc = np.exp(xPC)
regress.fit(tempxpc, y)
prediction = regress.predict(tempxpc)
rmse = mse(y, prediction)**0.5
r = regress.score(tempxpc, y)
r = np.corrcoef(prediction, np.log(y+1))
plt.plot(prediction, y , 'o')
#plt.plot(y)
#%%
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
yZscored = (y- np.mean(y))/np.std(y)
RMSE = []
#plt.hist(yZscored, bins = 30)
for i in np.arange(0, 21):
    score = -1*cross_val_score(regress, xReduced[:,:i], y, cv=cv, scoring='neg_root_mean_squared_error').mean()
    RMSE.append(score)
plt.plot(RMSE)
#plt.bar(x = range(0,21), height = mse)
plt.xlabel('Number of Principal Components')
plt.ylabel('MSE')
plt.title('hp')
#%% random forest
rf.fit(xPC, y)
predictionRf = rf.predict(xPC)
rmse = mse(y, prediction)**0.5
r = rf.score(xPC, y)

#%% validation
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_validate
def kFoldCV(model,x,y):
    kFold = KFold(n_splits = 10)
    model.fit(x,y)
    #result = -cross_validate(model,xPC,y,cv = kFold, scoring = ('neg_root_mean_squared_error', 'r2'))#.sum()
    #mean, std = np.mean(result), np.std(result)
    
    return cross_validate(model,xPC,y,cv = kFold, scoring = ('neg_root_mean_squared_error', 'r2'))#mean, std
a = kFoldCV(rf, xPC, y)
targetRange = max(y) - min(y)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
a = cross_val_score(rf,xPC,y,cv = cv, scoring = ('r2'))
rmses = []
for ii in range(100):
    xTrain, xTest, yTrain, yTest = train_test_split(xPC, y, test_size = 0.5)
    rf.fit(xTrain, yTrain)
    prediction = rf.predict(xTest)
    rmse = mse(yTest, prediction)
    rmses.append(rmse)
importance = rf.feature_importances_ #student ability
meanRmse = np.mean(rmses) #0.000340054281971137
stdRmse = np.std(rmses) #
print(max(y) - min(y))

#%% charter school initialize
csData = data.iloc[485:, :].reset_index() #all cs
csData = csData.drop(csData.columns[[0, 1, 2, 5,6]], axis = 1) #all cs
xcs = csData.drop(csData.columns[1], axis = 1)
ycs = csData['acceptances']/csData['school_size']

means = np.mean(xcs)
#replacing nan with mean
for ii in range(len(xcs.columns)):
    xcs[xcs.columns[ii]] = xcs[xcs.columns[ii]].replace(np.nan, means[ii])
print(xcs.columns[1])
#%%
x2 = cleanedData.drop(cleanedData.columns[[0, 1, 21]], axis = 1)
y2 = cleanedData[cleanedData.columns[21]]
cs2Data = data.iloc[485:, :].reset_index()
cs2Data = cs2Data.drop(cs2Data.columns[[0, 1, 2, 5,6]], axis = 1)
xcs2 = cs2Data.drop(cs2Data.columns[17], axis = 1)
ycs2 = cs2Data['student_achievement']

means = np.mean(xcs2)
ymeans = np.mean(ycs2)
#ycs2 = np.nan_to_num(ymeans)
print(len(ycs2))
for ii in range(len(xcs2.columns)):
    xcs2[xcs2.columns[ii]] = xcs2[xcs2.columns[ii]].replace(np.nan, means[ii])
for ii in range(len(ycs2)):
    if ycs2[ii] == np.nan:
        ycs2[ii] = ymeans
ycs2 = ycs2.replace(np.nan, ymeans)
#%% pca
xcsZscored = xcs.apply(zscore)
#plt.imshow(r) 
#plt.colorbar()
pca = PCA().fit(xcsZscored)
eigenValues = pca.explained_variance_
eigenVectors =  pca.components_
xReduced = pca.fit_transform(xcsZscored)
ratio = pca.explained_variance_ratio_
csCatName = xcs.columns
temp = np.linspace(0,19,19)
plt.plot(eigenValues)
#plt.bar(temp, eigenVectors[3,:])
#%%
def pcaInfo(x):
    xz = x.apply(zscore)
    pca = PCA().fit(xz)
    eigenValues2 = pca.explained_variance_
    eigenVectors2 =  pca.components_
    xReduced = pca.fit_transform(xz)
    CatName = x.columns
    temp = np.linspace(0,19,19)
    #plt.plot(eigenValues)
    nDraws = 10000 # How many repetitions per resampling?
    numRows = len(x) # How many rows to recreate the dimensionality of the original data?
    numColumns = len(x.columns) # How many columns to recreate the dimensionality of the original data?
    eigSata = np.empty([nDraws,numColumns]) # Initialize array to keep eigenvalues of sata
    eigSata[:] = np.NaN # Convert to NaN
    '''
    for i in range(nDraws):
        # Draw the sata from a normal distribution:
        sata = np.random.normal(0,1,[numRows,numColumns]) 
        # Run the PCA on the sata:
        pca = PCA()
        pca.fit(sata)
        # Keep the eigenvalues:
        temp = pca.explained_variance_
        eigSata[i] = temp
    
    # Make a plot of that and superimpose the real data on top of the sata:
    plt.plot(np.linspace(0,numColumns,numColumns),eigenValues2,color='blue') # plot eigVals from section 4
    plt.plot(np.linspace(0,numColumns,numColumns),np.transpose(eigSata),color='black') # plot eigSata
    plt.plot([1,numColumns],[1,1],color='red') # Kaiser criterion line
    plt.xlabel('Principal component (SATA)')
    plt.ylabel('Eigenvalue of SATA')
    plt.legend(['data','sata'])'''
    
    return xReduced, CatName, eigenVectors2, eigenValues2
    
#plt.bar(temp, eigenVectors[3,:]
#%%% horns
nDraws = 10000 # How many repetitions per resampling?
numRows = 109 # How many rows to recreate the dimensionality of the original data?
numColumns = 19 # How many columns to recreate the dimensionality of the original data?
eigSata = np.empty([nDraws,numColumns]) # Initialize array to keep eigenvalues of sata
eigSata[:] = np.NaN # Convert to NaN

for i in range(nDraws):
    # Draw the sata from a normal distribution:
    sata = np.random.normal(0,1,[numRows,numColumns]) 
    # Run the PCA on the sata:
    pca = PCA()
    pca.fit(sata)
    # Keep the eigenvalues:
    temp = pca.explained_variance_
    eigSata[i] = temp

# Make a plot of that and superimpose the real data on top of the sata:
plt.plot(np.linspace(0,numColumns,numColumns),eigenValues2,color='blue') # plot eigVals from section 4
plt.plot(np.linspace(0,numColumns,numColumns),np.transpose(eigSata),color='black') # plot eigSata
plt.plot([1,numColumns],[1,1],color='red') # Kaiser criterion line
plt.xlabel('Principal component (SATA)')
plt.ylabel('Eigenvalue of SATA')
plt.legend(['data','sata'])

#%%%
mse = []
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# Calculate MSE using cross-validation, adding one component at a time
for i in np.arange(1, 19):
    score = -1*cross_val_score(rf,
               x2Reduced[:,:i], ycs2, cv=10, scoring='neg_mean_squared_error').mean()
    mse.append(score)
plt.plot(mse)
temp = np.linspace(0,19,19)
for ii in range(0,5):
    plt.bar(temp, eigenVectors2[ii,:])
    plt.title('pc'+str(ii))
    plt.pause(0.01)
pcs2 = np.array(['school environment','diversity', 'white percent','student ability', 'disability'])


#pc2: 
#pc3: student ability
#pc4: disability
xcspc = xReduced[:,:5]
xcs2pc = pd.DataFrame(xReduced[:,:5], columns = [pcs2])
#%% cs multi regress

temp = pd.DataFrame(xcspc, columns = [pcs])
regress.fit(temp, ycs)
pre = regress.predict(temp)
a = regress.score(temp, ycs)
rmse = mse(pre, ycs) 
coefs = regress.coef_
plt.plot(pre, ycs, 'o')   
#plt.plot(ycs, color = 'orange')
rmsecs = []
for ii in range(100):
    xTrain, xTest, yTrain, yTest = train_test_split(temp, ycs, test_size = 0.5)
    regress.fit(xTrain, yTrain)
    prediction = regress.predict(xTest)
    rmse = mse(yTest, prediction)
    rmsecs.append(rmse)
meanrmsecs = np.mean(rmsecs) # 0.000340054281971137
stdrmsecs = np.std(rmsecs) # 4.2276193809594516e-06

#%% cs2 multi
def tryModel(x,y, model):
    model.fit(x,y)
    pre = model.predict(x)
    #a = regress.score(xcs2pc, ycs2)
    rmse = MSE(pre, y) 
    #coefs = regress.coef_
    plt.plot(pre, y, 'o')
    return rmse
#plt.plot(pre, ycs2, 'o')   
#plt.plot(ycs, color = 'orange')

def trainTest(x,y, model):
    rmsecs = []
    R2 = []
    for ii in range(100):
        xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.25)
        model.fit(xTrain, yTrain)
        prediction = model.predict(xTest)
        rmse = MSE(yTest, prediction)
        R2.append(np.corrcoef(prediction, yTest)[0,1])
        rmsecs.append(rmse)
    meanrmsecs = np.mean(rmsecs) # 0.000340054281971137
    stdrmsecs = np.std(rmsecs) # 4.2276193809594516e-06
    meanR2 = np.mean(R2)
    stdR2 = np.mean(R2)
    return meanrmsecs, stdrmsecs, meanR2, stdR2
    return meanrmsecs, stdrmsecs
#%% cs random forest
rf.fit(xcspc, ycs)
pre = rf.predict(xcspc)
rmse = mse(pre, ycs)

rmsecs = []
for ii in range(100):
    xTrain, xTest, yTrain, yTest = train_test_split(xcspc, ycs, test_size = 0.5)
    rf.fit(xTrain, yTrain)
    prediction = rf.predict(xTest)
    rmse = mse(yTest, prediction)
    rmsecs.append(rmse)
meanrmsecs = np.mean(rmsecs) # 1.2049382040828192e-05
stdrmsecs = np.std(rmsecs) # 4.3749002017370275e-06

csfeatures =  pd.Series( rf.feature_importances_,pcs)
print(max(ycs) - min(ycs))

#%%

rfcs2Result = trainTest(xcs2pc, ycs2, rf)
#0.411, 0.092

mrcs2Result = trainTest(xcs2pc, ycs2, regress)
#0.429, 0.118

rfcsResult = trainTest(xcspc, ycs, rf)
mrcsResult = trainTest(xcspc, ycs, regress)

#%%
x2 = cleanedData.drop(cleanedData.columns[[0, 1, 21]], axis = 1)
y2 = cleanedData['student_achievement']
x2Reduced, x2CatName, x2eigenVectors, x2eigenVals = pcaInfo(x2)
temp = []
for i in np.arange(1, 19):
    score = -1*cross_val_score(rf, x2Reduced[:,:i], y2, cv=10, scoring='neg_mean_squared_error').mean()
    temp.append(score)
plt.bar(x = range(0,18), height = temp)
plt.bar(x = range(0,21), height = x2eigenVals)
ratio = x2eigenVals / x2eigenVals.sum()
temp = np.linspace(0,21,21)
for ii in range(0,5):
    plt.bar(temp, x2eigenVectors[ii,:])
    plt.title('pc'+str(ii))
    plt.pause(0.01)
pc2 = np.array(['student ability', 'school environment', 'diversity', 'black percent'])
x2pc = pd.DataFrame(x2Reduced[:,:4], columns = [pc2])

#%% non charter model testing
rfnonCharterAchievementResult = trainTest(x2pc, y2, rf)
print(rfnonCharterAchievementResult)
#0.382, 0.044
multinonCharterAchievementResult = trainTest(x2pc, y2, regress)
print(multinonCharterAchievementResult)
#0.359, 0.043

#%% charter school
rfAdmission = trainTest(xPC, y, rf)
nonAdmImportance = pd.Series(rf.feature_importances_, charterAchivementPc)
print(rfAdmission)
rfcsAdmission = trainTest(xcspc, ycs, rf)
csAdmImportance = rf.feature_importances_#pd.Series(rf.feature_importances_, charterAchivementPc)
print(rfcsAdmission)
rfCSAchivement = trainTest(xcharterAchivement, ycs2, rf)
csachive = rf.feature_importances
print(rfCSAchivement)
nonAchieve = trainTest(xcs2Reduced, ycs2, rf)
#%% pca charter school achievement
xcs2Reduced, xcs2CatName, xcs2eigenVectors, xcs2eigenVals = pcaInfo(xcs2)
temp = []
for i in np.arange(1, 19):
    score = -1*cross_val_score(rf, xcs2Reduced[:,:i], ycs2, cv=10, scoring='neg_mean_squared_error').mean()
    temp.append(score)
plt.bar(x = range(0,18), height = temp)
plt.bar(x = range(0,21), height = xcs2eigenVals)

for ii in range(0,4):
    plt.bar(x = range(0,19) ,height = xcs2eigenVectors[ii,:])
    plt.title('pc'+str(ii))
    plt.pause(0.01)
xcharterAchivement = xcs2Reduced[:,:4]
charterAchivementPc = np.array(['hispanic','staff ability', 'white percent', 'student ability'])
xcharterAchivement = pd.DataFrame(xcs2Reduced[:,:4], columns = [charterAchivementPc])
#%%
charterAchieve = trainTest(xcharterAchivement, ycs2, rf)
print(charterAchieve)
rf.fit(xcharterAchivement, ycs2)
caImportance = pd.Series(rf.feature_importances_, charterAchivementPc)
plt.bar(x = range(0,19), height = xcs2eigenVectors[7,:])
#%% pca charter acceptances
xcsReduced, xcsCatName, xcseigenVectors, xcseigenVals = pcaInfo(xcs2)
temp = []
for i in np.arange(1, 19):
    score = -1*cross_val_score(rf, xcsReduced[:,:i], ycs, cv=10, scoring='neg_mean_squared_error').mean()
    temp.append(score)
plt.bar(x = range(0,18), height = temp)
plt.bar(x = range(0,21), height = xcseigenVals)

for ii in range(0,5):
    plt.bar(x = range(0,19) ,height = xcseigenVectors[ii,:])
    plt.title('pc'+str(ii))
    plt.pause(0.01)
    
charterAccPcs = np.array(['diversity', 'staff ability', 'white percent', 'student ability', 'applications'])
#%% pca non charter achive
x2Reduced, x2CatName, x2eigenVectors, x2eigenVals = pcaInfo(x2)

nonCharterAchieve = trainTest(x2pc, y2, rf)
noncharterimportance = rf.feature_importances_

#%%
a = xcs2['white_percent']
a1 = cleanedData['white_percent']
t, p =  stats.ttest_ind(a, a1, alternative='less')
mannwhitneyu(a, a1, alternative='greater')
