#Define plot_decision_regions function
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

#Part 1: Exploratory Data Analysis
import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['wine_category', 'Atrribute1', 'Atrribute2', 'Atrribute3', 'Atrribute4', 'Atrribute5', 'Atrribute6', 'Atrribute7', 'Atrribute8', 'Atrribute9', 'Atrribute10', 'Atrribute11', 'Atrribute12', 'Atrribute13']


#Head, tail and statistical summary of the data
print("Head, tail and statistical summary of the data:")
print(df_wine.head())
df_wine.head().to_excel("head.xls")
print(df_wine.tail())
df_wine.tail().to_excel("tail.xls")
summary=df_wine.describe()
summary.to_excel("statistical summary.xls")
print(summary) 

import matplotlib.pyplot as plt
import seaborn as sns
cols = ['Atrribute1', 'Atrribute2', 'Atrribute3', 'Atrribute4', 'Atrribute5', 'Atrribute6', 'Atrribute7', 'Atrribute8', 'Atrribute9', 'Atrribute10', 'Atrribute11', 'Atrribute12', 'Atrribute13']

#generate scatterplot matrix
print("Scatterplot Matrix:")
sns.pairplot(df_wine[cols], size=2.5)
plt.title('Scatterplot Matrix')
plt.tight_layout()
plt.savefig("scatterplot matrix")
plt.show()

#generate the correlation matrix array as a heat map
print("Correlation Heat Map:")
cm = np.corrcoef(df_wine[cols].values.T)
sns.set(font_scale=1.5)
plt.rcParams['figure.figsize']=(13.0, 13.0)
plt.title('Correlation Heat Map')
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.savefig("correlation heat map")
plt.show()

#Generate the unstandardized box plot
print("Unstandardized Box Plot:")
array=df_wine[cols].values
plt.rcParams['figure.figsize']=(6.0, 6.0)
plt.title('Unstandardized Box Plot')
plt.boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges")
plt.savefig("unstandardized box plot")
plt.show()

#Generate the standardized box plot
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
np_std = sc.fit_transform(df_wine[cols])
df_std = DataFrame(np_std)
array=df_std.values
plt.rcParams['figure.figsize']=(6.0, 6.0)
plt.title('Standardized Box Plot')
plt.boxplot(array)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges")
plt.savefig("standarized box plot")
plt.show()

#split the data
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# standardize the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#Part 2: Logistic regression classifier v. SVM classifier - baseline
#Logistic regression 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)
accuracy_score_trainlr=lr.score(X_train_std, y_train)
print("The train accuarcy score of logistic regression: ", accuracy_score_trainlr)
accuracy_score_testlr=lr.score(X_test_std, y_test)
print("The test accuarcy score of logistic regression: ", accuracy_score_testlr)
#SVM classifier
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
accuracy_score_trainlr=svm.score(X_train_std, y_train)
print("The train accuarcy score of SVM regression: ", accuracy_score_trainlr)
accuracy_score_svm=svm.score(X_test_std, y_test)
print("The test accuarcy score of SVM classifier: ", accuracy_score_svm,'\n')


#Part 3: Perform a PCA on both datasets
print("Perform a PCA on both datasets, the results are as follows: ")
#Perform PCA fit and transform
from sklearn.decomposition import PCA 
pca = PCA(n_components=2) 
pca.fit(X_train_std)
X_train_pca = pca.transform(X_train_std) 
X_test_pca = pca.transform(X_test_std) 
#Logistic regression 
from sklearn.linear_model import LogisticRegression
lr_pca = LogisticRegression(C=100.0, random_state=1)
lr_pca.fit(X_train_pca , y_train)
accuracy_score_train_lrpca=lr_pca.score(X_train_pca , y_train)
print("The train accuarcy score of logistic regression: ", accuracy_score_train_lrpca)
accuracy_score_test_lrpca=lr_pca.score(X_test_pca, y_test)
print("The test accuarcy score of logistic regression: ", accuracy_score_test_lrpca)
plot_decision_regions(X_test_pca, y_test, classifier=lr_pca)
plt.title('LR_PCA Desicion Region Plot')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='upper left')
plt.savefig("LR_PCA Desicion Region Plot")
plt.show()
#SVM classifier
from sklearn.svm import SVC
svm_pca = SVC(kernel='linear', C=1.0, random_state=1)
svm_pca.fit(X_train_pca , y_train)
accuracy_score_train_svmpca=svm_pca.score(X_train_pca , y_train)
print("The train accuarcy score of SVM regression: ", accuracy_score_train_svmpca)
accuracy_score_test_svmpca=svm_pca.score(X_test_pca, y_test)
print("The test accuarcy score of SVM classifier: ", accuracy_score_test_svmpca,'\n')
plot_decision_regions(X_test_pca, y_test, classifier=svm_pca)
plt.title('SVM_PCA Desicion Region Plot')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='upper left')
plt.savefig("SVM_PCA Desicion Region Plot")
plt.show()

#Part 4: Perform and LDA on both datasets
print("Perform a LDA on both datasets, the results are as follows: ")
#Perform LDA fit and transform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
lda.fit(X_train_std, y_train)
X_train_lda = lda.transform(X_train_std)
X_test_lda = lda.transform(X_test_std)
#Logistic regression 
from sklearn.linear_model import LogisticRegression
lr_lda = LogisticRegression(C=100.0, random_state=1)
lr_lda.fit(X_train_lda, y_train)
accuracy_score_train_lrlda=lr_lda.score(X_train_lda, y_train)
print("The train accuarcy score of logistic regression: ", accuracy_score_train_lrlda)
accuracy_score_test_lrlda=lr_lda.score(X_test_lda, y_test)
print("The test accuarcy score of logistic regression: ", accuracy_score_test_lrlda)
plot_decision_regions(X_test_lda, y_test, classifier=lr_lda)
plt.title('LR_LDA Desicion Region Plot')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='upper left')
plt.savefig("LR_LDA Desicion Region Plot")
plt.show()
#SVM classifier
from sklearn.svm import SVC
svm_lda = SVC(kernel='linear', C=1.0, random_state=1)
svm_lda.fit(X_train_lda, y_train)
accuracy_score_train_svmpca=svm_lda.score(X_train_lda, y_train)
print("The train accuarcy score of SVM regression: ", accuracy_score_train_svmpca)
accuracy_score_test_svmpca=svm_lda.score(X_test_lda, y_test)
print("The test accuarcy score of SVM classifier: ", accuracy_score_test_svmpca,'\n')
plot_decision_regions(X_test_lda, y_test, classifier=svm_lda)
plt.title('SVM_LDA Desicion Region Plot')
plt.savefig("SVM_LDA Desicion Region Plot")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='upper left')
plt.savefig("SVM_LDA Desicion Region Plot")
plt.show()

#Part 5: Perform a kPCA on both datasets
import math
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
list_gamma=[]
list_accuracyscore_train_lrkpca=[]
list_accuracyscore_test_lrkpca=[]
list_accuracyscore_train_svmkpca=[]
list_accuracyscore_test_svmkpca=[]
for i in range (-3,3):
    print("Perform a KPCA on both datasets when gamma = 10^",i,"，the results are as follows: ")
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=math.pow( 10, i ))
    scikit_kpca.fit(X_train_std)
    X_train_kpca=scikit_kpca.transform(X_train_std)
    X_test_kpca=scikit_kpca.transform(X_test_std)
    
    #Logistic regression 
    lr = LogisticRegression(C=100.0, random_state=1)
    lr.fit(X_train_kpca, y_train)
    accuracy_score_train_lrkpca=lr.score(X_train_kpca, y_train)
    print("The train accuarcy score of logistic regression when gamma = 10^",i,": ", accuracy_score_train_lrkpca)
    accuracy_score_test_lrkpca=lr.score(X_test_kpca, y_test)
    print("The test accuarcy score of logistic regression when gamma = 10^",i,": ", accuracy_score_test_lrkpca)
    #SVM classifier
    svm = SVC(kernel='linear', C=1.0, random_state=1)
    svm.fit(X_train_kpca, y_train)
    accuracy_score_train_svmkpca=svm.score(X_train_kpca, y_train)
    print("The train accuarcy score of logistic regression: ", accuracy_score_train_svmkpca)
    accuracy_score_test_svmkpca=svm.score(X_test_kpca, y_test)
    print("The test accuarcy score of SVM classifier: ", accuracy_score_test_svmkpca)
    
    #define arrays
    gamma=math.pow( 10, i)
    list_gamma.append(gamma)
    list_accuracyscore_train_lrkpca.append(accuracy_score_train_lrkpca)
    list_accuracyscore_test_lrkpca.append( accuracy_score_test_lrkpca)
    list_accuracyscore_train_svmkpca.append(accuracy_score_train_svmkpca)
    list_accuracyscore_test_svmkpca.append(accuracy_score_test_svmkpca)

accuracyscore_kpca=np.vstack((list_gamma, list_accuracyscore_train_lrkpca, list_accuracyscore_test_lrkpca,list_accuracyscore_train_svmkpca, list_accuracyscore_test_svmkpca))   
df_accuracyscore_kpca=pd.DataFrame(accuracyscore_kpca,columns=['','','','','',''])
df_accuracyscore_kpca.index=Series(['gamma_val','accuracyscore_train_lr_kpca','accuracyscore_test_lr_kpca','accuracyscore_train_svmkpca','accuracyscore_test_svmkpca'])
df_accuracyscore_kpca.to_excel('accuracyscore_kpca.xls')
print(df_accuracyscore_kpca)

print("My name is QI GUO")
print("My NetID is: qiguo3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################






























