import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


path = '../kaggledb/train.csv'
df = pd.read_csv(path)

keys = df.keys()
nondiscrete_keyvals = []

for key in keys:
    print(key)
    if (df[key].dtype == 'int64') and (key!='Id') and (key!='SalePrice'): nondiscrete_keyvals.append(key)
    print(df[key].head())
    print('-'*20 +'\n\n')

Y = np.array(df['SalePrice'])
X = np.array(df[nondiscrete_keyvals])


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_std_train = scaler.fit_transform(X_train)
X_std_test = scaler.transform(X_test)

pca = PCA(n_components=2, svd_solver='full')
X_pca_train = pca.fit_transform(X_std_train)
X_pca_test = pca.transform(X_std_test)


############################
## example run
##
## turns out with the pca 
## 2 components is the 
## optimum > some values
## are not meant to be 
## used for regression
## ie yes/no numbers
## we'll slove that later
## with mutual information
############################

svr = SVR(C=438269, epsilon=10571)
svr.fit(X_pca_train,Y_train)


X_xmin = np.min(X_pca_train, axis=0)[0]
X_ymin = np.min(X_pca_train, axis=0)[1]
X_xmax = np.max(X_pca_train, axis=0)[0]
X_ymax = np.max(X_pca_train, axis=0)[1]
X_xspan = np.linspace(X_xmin, X_xmax, 100)
X_yspan = np.linspace(X_ymin, X_ymax, 100)

X_mesh = np.meshgrid(X_xspan, X_yspan)
X_x, X_y = X_mesh[0].ravel(), X_mesh[1].ravel()
X_mesh_raveled = np.vstack((X_x, X_y)).T
Y_mesh = svr.predict(X_mesh_raveled).reshape(X_mesh[0].shape)



fig = plt.figure()
ax1 = fig.add_subplot(1,2,1, projection="3d")
ax2= fig.add_subplot(1,2,2, projection="3d")
ax1.scatter(*X_pca_train.T,Y_train, c=Y_train, cmap='viridis')
ax1.plot_surface(*X_mesh, Y_mesh, cmap='viridis')
ax2.scatter(*X_pca_test.T, Y_test, c=Y_test, cmap='viridis')
ax1.set_title('train pca with svr')
ax2.set_title('test pca')
fig.suptitle('pca for regression')
plt.show()




#####################
## hyperparams search
##
## meshriding C and epsilon
## with kfold
#####################


kf = KFold(n_splits=20)

C_mesh = 10**np.linspace(3,6,10)
epsilon_mesh = 10**np.linspace(-6,6, 20)
hyperparameters_mesh = np.meshgrid(C_mesh, epsilon_mesh)
C_listed, epsilon_listed = hyperparameters_mesh[0].ravel(), hyperparameters_mesh[1].ravel()

list_hyperparams = []

for j, (train_index, test_index) in enumerate(kf.split(X)):
    
    X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]


    scaler = StandardScaler()
    X_std_train = scaler.fit_transform(X_train)
    X_std_test = scaler.transform(X_test)

    pca = PCA(n_components=2, svd_solver='full')
    X_pca_train = pca.fit_transform(X_std_train)
    X_pca_test = pca.transform(X_std_test)

    best_C = C_listed[0]
    best_epsilon = epsilon_listed[0]
    best_error = float('inf')

    for i in range(len(C_listed)):
        C = C_listed[i]
        epsilon = epsilon_listed[i]
        
        svr = SVR(C=C, epsilon=epsilon)
        svr.fit(X_pca_train,Y_train)

        prediction = svr.predict(X_pca_test)
        error = mean_squared_error(Y_test, prediction)
        

        if error< best_error:
            best_error = error
            best_C = C_listed[i]
            best_epsilon = epsilon_listed[i]
    
    
    print(f'Best hyperparams for fold {j}: {int(best_C), int(best_epsilon)}')
    list_hyperparams.append((best_C, best_epsilon))
print(f'Mean best hyperparam : {np.mean(list_hyperparams, axis=0)}')



##########################
## kaggle test
##
## using k fold we found
## best hyperparams and
## now we shall test them
## (the chosen values of
## C and epsilon below)
##
## got 1.8 RMSE on kaggle
## ~4k world leaderboard
## frankly ok for such a
## naive approach
#########################



scaler = StandardScaler()
scaler.fit(X)
pca = PCA(n_components=2, svd_solver='full')
X_full_pca = pca.fit_transform(X)
svr = SVR(C=438269, epsilon=10571)
svr.fit(X_full_pca,Y)

path = '../kaggledb/test.csv'
df = pd.read_csv(path)
print(df)

for key in nondiscrete_keyvals:
    for (i,value) in enumerate(df[key]):
        if np.isnan(value):
            df.loc[i,key]=0

X_valid = np.array(df[nondiscrete_keyvals])

print(X.shape, X_valid.shape)

X_std_valid = scaler.transform(X_valid)
X_pca_valid = pca.transform(X_std_valid)

prediction = svr.predict(X_pca_valid)


output_df = pd.DataFrame({'Id':range(1461,2920), 'SalePrice':prediction})
print(output_df)
output_df.to_csv('Prediction_simple_svr.csv', index=False)



##TODO
##MUTUAL INFORMATION????
##ENSEMBLE LEARNING
##XGBOOST

