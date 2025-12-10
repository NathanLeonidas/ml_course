import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.decomposition import PCA, KernelPCA
import struct
from matplotlib import offsetbox
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap

def extract_images(file_path):
    with open(file_path, 'rb') as file:
        magic_number = struct.unpack('>I', file.read(4))[0]
        num_images = struct.unpack('>I', file.read(4))[0]
        num_rows = struct.unpack('>I', file.read(4))[0]
        num_cols = struct.unpack('>I', file.read(4))[0]
        
        images = []
        for _ in range(num_images):
            image = []
            for _ in range(num_rows):
                row = []
                for _ in range(num_cols):
                    pixel = struct.unpack('>B', file.read(1))[0]
                    row.append(pixel)
                image.append(row)
            images.append(image)
        
        return images


def flatten_images(list_img):
    flattened_list = []
    for img in list_img:
        tmp=[]
        for line in img:
            tmp+=line
        flattened_list.append(tmp)
    return flattened_list


def extract_labels(file_path):
    with open(file_path, 'rb') as file:
        magic_number = struct.unpack('>I', file.read(4))[0]  # Lire le magic number
        num_labels = struct.unpack('>I', file.read(4))[0]    # Lire le nombre de labels
        
        labels = []
        for _ in range(num_labels):
            label = struct.unpack('>B', file.read(1))[0]      # Lire un label (1 octet)
            labels.append(label)
        
        return labels


cmap = ListedColormap(plt.cm.tab10.colors)

def plot_embedding(X, images, labels, ax):
    for digit in range(10):
        ax.scatter(*X[np.array(labels) == digit].T,
                   marker=f'${digit}$',
                   s=60,
                   color=cmap(digit),
                   alpha=0.425,
                   zorder=2)
    shown_images = np.array([[0.0,0.0]])
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images)**2, 1)
        if np.min(dist) < 5e-2:
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap = plt.cm.gray_r), X[i]
                )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)


    


train_pathi = '/home/ross/Coding/ML_course/mnistdb/trainimages'
train_pathl = '/home/ross/Coding/ML_course/mnistdb/trainlabels'
test_pathi = '/home/ross/Coding/ML_course/mnistdb/testimages'
test_pathl = '/home/ross/Coding/ML_course/mnistdb/testlabels'

n_samples = 3000
train_images = extract_images(train_pathi)[:n_samples]
train_labels = extract_labels(train_pathl)[:n_samples]
test_images = extract_images(test_pathi)[:n_samples//4]
test_labels = extract_labels(test_pathl)[:n_samples//4]
train_flat_img = np.array(flatten_images(train_images))
test_flat_img = np.array(flatten_images(test_images))


scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(train_flat_img)
X_test_scaled = scaler.transform(test_flat_img)






##########################
## svm only approach
##
## turns out we need to
## reduce gamma in higher
## dimensions to avoid
## underfitting
##
## as such there should be
## 3 svms (2d, 3d, Nd)
##########################

svc = svm.SVC(kernel='rbf', decision_function_shape='ovo',C=1, gamma = 0.001)
svc.fit(X_train_scaled, train_labels)
predicted = svc.predict(X_test_scaled)
cm = confusion_matrix(test_labels, predicted, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
disp.plot()
plt.title("simple svm with gaussian kernel")
plt.show()

print('-'*20)
print('simple svm results \n')
print(classification_report(test_labels, predicted))
print('-'*20)




#########################
## tsne then svm approach
##
## note : tsne has to be
## applied on train and
## test at the same time
## bcz it is non 
## deterministic
## as such it is not
## paralleizable
##
## note: tsne separates
## better than pca and
## C can be set higher
#########################


svc = svm.SVC(kernel='rbf', decision_function_shape='ovo',C=1000, gamma = 0.0002)
embedder = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=100)



X_train_and_test = embedder.fit_transform(np.vstack((X_train_scaled, X_test_scaled)))
X_train_embedded_unprocessed = X_train_and_test[:len(train_flat_img),:]
X_test_embedded_unprocessed = X_train_and_test[len(train_flat_img):,:]
X_train_embedded = scaler.fit_transform(X_train_embedded_unprocessed)
X_test_embedded = scaler.transform(X_test_embedded_unprocessed)

svc.fit(X_train_embedded, train_labels)
predicted = svc.predict(X_test_embedded)
cm = confusion_matrix(test_labels, predicted, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)



fig = plt.figure()
subfigs = fig.subfigures(1,4, wspace=0.07)
axL = subfigs[0].subplots(1,1)
subfigs[0].suptitle('train tsne')
axL.scatter(X_train_embedded[:,0], X_train_embedded[:,1] , c=train_labels)

axR = subfigs[2].subplots(1,1)
subfigs[2].suptitle('test actual labels')
axR.scatter(X_test_embedded[:,0], X_test_embedded[:,1] , c=test_labels)

axM = subfigs[1].subplots(1,1)
subfigs[1].suptitle('all actual labels with tsne')
axM.scatter(X_train_and_test[:,0], X_train_and_test[:,1] , c=train_labels+test_labels)

axRR = subfigs[3].subplots(2,1)
subfigs[3].suptitle('predictions')
disp.plot(ax=axRR[1])
axRR[0].scatter(X_test_embedded[:,0], X_test_embedded[:,1] , c=predicted)
plt.title('tnse with svm')
plt.show()


fig = plt.figure()
ax = fig.add_subplot()
plot_embedding(X_train_embedded, train_images, train_labels, ax)
plt.title('embeddings of tsne')
plt.show()


print('-'*20)
print('tsne svm results \n')
print(classification_report(test_labels, predicted))
print('-'*20)


#######################
## svm pca approach
#######################

svc = svm.SVC(kernel='rbf', decision_function_shape='ovo',C=10000, gamma = 0.002)

pca = PCA(n_components=2)
X_train_pca_unprocessed = pca.fit_transform(X_train_scaled)
X_test_pca_unprocessed = pca.transform(X_test_scaled)
X_train_pca = scaler.fit_transform(X_train_pca_unprocessed)
X_test_pca = scaler.transform(X_test_pca_unprocessed)

svc.fit(X_train_pca, train_labels)
predicted = svc.predict(X_test_pca)
cm = confusion_matrix(test_labels, predicted, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)



fig = plt.figure()
subfigs = fig.subplots(1,2)
ax = subfigs[0]
ax2 = subfigs[1]
decision_boundary = DecisionBoundaryDisplay.from_estimator(
        svc,
        X_train_pca,
        response_method="predict",
        cmap=cmap,
        alpha=0.8,
        ax=ax,
    )
plot_embedding(X_train_pca, train_images, train_labels, ax)
disp.plot(ax=ax2)
plt.title('pca and svm')
plt.show()




print('-'*20)
print('pca svm results \n')
print(classification_report(test_labels, predicted))
print('results are genuinely awful. from the plots it is easy to see that 2 dimensions of projections are not enough')
print('-'*20)



#######################
## kernel svm pca approach
##
## same here, gaussian
## kernel requires
## a small gamma
## curse of dimension?
#######################

gamma = 0.0002
pca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
X_train_pca_unprocessed = pca.fit_transform(X_train_scaled)
X_test_pca_unprocessed = pca.transform(X_test_scaled)
X_train_pca = scaler.fit_transform(X_train_pca_unprocessed)
X_test_pca = scaler.transform(X_test_pca_unprocessed)


svc.fit(X_train_pca, train_labels)
predicted = svc.predict(X_test_pca)
cm = confusion_matrix(test_labels, predicted, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)



fig = plt.figure()
subfigs = fig.subplots(1,2)
ax = subfigs[0]
ax2 = subfigs[1]
plot_embedding(X_train_pca, train_images, train_labels, ax)
disp.plot(ax=ax2)
plt.title('KPCA and svm')
plt.show()


print('-'*20)
print('kernel pca svm results \n')
print(classification_report(test_labels, predicted))
print('results are not much better with a kernel pca, we need to have more components of projection')
print('-'*20)



#######################
## 3d svm pca approach
##
## new svm because new
## dimension (3D)
#######################


svc = svm.SVC(kernel='rbf', decision_function_shape='ovo',C=100, gamma = 0.1)

pca = PCA(n_components=3)
X_train_pca_unprocessed = pca.fit_transform(X_train_scaled)
X_test_pca_unprocessed = pca.transform(X_test_scaled)
X_train_pca = scaler.fit_transform(X_train_pca_unprocessed)
X_test_pca = scaler.transform(X_test_pca_unprocessed)

svc.fit(X_train_pca, train_labels)
predicted = svc.predict(X_test_pca)
cm = confusion_matrix(test_labels, predicted, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)



fig = plt.figure()
ax = fig.add_subplot(1,2,1, projection="3d")
ax2 = fig.add_subplot(1,2,2)

ax.scatter(*X_train_pca.T, c=train_labels)
disp.plot(ax=ax2)
plt.suptitle('3d pca and svm')
plt.show()


print('-'*20)
print('pca svm results \n')
print(classification_report(test_labels, predicted))
print('already better. Indeed:')
print('\nvariance of compontents:')
print(pca.explained_variance_)
print('\nthey each occupy this % of the variance')
print(pca.explained_variance_ratio_)
print('\nnote that it is only 15% and we need more components (heavy tail)')
print('-'*20)


#######################
## kernel 3d svm pca approach
#######################


gamma = 0.0001
pca = KernelPCA(n_components=3, kernel='rbf', gamma=gamma)
X_train_pca_unprocessed = pca.fit_transform(X_train_scaled)
X_test_pca_unprocessed = pca.transform(X_test_scaled)
X_train_pca = scaler.fit_transform(X_train_pca_unprocessed)
X_test_pca = scaler.transform(X_test_pca_unprocessed)

svc.fit(X_train_pca, train_labels)
predicted = svc.predict(X_test_pca)
cm = confusion_matrix(test_labels, predicted, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)



fig = plt.figure()
ax = fig.add_subplot(1,2,1, projection="3d")
ax2 = fig.add_subplot(1,2,2)

ax.scatter(*X_train_pca.T, c=train_labels)
disp.plot(ax=ax2)
plt.suptitle('3d KPCA and svm')
plt.show()


print('-'*20)
print('kernel pca svm results \n')
print(classification_report(test_labels, predicted))
print('still worse than a simple svm with rbf kernel though...')
print('-'*20)
