import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, KernelPCA
from sklearn.preprocessing import StandardScaler





#importing the data
import struct

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

pathi = '/home/ross/Coding/ML_course/mnistdb/trainimages'
pathl = '/home/ross/Coding/ML_course/mnistdb/trainlabels'
images = extract_images(pathi)
flat_img = flatten_images(images)
labels = extract_labels(pathl)

#tests unitaires
print(images[0],images[1])
print(labels[0],labels[1])
print(flat_img[0], flat_img[1])
plt.imshow(images[0], cmap='gray')
plt.title(f'Label: {labels[0]}')
plt.savefig('image_mnist.png')  # Sauvegarde l'image dans un fichier
plt.close()


#testing centering and SVD Calculations
x = flat_img #list of vectors in Rd

X = np.transpose(np.array(x))
moy = np.array([np.sum(X,axis=1)/len(x)])
X_moy = np.matmul(moy.T, np.array([[1]*len(x)]))
X_centered = X - X_moy

print(moy, np.shape(moy))
print(X_centered[:,1])

#PCA
svd = TruncatedSVD(n_components=2, n_iter=10, random_state=5)
U = svd.fit_transform(np.transpose(X_centered))

print(svd.singular_values_)
plt.scatter(U[:,0],U[:,1], c=labels)
plt.title('principal components of 28*28 images')
plt.savefig('pca_result.png')
plt.close()


svd3D = TruncatedSVD(n_components=3, n_iter=10, random_state=5)
U = svd3D.fit_transform(np.transpose(X_centered))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
nm =500
ax.scatter(U[:nm,0],U[:nm,1],U[:nm,2], c=labels[:nm])
plt.savefig('3D_pca_result')
plt.show()
plt.close()


#kernelPCA
max_samples = 5000
kpca = KernelPCA(n_components=9, kernel='cosine')
data = np.transpose(X_centered[:,:max_samples])


scaler = StandardScaler()
X_std = scaler.fit_transform(data)
print(np.shape(data), np.shape(X_std))

U = kpca.fit_transform(X_std)
print(np.shape(X_centered[:,:max_samples]), np.shape(U))

plt.scatter(U[:,0],U[:,1], c=labels[:max_samples])
plt.title('kernel principal components of 28*28 images')
plt.savefig('k_pca_result.png')
plt.close()
print(kpca.eigenvalues_, np.sum(kpca.eigenvalues_), np.sum(kpca.eigenvalues_[:2]))




