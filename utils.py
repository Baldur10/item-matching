import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def query_crop(query_path, txt_path, save_path):
    query_img = cv2.imread(query_path)
    query_img = query_img[:,:,::-1] #bgr2rgb
    txt = np.loadtxt(txt_path)     #load the coordinates of the bounding box
    crop = query_img[int(txt[1]):int(txt[1] + txt[3]), int(txt[0]):int(txt[0] + txt[2]), :] #crop the instance region
    cv2.imwrite(save_path, crop[:,:,::-1])  #save the cropped region
    return crop

def visualization(retrived, query):
    plt.subplot(5, 5, 1)
    plt.title('query')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb_rgb)
    for i in range(10):
        img_path = './datasets_4186/gallery_4186/' + retrived[i][0]
        # print(img_path)
        img = cv2.imread(img_path)
        img_rgb = img[:,:,::-1]
        img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
        plt.subplot(5, 5, i+2)
        plt.tight_layout(h_pad=2)
        plt.title(round(retrived[i][1],2))
        plt.imshow(img_rgb)
    plt.show()

def visualization_hist(retrived, query):
    plt.subplot(5, 5, 1)
    plt.title('query')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb_rgb)
    for i in range(10):
        # print(retrived[i][0][0:retrived[i][0].find('_')])
        img_path = './datasets_4186/gallery_4186/' + retrived[i][0][0:retrived[i][0].find('_')] + '.jpg'
        # print(retrived[i][0][0:retrived[i][0].find('_')])
        # print(img_path)
        img = cv2.imread(img_path)
        img_rgb = img[:,:,::-1]
        img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_CUBIC)
        plt.subplot(5, 5, i+2)
        plt.tight_layout(h_pad=2)
        plt.title(round(retrived[i][1],2))
        plt.imshow(img_rgb)
    plt.show()

def retrival_idx(query_path, gallery_dir, similarityType = 'cosine'):
    query_feat = np.load(query_path, allow_pickle=True)
    # print(query_feat)
    
    dict_values = {}
    # print(dict_values)
    
    for gallery_file in os.listdir(gallery_dir):
        gallery_feat = np.load(os.path.join(gallery_dir, gallery_file),  allow_pickle=True)
        gallery_idx = gallery_file.split('.')[0] + '.jpg'

        if similarityType == 'cosine':
            sim = similarity_cosine(query_feat, gallery_feat)
        elif similarityType == 'pearson':
            sim = similarity_pearson(query_feat, gallery_feat)
        elif similarityType == 'euclidean':
            sim = similarity_distance(query_feat, gallery_feat)
        # print(sim)
        dict_values[gallery_idx] = sim
    sorted_dict = sorted(dict_values.items(), key=lambda item: item[1]) # Sort the similarity score
    if similarityType == 'euclidean':
        best_ten = sorted_dict[:10]
    else:
        best_ten = sorted_dict[-10:] # Get the best five retrived images
    # print(best_ten)
    return best_ten

def similarity_cosine(query_feat, gallery_feat):
    sim = cosine_similarity(query_feat, gallery_feat)
    sim = np.squeeze(sim)
    return sim
    
def similarity_pearson(query_feat, gallery_feat):
    sim = (np.corrcoef(query_feat, gallery_feat)[0,1])**2
    sim = np.squeeze(sim)
    return sim

def similarity_distance(query_feat, gallery_feat):
    sim = np.linalg.norm(query_feat-gallery_feat)
    sim = np.squeeze(sim)
    return sim
