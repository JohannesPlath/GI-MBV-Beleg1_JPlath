import cv2
import numpy as np
from numpy import median, mean

img_path = 'CT.png'
img_to_save = 'to_save.png'
img_to_save2 = 'to_save_2.png'

image = cv2.cvtColor (cv2.imread (img_path), cv2.COLOR_BGR2RGB)



def find_biggest_range_cluster_list(cluster_list):
    cluster_result_list = []
    counter = 0
    for cluster in cluster_list:
        tmp_cluster_list = []
        for i in range (cluster.shape[1]):
            tmp_cluster_list.append (np.max (cluster[:, i]) - np.min (cluster[:, i]))
        cluster_result_list.append (tmp_cluster_list)
        counter = counter + 1
    max_found_value = 0
    searched_cluster = 0
    searched_right = -1
    for item in enumerate (cluster_result_list):
        tmp_value = max (item[1])
        if tmp_value > max_found_value:
            max_found_value = tmp_value
            searched_cluster = item[0]
            searched_right = item[1].index (tmp_value)
    return {"cluster": searched_cluster, "value": max_found_value, "right/dimension": searched_right}


def find_median_cut_in_array_with_num_of_cluster(array_in_2_d, num_cluster):
    while array_in_2_d.__len__() < num_cluster:
        cluster_value_dimension = find_biggest_range_cluster_list (array_in_2_d)
        cluster_to_split = array_in_2_d.pop(cluster_value_dimension.get("cluster"))
        median_list = []
        for item in cluster_to_split:
            median_list.append(item[cluster_value_dimension.get("right/dimension")])
        t = median(median_list)
        cluster_smaller = cluster_to_split[np.where(cluster_to_split[:, cluster_value_dimension.get("right/dimension")] <= t)]
        cluster_larger = cluster_to_split[np.where(cluster_to_split[:, cluster_value_dimension.get("right/dimension")] > t)]
        array_in_2_d.append(cluster_larger)
        array_in_2_d.append(cluster_smaller)
    resultlist = []
    for cluster in array_in_2_d:
        mittelwerte = []
        for richtung in range(cluster[0].shape[0]):
            mittelwerte.append(int(round(np.mean(cluster[:, richtung]))))
        resultlist.append(mittelwerte)
        res =  np.array(resultlist).reshape(resultlist.__len__(), 3)
    res = sorted(res, key=lambda x:x.sum())
    return res


def find_new_color(item, clusterlist):
    x = 0;
    while x < clusterlist.__len__() -1 :
        first = int(round(clusterlist[x].sum()))
        second =int(round(clusterlist[x+1].sum()))
        diff_one = item.sum() - first
        diff_two = second - item.sum()
        if (diff_one < diff_two ):
            return clusterlist[x]
        x = x +1
    return clusterlist[0]






if __name__ == '__main__':


    img = cv2.imread('Baboon.png')
    #cluster_list2 = np.array(img).reshape(-1, img.shape[2])
    #img = np.array(tmp_img)
    imgList = [np.array(img).reshape(img.shape[0] * img.shape[1], img.shape[2])]
    print ("<imgList ", imgList)
    # print("cluster_list.pop().shape ", cluster_list.pop().shape)
    # print("imgList.pop().shape ", imgList.pop().shape)



    mediancut_list = find_median_cut_in_array_with_num_of_cluster(imgList, 16)


    #mediancut_list2 = mediancut_list[:][:]


    #imgList = mediancut_list

    result = np.eye(img.shape[0] * img.shape[1], img.shape[2])
    next_img = np.array(img).reshape(img.shape[0] * img.shape[1], img.shape[2])
    for first_dim, arr in enumerate(next_img):
           result[first_dim] = find_new_color(arr, mediancut_list )

    resultImage = np.array(result)
    resultImage[np.where(resultImage==np.max(resultImage))] = 0
    resultImage[np.where(resultImage==np.min(resultImage))] = 0
    print("resultImage.shape ",resultImage.shape )
    resultImage = resultImage.reshape(img.shape[0] , img.shape[1], img.shape[2])
    print("resultImage.shape ", resultImage.shape)
    print ("mediancut_list", mediancut_list)
    print("result ", result)

    cv2.imshow('matrix', resultImage)
    cv2.waitKey(0)
    cv2.imwrite(img_to_save2, resultImage)



    #print ("mediancut_list.pop().pop().dtype", mediancut_list.pop().pop().dtype)


