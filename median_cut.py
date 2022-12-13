import cv2
import numpy as np
from numpy import median, mean
import matplotlib.pyplot as plt

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
    while array_in_2_d.__len__ () < num_cluster:
        cluster_value_dimension = find_biggest_range_cluster_list (array_in_2_d)
        cluster_to_split = array_in_2_d.pop (cluster_value_dimension.get ("cluster"))
        median_list = []
        for item in cluster_to_split:
            median_list.append (item[cluster_value_dimension.get ("right/dimension")])
        t = median (median_list)
        cluster_smaller = cluster_to_split[
            np.where (cluster_to_split[:, cluster_value_dimension.get ("right/dimension")] <= t)]
        cluster_larger = cluster_to_split[
            np.where (cluster_to_split[:, cluster_value_dimension.get ("right/dimension")] > t)]
        array_in_2_d.append (cluster_larger)
        array_in_2_d.append (cluster_smaller)
    resultlist = []
    for cluster in array_in_2_d:
        mittelwerte = []
        for richtung in range (cluster[0].shape[0]):
            mittelwerte.append (int (round (np.mean(cluster[:, richtung]))))
        resultlist.append (mittelwerte)
        res = np.array (resultlist).reshape (resultlist.__len__ (), 3)
    res = sorted (res, key=lambda x: x.sum ())
    return res


def find_new_color(item, color_list):
    x = 0 ;
    while x <= color_list.__len__ () - 1:
        first = int (round (color_list[x].sum ()))
        second = int (round (color_list[x + 1].sum ()))
        diff_one = item.sum () - first
        diff_two = second - item.sum ()
        if (diff_one < diff_two):
            return color_list[x]
        if (x == color_list.__len__ () - 2):
            return color_list[x + 1]
        x = x + 1
    return color_list[0]



def median_cut(img_path, path_to_save, color_count):
    img = cv2.imread (img_path)
    imgList = [np.array (img).reshape (img.shape[0] * img.shape[1], img.shape[2])]
    mediancut_list = find_median_cut_in_array_with_num_of_cluster (imgList, color_count)
    # correct color range
    result = np.eye (img.shape[0] * img.shape[1], img.shape[2])
    next_img = np.array (img).reshape (img.shape[0] * img.shape[1], img.shape[2])
    for first_dim, arr in enumerate (next_img):
        result[first_dim] = find_new_color (arr, mediancut_list)
    # generate Integer
    int_image = result.astype(int)

    # correct failures
    int_image[np.where (int_image == np.max (int_image))] = 0
    int_image[np.where (int_image == np.min (int_image))] = 0
    print ("resultImage.shape bofor .reshape()", int_image.shape)
    int_image2 = int_image.reshape (img.shape[0], img.shape[1], img.shape[2])
    print ("resultImage.shape ", int_image2.shape)
    print ("mediancut_list", mediancut_list)
    print ("result_image ", int_image2)
    # save image
    image = int_image2.astype(np.uint8)
    cv2.imshow ('matrix', image)
    cv2.waitKey (0)
    cv2.imwrite (path_to_save, image)
    return image

if __name__ == '__main__':

    # cluster_list2 = np.array(img).reshape(-1, img.shape[2])
    # img = np.array(tmp_img)

    # print ("<imgList ", imgList)
    # print("cluster_list.pop().shape ", cluster_list.pop().shape)
    # print("imgList.pop().shape ", imgList.pop().shape)
    # mediancut_list2 = mediancut_list[:][:]
    # imgList = mediancut_list


    #median_cut('Baboon.png', img_to_save2, 2)
    # print ("mediancut_list.pop().pop().dtype", mediancut_list.pop().pop().dtype)
    result_image = median_cut('Baboon.png', 'Baboon_2_color.png', 16)
    plt.imshow(result_image)
    plt.show()
    plt.close()