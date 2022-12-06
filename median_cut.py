import cv2
import numpy as np
from numpy import median, mean

img_path = 'CT.png'
img_to_save = 'CT_new.png'
img_to_save2 = 'CT-new_2.png'

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
    # todo erstelle clusterList und append(array_in_2_d)
    while cluster_list.__len__() < num_cluster:
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
    result_list2 = []
    for cluster in array_in_2_d:
        mittelwerte = []
        for richtung in range(cluster[0].shape[0]):
            mittelwerte.append(np.mean(cluster[:, richtung]))
        resultlist.append(mittelwerte)
        result_list2.append(np.mean(cluster, axis=0))

    return resultlist, result_list2




cluster_list = [np.array ([[3, 0, 0],
                           [3, 1, 0],
                           [3, 0, 5],
                           [10, 0, 10],
                           [3, 0, 7],
                           [3, 0, 0],
                           [3, 4, 2],
                           [3, 0, 6],
                           [3, 0, 0],
                           ]), (np.random.rand (7, 3) * (10))]


if __name__ == '__main__':

    img = cv2.imread('CT.png')
    cluster_list2 = np.array(img).reshape(-1, img.shape[2])

    mediancut_list, median_array_list = find_median_cut_in_array_with_num_of_cluster(cluster_list2, 10)

    print("mediancutList ", mediancut_list)
    print("median_array_list ", median_array_list)

