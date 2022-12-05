import numpy as np
from numpy import median
from scipy.cluster._hierarchy import cluster_in


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
    while cluster_list.__len__() < num_cluster:
        cluster_value_dimension = find_biggest_range_cluster_list (array_in_2_d)
        cluster_to_split =  array_in_2_d.pop(cluster_value_dimension.get("cluster"))
        median_list = []
        for item in cluster_to_split:
            median_list.append(item[cluster_value_dimension.get("right/dimension")])
        t = median(median_list)
        cluster_smaller = cluster_to_split[np.where(cluster_to_split[:,cluster_value_dimension.get("right/dimension")] <= t)]
        cluster_larger = cluster_to_split[np.where(cluster_to_split[:,cluster_value_dimension.get("right/dimension")] > t)]
        array_in_2_d.append(cluster_larger)
        array_in_2_d.append(cluster_smaller)
    resultlist = []
    for cluster in array_in_2_d:
        resultlist.append(median(cluster))
    resultlist.sort()
    return resultlist


cluster_list = [np.array ([[3, 0, 0],
                           [3, 1, 0],
                           [3, 0, 5],
                           [10, 0, 10],
                           [3, 0, 7],
                           [3, 0, 0],
                           [3, 4, 2],
                           [3, 0, 6],
                           [3, 0, 0],
                           ]), (np.random.rand (2, 3) * (10)), (np.random.rand (5, 3) * (10))]


if __name__ == '__main__':



       # for cluster in cluster_list:
       #          print (np.max (cluster, axis=0) - np.min (cluster, axis=0))
       #
       #          for i in range (cluster.shape[1]):
       #              print (f'breite in Richtung {i}', np.max ((cluster[:i]) - np.min (cluster[:i])))
       #
       #      for i in range (cluster.shape[1]):
       #          max_clustrer = -np.inf
       #          min_clustrer = np.inf
       #          for point in cluster:
       #              if point[i] > max_clustrer:
       #                  max_clustrer = point[i]
       #              if point[i] < min_clustrer:
       #                  min_clustrer = point[i]
       #          print (i, max_clustrer - min_clustrer)

            # finde von diesen breiten die maximale für diesen cluster und merke sich breite und richtung
            # finde dden cluster mit maximaler breite und merke sich cluster und richtung
            # teile den cluster mit maximaler breite in richtung mit maximaler breite
            # hilfriech zum teilen

            # dummy_cluster = np.random.rand (10, 3)  # cluser mit max breite ersetzen
            # richhung = 0
            # t = 0.5  # grenze zum teieln - muss durch median ersetzt werden
            #
            # cluster_smaller = dummy_cluster[np.where (dummy_cluster <= t)]
            # cluster_bigger = dummy_cluster[np.where (dummy_cluster > t)]

    mediancutList = find_median_cut_in_array_with_num_of_cluster(cluster_list, 10)

    print("mediancutList ", mediancutList)

