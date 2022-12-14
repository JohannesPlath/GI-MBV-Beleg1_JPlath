import cv2
import matplotlib.pyplot as plt
import numpy as np



def fold_filter_3x3(image, filter):
    shape_of_image = np.array(image).shape
    filter = filter.reshape(filter.shape[0] * filter.shape[1])
    filter_sum = filter.sum()
    result_image = np.zeros(shape_of_image)
    testlist = []
    median_list = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for colour in range(image.shape[2]):
                if 0 < x < (image.shape[0] - 1) and 0 < y < (image.shape[1] - 1):
                    median_list.append(image[x-1][y-1][colour] * filter[0])
                    median_list.append(image[x][y-1][colour] * filter[1])
                    median_list.append(image[x+1][y-1][colour] * filter[2])
                    median_list.append(image[x-1][y][colour]  * filter[3])
                    median_list.append(image[x][y][colour] * filter[4])
                    median_list.append(image[x+1][y][colour] * filter[5])
                    median_list.append(image[x-1][y+1][colour] * filter[6])
                    median_list.append(image[x][y+1][colour] * filter[7])
                    median_list.append(image[x+1][y+1][colour] * filter[8])
                    searched_value = np.array(median_list).sum() / filter_sum
                    #testlist.append([image[x][y][colour] , filter[4], searched_value])
                    result_image[x][y][colour] = np.array(searched_value).astype(int)
                    median_list = []
    result_image = np.array(result_image).astype(np.uint8)
    return result_image



if __name__ == '__main__':



    filter_3x3_gaus = np.array([[1,2,1],
                      [2,4,2],
                      [1,2,1]])

    filter_3x3 = np.array([[1,2,1],
                           [2,4,2],
                           [1,2,1]])
    print("filter.shape ", filter_3x3.shape)

    # test_image = np.zeros((5, 5, 1))
    # test_image[1][1][:] = 1
    # test_image[2][2][:] = 2
    # test_image[3][3][:] = 3
    # test_image[4][4][:] = 4
    # test_image[0][0][:] = 0
    # print ("test_image --->", test_image)

    # plt.imshow(result_img)
    # plt.show()
    # plt.close()
    # cv2.imshow ('matrix', result_img)
    # cv2.waitKey (0)
    # cv2.imwrite ("Baboon3x3gaus.png", result_img)

    img_path = 'Baboon.png'
    image = cv2.cvtColor (cv2.imread (img_path), cv2.COLOR_BGR2RGB)

    filter_3x3_gaus = np.array([[1,2,1],
                                [2,4,2],
                                [1,2,1]])

    filter_3x3 = np.array([[1,2,1],
                           [2,4,2],
                           [1,2,1]])

    image = cv2.cvtColor (cv2.imread (img_path), cv2.COLOR_BGR2RGB)
    result_img = fold_filter_3x3(image, filter_3x3_gaus)
    from PIL import Image
    im = Image.fromarray(result_img)
    im.save("Baboon3x3GausAsPIL.png")


