import cv2
import numpy as np
import scim as scim
import scipy as sp
import scipy.ndimage as scim

def get_binom_vector(n):
    return np.array([sp.special.binom(n -1, i) for i in range(n)])


def generate_binomial_filter(size):
    vec1 = get_binom_vector(size).reshape(-1,1)
    vec2 = get_binom_vector(size).reshape(1,-1)
    #for i in range(size -1):
    filter =  vec1.dot(vec2)
    filter = filter/np.sum(filter)
    return filter

def print_image_in_notebook(result_image):
    import matplotlib.pyplot as plt
    plt.imshow(result_image)
    plt.show()
    plt.close()


def fold_the_pic(image, weight):
    result_image = []
    image = np.array(image)
    for i in range(image.shape[2]):
        result_image.append(scim.correlate(image[:, :, i], weight, output=int, mode='mirror'))
    return np.moveaxis(result_image, 0, 2)


if __name__ == '__main__':

    filter_3x3 = np.array([[1,2,1],
                           [2,4,2],
                           [1,2,1]]) /16

    kern = generate_binomial_filter(3)
    image = cv2.cvtColor (cv2.imread ("Lena.png"), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor (cv2.imread ("Lena.png"), cv2.COLOR_BGR2RGB)
    result_image = fold_the_pic(image, filter_3x3)
    print_image_in_notebook(result_image)
    print(kern)
    weights = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
    print(weights)

    print("image[3][3][:]", image[3][3][:])
    print("image.shape", image.shape)
    print("image.type", image.dtype)
    new_image = fold_the_pic(image, kern)

    print_image_in_notebook(image)
    print_image_in_notebook(new_image)