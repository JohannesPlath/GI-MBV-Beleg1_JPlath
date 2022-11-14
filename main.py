import cv2
import matplotlib.pyplot as plt
import numpy as np

# Sources

img_path = 'CT.png'
img_to_save = 'CT_new.png'
img_to_save2 = 'CT-new_2.png'

image = cv2.cvtColor (cv2.imread (img_path), cv2.COLOR_BGR2RGB)


# image = plt.imread(img_path)

def get_one_chanal_image(image):
    transp_image = np.array (image).T
    flatted_transp_image = transp_image[:][:1]
    flatted_image = flatted_transp_image.T
    print ('flatted_image', flatted_image)
    print ('flatted_image.shape', flatted_image.shape)
    return flatted_image


def print_dim_min_max_and_return_max_min(image):
    print ("image.shape", image.shape)
    max_vlaue = np.amax (image)
    min_value = np.amin (image)
    print ("max_vlaue: ", max_vlaue)
    print ("min_value: ", min_value)


def crop(img):
    y_nonzero, x_nonzero, _ = np.nonzero(img)
    return img[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]




if __name__ == '__main__':
    print ('PyCharm')
    print ('image.shape: ', image.shape)
    # print('image: ', image)

    one_ch_im = get_one_chanal_image (image)

    print (one_ch_im.shape)
    print ('image[255][255][:]: ', image[255][255][:])

    # plt.close()
    # plt.imshow(one_ch_im, cmap='gray')
    # plt.show()
    print_dim_min_max_and_return_max_min(one_ch_im)
    img = cv2.imread('CT.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    print('x,y,w,h:', x,y,w,h)
    crop = img[y:y+h,x:x+w]
    cv2.imwrite('CT_new.png',crop)
     cropped_img = crop(one_ch_im)
    print(  'cropped_img.shape: ', cropped_img.shape)
    cv2.imwrite('CT_new.png',cropped_img)
    plt.close()
    plt.imshow(cropped_img, cmap='gray')
    plt.show()
