import cv2
import numpy as np


def Corner_Response(gdx2, gdy2, gdxy, alpha):
    r = np.array(gdx2)

    r[:,:] = (gdx2[:,:] * gdy2[:,:]) - alpha * ((gdx2 + gdy2)*(gdx2 + gdy2))
    
    return r


def Threshold(r, threshold):
    corners = np.zeros(r.shape)
    for i in range(1, r.shape[0]-1):
        for j in range(1, r.shape[1]-1):
            if(r[i,j] > threshold):
                corners[i,j] = r[i,j]

    return corners                    



def Local_Maxima(corner):
    corner_max = np.zeros(corner.shape)
    for i in range(1, corner.shape[0] - 1):
        for j in range(1, corner.shape[1] - 1):
            window = corner[i-1:i+2, j-1:j+2]
            if(corner[i,j] == np.max(window)):
                corner_max[i,j] = 255

    return corner_max
            
def Sobel(Image):
    x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1,0,1]])
    y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1,2,1]])
    dx2 = np.array(Image)
    dy2 = np.array(Image)
    dxy = np.array(Image)
    print(x_filter.shape)

    for i in range(1,Image.shape[0]-1):
        for j in range(1, Image.shape[1]-1):
            window = Image[i-1:i+2, j-1:j+2]
            tx = np.sum(window[:,:] * x_filter[:,:])
            ty = np.sum(window[:,:] * y_filter[:,:])

            dx2[i,j] = tx * tx
            dy2[i,j] = ty * ty
            dxy[i,j] = tx * ty

    return dx2, dy2, dxy

def Gaussian(Image):
    gaussian_filter = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]], dtype=float)
    gaussian_filter[:,:] = gaussian_filter[:,:] / float(273)
    print(gaussian_filter)

    gImg = np.array(Image)
    
    tx2, ty2, txy = 0, 0, 0
    for i in range(2, Image.shape[0] -2):
        for j in range(2, Image.shape[1] - 2):
            window = Image[i-2:i+3, j-2:j+3]

            gImg[i,j] = int(np.sum(window[:,:] * gaussian_filter[:,:]) )

    return gImg

def Composition(img, corners):
    composed_img = np.array(img)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if(not(corners[i,j] == 255)):
                composed_img[i,j, 0] = 0
                composed_img[i,j, 1] = 0
                composed_img[i,j, 2] = 255

    return composed_img

def main():
    alpha = 0.06 # 0.04 ~ 0.06
    threshold = 245

    color_img = cv2.imread("./city.png",cv2.IMREAD_COLOR)
    print(color_img.shape)
    img = cv2.imread("./city.png",cv2.IMREAD_GRAYSCALE)
    img = Gaussian(img)
    dx2, dy2, dxy = Sobel(img)
    # gdx2, gdy2, gdxy = Gaussian(dx2, dy2, dxy)
    r = Corner_Response(dx2, dy2, dxy, alpha)
    thresholded_corners = Threshold(r, threshold)
    corners = Local_Maxima(thresholded_corners)

    cv2.imwrite('city_r.png',r)
    cv2.imwrite('city_thresholded_corners.png',thresholded_corners)
    cv2.imwrite('city_corners.png',corners)

    result = Composition(color_img, corners)
    cv2.imwrite('city_result.png',result,)



if __name__=="__main__":
    main()
    
