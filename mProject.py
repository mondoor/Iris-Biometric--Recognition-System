import math
import numpy as np
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt
import cv2


def gammaCorrection(image, gamma):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    res = cv2.LUT(image, table)
    return res


'''
this method preform the process to extract the iris from the image 

input: image

output: iris 
'''


def encode_photo(image):
    # make a safe copy of the image
    newImage = image.copy()
    # correct the gamma to pupil detection with 0.3 gamma
    image = gammaCorrection(image, 0.3)
    # get pupil location
    x, y, r = findPupil(image)
    # get the iris location
    x_iris, y_iris, r_iris = findIris(image)
    # correct the gamma for extract iris from the image
    newImage = gammaCorrection(newImage, 0.5)
    # extract iris
    iris = extractIrirs(newImage, x, y, r, x_iris, y_iris, r_iris)
    # return iris

    return iris


'''
this method prepare the image for pupil extraction

input: image 

output: modified image 
'''


def preprocessP(image):
    # apply median blur
    image = cv2.medianBlur(image, 41)
    # apply thresholding to make the pupil clear
    _, thimage = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    # display the image to test
    plt.imshow(thimage, cmap='gray')
    plt.title("binary image")
    plt.show()
    # return the image
    return thimage


'''
this method prepare the image for pupil extraction

input: image 

output: modified image 
'''


def preprocessI(image):
    # apply median blur
    image = cv2.medianBlur(image, 41)
    # apply thresholding to make the iris clear
    _, th = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    # display the image to test
    plt.imshow(th, cmap='gray')
    plt.title("binary image")
    plt.show()
    # return the image
    return th


'''
this method  for pupil detection

input: image 

output: pupil concordats of the canter and the radius 
'''


def findPupil(img):
    # process to find pupil
    image0 = preprocessP(img)
    # detect circlesList
    circlesList = cv2.HoughCircles(image0, cv2.HOUGH_GRADIENT, 1, 700,
                                   param1=70, param2=0.85, minRadius=50, maxRadius=400)

    if circlesList is not None:  # in case circlesList detected
        circlesList = np.uint16(np.around(circlesList))
        # drawing detected circle
        center = (circlesList[0, 0][0], circlesList[0, 0][1])
        # circle center
        cv2.circle(img, center, 1, (0, 255, 0), 3)
        # circle outline
        radius = circlesList[0, 0][2]
        cv2.circle(img, center, radius, (0, 255, 0), 3)
        plt.imshow(img, cmap='gray')
        plt.title("Pupil")
        plt.show()
        # return coordinates (x,y) and radius
        return circlesList[0, 0][0], circlesList[0, 0][1], circlesList[0, 0][2]

    else:
        # in case nothing detected
        return 0, 0, 0


'''
this method  for outer iris detection

input: image 

output: iris concordats of the canter and the radius 
'''


def findIris(img):
    # process to find pupil
    image0 = preprocessI(img)
    # detect circlesList
    circles = cv2.HoughCircles(image0, cv2.HOUGH_GRADIENT, 1, 700,
                               param1=70, param2=0.85, minRadius=100, maxRadius=500)

    if circles is not None:
        # drawing detected circle
        circles = np.uint16(np.around(circles))
        # drawing detected circle
        center = (circles[0, 0][0], circles[0, 0][1])
        # circle center
        cv2.circle(img, center, 1, (0, 255, 0), 3)
        # circle outline
        radius = circles[0, 0][2]
        cv2.circle(img, center, radius, (0, 255, 0), 3)
        plt.imshow(img, cmap='gray')
        plt.title("Iris")
        plt.show()
        # return coordinates (x,y) and radius
        return circles[0, 0][0], circles[0, 0][1], circles[0, 0][2]
    else:
        return 0, 0, 0


'''
this method for iris extraction

input: image , pupil  concordats(xp,yp,rp) , iris  concordats(xi,yi,ri), size of rectangular output image 

output: iris image
'''


def extractIrirs(img, xp, yp, rp, xi, yi, ri, phase_width=300, iris_width=150):
    # create empty iris to be filled
    iris = np.zeros((iris_width, phase_width))
    # create vectors lines  of phase values
    theta = np.linspace(0, 2 * np.pi, phase_width)

    # for each phase in the iris, calculating coordinates and pixel value
    for i in range(phase_width):
        # calculate pixels coordinates of start and end of the phase
        start = startEndPixels(rp, xp, yp, theta[i])
        end = startEndPixels(ri, xi, yi, theta[i])

        # find coordinates of pixels between the start and ens pixels
        xspace = np.linspace(start[0], end[0], iris_width)
        yspace = np.linspace(start[1], end[1], iris_width)

        # set value of each pixel in phase line
        iris[:, i] = [255 - img[int(y), int(x)]
                      if 0 <= int(x) < img.shape[1] and 0 <= int(y) < img.shape[0]
                      else 0
                      for x, y in zip(xspace, yspace)]

    # display iris
    plt.imshow(iris, cmap='gray')
    plt.title("Iris")
    plt.show()

    return iris


def startEndPixels(r, x0, y0, theta):
    x = int(x0 + r * math.cos(theta))
    y = int(y0 + r * math.sin(theta))
    return x, y


'''
Calculates gabor wavelet.
    :param rho: Radius of the input coordinates
    :param phi: Angle of the input coordinates
    :param w: Gabor wavelet parameter (see the formula)
    :param theta0: Gabor wavelet parameter (see the formula)
    :param r0: Gabor wavelet parameter (see the formula)
    :param alpha: Gabor wavelet parameter (see the formula)
    :param beta: Gabor wavelet parameter (see the formula)
    :return: Gabor wavelet value at (rho, phi)

'''


def gabor(rho, phi, w, theta0, r0, alpha, beta):
    return np.exp(-w * 1j * (theta0 - phi)) * np.exp(-(rho - r0) ** 2 / alpha ** 2) * \
           np.exp(-(phi - theta0) ** 2 / beta ** 2)


'''
Uses gabor wavelets to extract iris features.
    :param img: Image of an iris
    :param w: w parameter of Gabor wavelets
    :param alpha: alpha parameter of Gabor wavelets
    :param beta: beta parameter of Gabor wavelets
    :return: Transformed image of the iris (real and imaginary)
    :rtype: tuple (ndarray, ndarray)

'''


def gabor_convolve(img, w, alpha, beta):
    """
    """
    rho = np.array([np.linspace(0, 1, img.shape[0]) for i in range(img.shape[1])]).T
    x = np.linspace(0, 1, img.shape[0])
    y = np.linspace(-np.pi, np.pi, img.shape[1])
    xx, yy = np.meshgrid(x, y)
    return rho * img * np.real(gabor(xx, yy, w, 0, 0.5, alpha, beta).T), rho * img * np.imag(
        gabor(xx, yy, w, 0, 0.5, alpha, beta).T)


'''
Encodes the straightened representation of an iris with gabor wavelets.
    :param img: Image of an iris
    :param dr: Width of image patches producing one feature
    :param dtheta: Length of image patches producing one feature
    :param alpha: Gabor wavelets modifier (beta parameter of Gabor wavelets becomes inverse of this number)
    :return: Iris code and its mask
    :rtype: tuple (ndarray, ndarray)
'''


def iris_encode(img, dr=15, dtheta=15, alpha=0.4):
    mask = view_as_blocks(np.logical_and(100 < img, img < 230), (dr, dtheta))
    norm_iris = (img - img.mean()) / img.std()
    patches = view_as_blocks(norm_iris, (dr, dtheta))
    code = np.zeros((patches.shape[0] * 3, patches.shape[1] * 2))
    code_mask = np.zeros((patches.shape[0] * 3, patches.shape[1] * 2))
    for i, row in enumerate(patches):
        for j, p in enumerate(row):
            for k, w in enumerate([8, 16, 32]):
                wavelet = gabor_convolve(p, w, alpha, 1 / alpha)
                code[3 * i + k, 2 * j] = np.sum(wavelet[0])
                code[3 * i + k, 2 * j + 1] = np.sum(wavelet[1])
                code_mask[3 * i + k, 2 * j] = code_mask[3 * i + k, 2 * j + 1] = \
                    1 if mask[i, j].sum() > dr * dtheta * 3 / 4 else 0
    code[code >= 0] = 1
    code[code < 0] = 0
    return code, code_mask


'''
this method is Hamming Distance  to get deference
'''


def compare_codes(a, b, mask_a, mask_b):
    return np.sum(np.remainder(a + b, 2) * mask_a * mask_b) / np.sum(mask_a * mask_b)


'''
run the program
'''


def main():
    # read images
    image = cv2.imread('2.jpg')
    image2 = cv2.imread('2 (2).jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    iris1 = encode_photo(image)
    iris2 = encode_photo(image2)

    # correct images
    iris1 = cv2.normalize(iris1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    iris2 = cv2.normalize(iris2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    code, mask = iris_encode(iris1)
    code2, mask2 = iris_encode(iris2)

    if compare_codes(code, code2, mask, mask2) <= 0.48:
        print(compare_codes(code, code2, mask, mask2))
        print("Iris Match found")
        # show the images
        plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original')
        plt.subplot(122), plt.imshow(image2, cmap='gray'), plt.title('To be matches ')
        plt.suptitle('Iris Biometric sample did match', fontsize=20)
        plt.show()

    else:
        print(compare_codes(code, code2, mask, mask2))
        print("Iris match not found")
        # show the images
        plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original')
        plt.subplot(122), plt.imshow(image2, cmap='gray'), plt.title('To be matches')
        plt.suptitle('Iris Biometric sample did not match!!!', fontsize=20)
        plt.show()


main()
