import cv2
import numpy as np
import math
import mpmath
import scipy.linalg

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, fov=math.pi/2):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """

        ### TODO ###
        resolution = 512
        translate = 1
        theta = theta*math.pi/180   # degree to radian
        focal_length = float(resolution/2*mpmath.cot(fov/2)) 
     
        intrinsic_mat = np.array([[focal_length, 0, resolution/2], [0, focal_length, resolution/2], [0, 0, 1]])
        trasformation_mat = np.array([[1, 0, 0, 0], 
                                        [0, math.cos(theta), -math.sin(theta), translate], 
                                        [0, math.sin(theta), math.cos(theta), 0],
                                        [0, 0, 0, 1]])

       
        # print(intrinsic_mat)
        # print(trasformation_mat)
        new_pixels = []
        for i in range(4):
            
            new_point = np.append(np.array(points[i]).transpose(), 1)                   # homoginious 
            new_point = np.dot(scipy.linalg.inv(intrinsic_mat), new_point)              # image plane to 3D coordinate
            new_point = new_point*(-2.5)                                                # *Z (Z = -2.5)
            
            new_point = np.append(new_point, 1)                                         # homoginious 
            new_point = np.dot(trasformation_mat, new_point)
            new_point = np.dot(np.array([[1,0,0,0], [0,1,0,0],[0,0,1,0]]), new_point)   
            new_point = np.dot(intrinsic_mat, new_point)                                # 3D coordinate to image plane
            new_pixels.append(np.int32(new_point[0:2]/new_point[2]))

        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    x_ang = -90
    

    front_rgb = "./task1_data/front_view1.png"
    top_rgb = "./task1_data/top_view1.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=x_ang)
    projection.show_image(new_pixels)
