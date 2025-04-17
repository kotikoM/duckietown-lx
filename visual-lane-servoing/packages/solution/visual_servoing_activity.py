from typing import Tuple

import numpy as np
import cv2
from matplotlib.path import Path


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.
    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """
    # TODO: implement your own solution here
    # steer_matrix_left = np.random.rand(*shape)
    # ---
    return get_mask(shape, "left")

def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args: 
    shape:               The shape of the steer matrix.

    Return:
    steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """
    # TODO: implement your own solution here
    # steer_matrix_right = np.random.rand(*shape)
    # ---
    return get_mask(shape, "right")

def get_symmetry(P, W):
    Q = []
    for (x,y) in P:
        Q.append((x,W-y))
    return Q

def get_mask(shape: Tuple[int, int], side: str):
    
    EVAL_CALIB_CONST = 300 #https://github.com/duckietown/duckietown-lx-recipes/blob/mooc2022/visual-lane-servoing/assets/environment/challenges/scenarios/sampled/LF-small-loop-000/scenario.yaml#L2
    SIM_CALIB_CONST_RIGHT = 74000  #  
    SIM_CALIB_CONST_LEFT =  66700 # from calibration during simulation exercise
    
    assert side in ["right","left"], "ERROR: side not correct"
    height, width = shape
    x, y = np.mgrid[:height, :width]
    coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) 

    # Duckiebot
    # pts_left = [(1.0*height, 0.07*width), 
    #             (0.45*height, 0.45*width), 
    #             (0.45*height, 0.5*width), 
    #             (1*height, 0.5*width), 
    #             (1*height, 0.02*width)] 
    # Simulation
    pts_left = [(0.89*height, 0.03*width), 
                (0.56*height, 0.26*width), 
                (0.56*height, 1.0*width), 
                (0.89*height, 1.0*width), 
                (0.89*height, 0.0*width)] 
            
    mask_Left = Path(pts_left).contains_points(coors) * -0.0037 #(EVAL_CALIB_CONST / SIM_CALIB_CONST_LEFT) 

    # Points zone right: symmetric to zone1
    pts_right = get_symmetry(pts_left, width)
    mask_Right = Path(pts_right).contains_points(coors) * 0.0031 # (EVAL_CALIB_CONST / SIM_CALIB_CONST_RIGHT) 

    if side == "left":
        return mask_Left.reshape(height, width)
    elif side == "right":
        return mask_Right.reshape(height, width)
    

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape
    imgbgr = image.copy()

    # TODO: implement your own solution here
    # mask_left_edge = np.random.rand(h, w)
    # mask_right_edge = np.random.rand(h, w)    
        
    # Image processing
    imgrgb = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB) # RGB version for the sake of visualization
    imghsv = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2HSV)# Convert the image to HSV for any color-based filtering
    imggray = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2GRAY) #  grayscale version

    # Smooth the image using a Gaussian kernel
    sigma = 2
    img_gaussian_filter = cv2.GaussianBlur(imggray,(0,0), sigma)

    # Horizon mask
    mask_ground = np.zeros_like(imggray, dtype=np.uint8) # TODO: CHANGE ME
    mask_ground[200:,:] = 1

    # Convolve the image with the Sobel operator (filter) to compute the numerical derivatives in the x and y directions
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # Apply threshold to gradients magnitude
    threshold = 15
    mask_mag = (Gmag > threshold)

    # White and yellow HSV thresholds
    white_lower_hsv = np.array([0, 0, 130]) 
    white_upper_hsv = np.array([179, 48, 255]) 
    # yellow_lower_hsv = np.array([16, 114, 107]) 
    # yellow_upper_hsv = np.array([30, 255, 255])
    yellow_lower_hsv = np.array([20, 100, 95]) 
    yellow_upper_hsv = np.array([30, 255, 255]) 

    # Define mask for 1st image
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv) 
        
    # masks for the left- and right-halves of the image
    width = imggray.shape[1] 
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(width/2))] = 0

    # Generate a mask that identifies pixels based on the sign of their x-derivative
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    # masks with the gradient magnitude mask and color-based
    mask_left_edge = mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return Gmag * mask_left_edge, Gmag * mask_right_edge