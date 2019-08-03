## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chess_calibration1.jpg "Undistorted"
[image2]: ./output_images/undistorted_image.jpg "Undistorted Road Transformed"
[image3]: ./output_images/unwarped_image.jpg "Unwraped Road Transformed"
[image4]: ./output_images/sobel_thresholds.jpg "Sobel Thresholds"
[image5]: ./output_images/color_thresholds.jpg "Color Thresholds"
[image6]: ./output_images/sobels.jpg "Sobel Thresholds"
[image7]: ./output_images/pipeline.jpg "Pipeline"
[image8]: ./output_images/histogram.jpg "Histogram"
[image9]: ./output_images/sliding_window.jpg "Sliding Windows"
[image10]: ./output_images/polynomial.jpg "Polynomial"
[image11]: ./output_images/test_images.jpg "Result images"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration on Chessboard

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./01-chessboard.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.

I have converted image to grayscale, and using data from `wide_dist_pickly.p`, I have recevied Matrix and Destition, and then Undistored image.
![alt text][image2]

#### 2. Unwarped image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
h,w = undistort.shape[:2]

src = np.float32([(585,  460),
                  (697,  460),
                  (1044, 690 - 20),
                  (259,  690 - 20)])

dst = np.float32([(320, 0),
                  (960, 0),
                  (960, h),
                  (320, h)])
```

This resulted in the following source and destination points:

| Source        | Destination |
|:-------------:|:-----------:|
| 585, 460      | 320, 0      |
| 697, 460      | 960, 0      |
| 1044,670      | 960, height |
| 259, 670      | 320, height |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 4. Grad binary and RGB, HLS Binary

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image4]
![alt text][image5]

#### 5. Sobels Thresholding

![alt text][image6]
![alt text][image7]

##### 6. Histogram and polynomial
![alt text][image8]
![alt text][image9]
![alt text][image10]


### Result

#### 1. Images
![alt text][image11]

#### 2. Videos

##### Test video
<img src="samples/video.gif"/>

##### Sliding windows
<img src="samples/result_sliding.gif"/>

##### Polynomial
<img src="samples/result_poly.gif"/>


##### Final
<img src="samples/result.gif"/>