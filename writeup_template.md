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

I have converted image to grayscale, calling `cv2.undistort` (in Distort functions cell, function "cal_undistort" using data from `wide_dist_pickly.p`, I have recevied Matrix and Destition, and then Undistored image.
![alt text][image2]

#### 2. Unwarped image.

The code for my perspective transform includes a function called `warper()`,  in the "Distort functions" code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
h,w = undistort.shape[:2] # 1280x720

src = np.float32([(585,  460),
                  (697,  460),
                  (1044, 690),
                  (259,  690)])

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
| 1044,690      | 960, 720    |
| 259, 690      | 320, 720    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 3. Gradients and Color Spaces

##### Gradients

~~Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:~~
Functions are stored in "Gradient functions" cell.

Then I processed unwarped image trough Sobel operators; x-gradient, y-gradient and combined xy-gradient.
Also tried magnitude of gradient (with kernel size 3), direction of gradient (arctan(sobel<sub>y</sub> / sobel<sub>x</sub>)), and combing all thresholding combinations.

```python
# Kernel size; ksize=3
gradx      = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady      = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0, np.pi/2))
combined   = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
```

![alt text][image4]

##### Color Thresholding

Functions are stored in "Color functions" cell.

###### RGB

Using unwarped image I have tried many combinations to convert to gray, and Gray binary; then for each color palate, vanilla and binary.

```python
gray_vanilla = cv2.cvtColor(unwarped, cv2.COLOR_RGB2GRAY)
# bin_example
gray_binary  = np.zeros_like(gray_vanilla)
thresh       = (180, 255)
gray_binary[(gray_vanilla > thresh[0]) & (gray_vanilla <= thresh[1])] = 1

# R Example (red)
r_vanilla = unwarped[:,:,0] # 0 -> RED, 1 -> GREEN, 2 -> BLUE
# bin example
r_binary  = np.zeros_like(r_vanilla)
thresh    = (200, 255)
r_binary[(r_vanilla > thresh[0]) & (r_vanilla <= thresh[1])] = 1
```

###### HLS

Converted unwraped image from RGB into HLS color space via "cv2.COLOR_RGB2HLS", and same as in RGB looped all scenarios

```python
# Convert from RGB into HLS colorspace
hls = cv2.cvtColor(unwarped, cv2.COLOR_RGB2HLS)

# H example (hue)
h_vanilla = hls[:, :, 0] # 0 -> HUE, 1 -> VA(L)UE, 2 -> SATURATION
# bin example
h_binary = np.zeros_like(h_vanilla)
thresh   = (15, 100)
h_binary[(h_vanilla > thresh[0]) & (h_vanilla <= thresh[1])] = 1
```

![alt text][image5]

###### Thresholding combined

Functions are stored in "Pipeline" cell.

* Convert to HLS colorspace
* Separate L (Value)
* Take the derative in x
* Absoulute x derative
* Threshold x binary -> Combined S channel and gradient treshold(example)
* Threshold color channel
* Stack each channel -> Stacked thresholds(example)

![alt text][image7]

##### 4. Finding Lines
Functions are stored in "Finding Lines" cell.

For finding lines I choose binary grayscaled warped image.

###### Histogram
Steps:
* Grab only bottom half of image
* Sum across image pixels vertically, axis set to 0
* Visuale graph trough matplot plot function (given Line2D array)
![alt text][image8]

###### Sliding windows
Steps:
* Use histogram (prev. example)
* Split the histogram into two lines (Left: red, Right: blue)
* Window and window hyperparameters (sliding windows:9, margin: 100, minimum number of pixels to recent window 50)
![alt text][image9]

###### Polynomial
Steps:
* Find lane lines
* Using "np.polyfit" find second order polynomial
* Generate x, y values for plotting
![alt text][image10]

##### 5. Result
Note: Curvature and offset info are in video only.

On test images, using previous steps(as in "Process image" cell, process_image function):
* Undistort
* Warp(using same hardcoded src, dest from before)
* Gray binary

Using "find_lane_pixels" ("Finding the lanes" cell), I got values that are needed for measuring curvature (measure_curvature function in "Draw Lines" cell)

```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30  / 720 # meters per pixel in y dimension
xm_per_pix = 3.7 / 700 # meters per pixel in x dimension

# Calculating mid point
left_mean   = np.mean(leftx)
right_mean  = np.mean(rightx)
camera_pos  = (binary.shape[1]/2)-np.mean([left_mean, right_mean])
dist_offset = np.round(camera_pos*xm_per_pix,2)
```

Using "cv2.getPerspectiveTransform" I converted to original image, and colored in green space between lines.

![alt text][image11]

---

### Pipeline (Video)
Using same pipeline as for image examples, with addition of adding curvature and offset info).
Video is solid, could be better.

#### Project [video](https://github.com/Horki/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4) (sample)
<img src="samples/video.gif"/>

#### Final [solution](https://github.com/Horki/CarND-Advanced-Lane-Lines/blob/master/output_videos/result.mp4), [YouTube solution](https://youtu.be/tcUMo39b0j4)
<img src="samples/result.gif"/>