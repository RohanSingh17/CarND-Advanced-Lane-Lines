#Project 2 - Advanced Lane Finding Project

The master file for the code is "Advanced_Lane_lines.py'. The code is basically arranged such that different functions are defined which perform different tasks based on requirements. In order to explain the rubric points the references from the master code file is given based line number.

##Goals
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

[image1]: ./output_images/undistorted1.jpg "Undistorted"
[image2]: ./output_images/distortion_corrected1.jpg "Road Transformed"
[image3]: ./output_images/thresholded1.jpg "Binary Example"
[image4]: ./output_images/thresholded_comb1.jpg "Binary Example"
[image5]: ./output_images/Perspective1.jpg "Warp Example"
[image6]: ./output_images/find_lanes1.jpg "Find lanes"
[image7]: ./output_images/polylines1.jpg "Fit Visual"
[image8]: ./output_images/test_images_straight_lines_report.jpg "Output"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
The points shown below are to explain items from rubric

### Camera Calibration

The code for this step is contained in the function **`calibrate_camera()`** which is defined in lines #29 through #51 of 
the master python file **`Advanced_Lane_lines.py`**.  

The function starts by preparing empty object points `Objpts`, which will be the (x, y, z) coordinates of the chessboard
 corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points 
 are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpts` will be 
 appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpts` will be 
 appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 
 The output arrays `objpts` and `imgpts` are used to compute the camera calibration and distortion coefficients using the 
 `cv2.calibrateCamera()` function. 

To undistort the image, appropriate parameters are passed from this function to another function **`cal_undistort()`** which 
uses `cv2.undistort()` function to obtain the undistorted image. The undistorted output image from the functions described 
above are shown below (located in folder `.output_images/undistorted1.jpg`)

![alt text][image1]

### Pipeline (single images)

The pipeline is implemented in the code (lines #412 through #451) which calls different functions as defined in rubric items to plot
lanes in the image shown in the final image.

#### 1. An example of a distortion-corrected image.

The distortion correction on the image is applied in the same way as described above (in part **camera calibration**). 
The example of a distortion corrected image is shown below (located in folder `.output_images/distortion_corrected1.jpg`)
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I have defined a function called **`sobel_threshold()`** to calculate image gradients using sobel function as defined in the code lines #75
through #103. The gradients in x direction,y direction or magnitude and slope can be used as parameters. The image shown below shows 
image gradient in X-direction with threshold of 30 to 250 (located in folder `.output_images/thresholded1.jpg`)

![alt text][image3]

Another function called **`color_threshold()`** is also defined (code lines #62 through #72) which converts the image in HLS space 
and requires channel as an input to be used on to which threshold values can be applied. The threshold values of 100 to 250 and
'_`S`_' channel (channel=2) is used to obtain color threshold image shown below.

The combined image shown is actually obtained by combining the binary images from X-threshold (shown above) and color threshold.
(located in folder `.output_images/thresolded_comb1.jpg`)

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines #105 through #143.
The function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. I chose the hardcoded
source and destination points in the following manner:

```python
src=np.float32([[190,720],[580,457],[705,457],[1140,720]])
dest=np.float32([[200,720],[200,50],[1080,50],[1080,720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 200, 720        | 
| 580, 457      | 200, 50      |
| 705, 457     | 1080, 50      |
| 1140, 720      | 1080, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and 
its warped counterpart to verify that the lines appear almost parallel in the warped image (as shown in the image below located in
folder `.output_images/Perspective1.jpg`)

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

A function **`find_lanes()`** (lines #145 through #223) is defined to identify lanes from the warped binary image. It uses 
similar approach of sliding windows as shown in the lectures. The image shown below shows the lanes identification process 
using sliding window (located in `.output_images/find_lanes1.jpg`)  

![alt text][image6]

The lanes pixels identified above are passed to another function **`fit_poly()`** (lines #311 through #315) which fits polynomial 
to lanes pixels and is passed to another function **`plot_poly()`** (lines #317 through #362) for lane plotting. The output is 
shown below (located in `.output_images/polylines1.jpg`) 

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To identify radius of curvature of the lanes a function called **`rad_curvature()`** (line #285 through #309) is defined. The 
function takes in lane image and pulls the current left and right lane pixels from variable allx and ally defined in class line.
The current allx and ally pixels along with pixel to real life dimension conversion are used to find out radius of curvature and car 
position from lane center.

The position of the car from lane center is found by finding lane center (average of left and right lane starting positions) and subtracting
that from image center. 

The pixel to real life dimension used are shown below :
```python
ym_per_pix=30/720
xm_per_pix=3.7/700
```
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #412 through #450 in my code. Here is an example of the lanes identified in the test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The pipeline for video is implemented in lines #454 through #457. Following is the [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

#####The problems which were faced by me during the implementation of Advanced lane finding project were as follows:

1) **Understanding of classes**: it was hard to develop an understanding of the classes initally which took up majority of the time.
However, implementation of it wasn't a challenge. 

2) **The radius of curvature for both the lanes were coming out to be different**. I think the problems in not being able to 
average the polynomials over multiple frames within the stipulated amount of time as i was not comfortable with using classes 
initially. However, i am implementing it as i am submitting the project. 

#####The pipeline will likely fail when:
1) There is not proper illumination on the lanes 
2) Lanes are not visible during heavy rain, fog or snow
3) The road colour changes drastically within the lanes
4) The road is too curvy.
5) The road contains potholes.
6) The road is going up and down too much (as in hilly areas).
7) Another road is merging onto the road we are driving during which lanes are not present.

##### In order to make it more robust:
1) The averaging the lanes curvature from previous frames should definitely help stabilizing the lane polynomial
randomness and the radius of curvature prediction fluctuations from every frame.

2) Seperating the lane curvature from each other by a set distance might improve prediction of lanes and jumping of 
lanes all over the road.

3) I read about LAB colour space which also might improve prediction with yellow lanes as they can be starting point for
lane detection. Once yellow lane is detected another lane can be searched within a set distance.


