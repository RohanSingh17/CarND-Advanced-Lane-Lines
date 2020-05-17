import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import cv2
import glob


###########################################################################333333333333
####   Function to print images for comparison to original image
def print_images(orig,mod,nm,sav_path,orig_nm='Orignal Image',sav=0):
    f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(24, 12))
    f.tight_layout()

    ## Depending on color or grayscale image change plot options
    if len(orig.shape)==3:
        ax1.imshow(orig[:,:,::-1]) ### invert colours to get real image
    elif len(orig.shape)==2:
        ax1.imshow(orig,cmap='gray')
    ax1.set_title(orig_nm, fontsize=50)

    if len(mod.shape)==3:
        ax2.imshow(mod[:,:,::-1])
    elif len(mod.shape)==2:
        ax2.imshow(mod,cmap='gray')
    ax2.set_title(nm,fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

    ## To save plot figure
    if sav==1:
        f.savefig(sav_path)

#####################################################################################
#### Camera calibration function

def calibrate_camera(cal_images,nx=9,ny=6,prn=0,nm='calibrate_camera'):
    imgpts = [] # pixels identified in the image
    objpts = [] # Real life object points
    for name in glob.glob(cal_images):
        img=cv2.imread(name)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        objp=np.zeros((nx*ny,3),np.float32)
        objp[:,:2]=np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        ret,corners = cv2.findChessboardCorners(gray,(nx,ny),None)

        if ret==True:
            img = cv2.drawChessboardCorners(img,(nx,ny),corners,ret)

            imgpts.append(corners)
            objpts.append(objp)
    orig = np.copy(img)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts,imgpts,img.shape[1::-1],None,None)

    ## to print comparison with original image
    if prn==1:
        print_images(orig,img,nm)

    return ret, mtx, dist, rvecs, tvecs

################################################################################################
##### Image undistortion code

def cal_undistort(dist_img,mtx,dist,prn=0,nm='cal_undistort'):
    orig=np.copy(dist_img)
    undst=cv2.undistort(dist_img,mtx,dist,None,mtx)

    if prn==1:
        print_images(orig,undst,nm)

    return undst

################################################################################################
##### Color thresholding of the image

def color_threshold(img,channel,thresh=(30,100),prn=0,nm='color_threshold'):
    orig = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) ## converting image from RGB to HLS space

    binary_image=np.zeros((img.shape[0],img.shape[1]),np.float32)
    binary_image[(hls[:,:,channel]>=thresh[0]) & (hls[:,:,channel]<thresh[1])]=255

    if prn==1:
        print_images(orig,binary_image,nm+" - "+str(channel))

    return binary_image

####################################################################################################
#### Sobel threshold of the image
# orient == 'm' for magnitude or 'gr' for gradient or 'x' or 'y'

def sobel_threshold(img, orient='m',kern=3, thresh=(30, 100),prn=0,nm='sobel_threshold'):
    orig=np.copy(img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sobx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=kern)
    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kern)

    if orient=='m':     # magnitude of sobel
        sob=np.sqrt((sobx**2)+(sobx**2))
    elif orient=='gr':  # gradient of sobel
        abs_sobx=np.absolute(sobx)
        abs_soby = np.absolute(soby)

        sob=np.arctan2(abs_soby,abs_sobx)
    elif orient=='x':   # sobel gradient in x direction
        sob=sobx
    elif orient=='y':   # sobel gradient in y direction
        sob=soby
    else:
        print("Enter a valid value of orient")
        return gray

    binary_image = np.zeros_like(gray)
    binary_image[(sob>= thresh[0]) & (sob < thresh[1])]=1

    if prn==1:
        print_images(orig,binary_image,nm+orient)

    return binary_image

################################################################################################
##### Perspective transform for the bird's eye view of the image

def perspective_transform(img,src,dst,inv=0,rect=0,prn=0,nm='perspective_transform'):

    ## Based on image shape proper parameters will be selected
    if len(img.shape)==3:
        shp=img.shape[1::-1]
        in_img=np.copy(img)
    elif len(img.shape)==2:
        shp=img.shape[::-1]
        in_img = np.dstack((img,img,img))

    if inv==1:
        temp=src
        src=dst
        dst=temp

    M=cv2.getPerspectiveTransform(src,dst)
    warped=cv2.warpPerspective(img,M,shp,flags=cv2.INTER_LINEAR)

    ### used to draw rectangle on the image to visualize perspective
    #   transform from original image to bird's eye view
    if rect==1:
        if len(warped.shape)==3:
            out_img = np.copy(warped)
        else:
            out_img=np.dstack((warped,warped,warped))

        tupl=map(tuple,src.astype(int))
        rect_cord1=tuple(tupl)
        cv2.line(in_img, rect_cord1[0], rect_cord1[1], color=(0, 0, 255), thickness=5)
        cv2.line(in_img, rect_cord1[1], rect_cord1[2], color=(0, 0, 255), thickness=5)
        cv2.line(in_img, rect_cord1[2], rect_cord1[3], color=(0, 0, 255), thickness=5)
        cv2.line(in_img, rect_cord1[3], rect_cord1[0], color=(0, 0, 255), thickness=5)

        tupl=map(tuple,dst.astype(int))
        rect_cord=tuple(tupl)
        cv2.rectangle(out_img, rect_cord[1], rect_cord[3], color=(0, 0, 255),thickness=5)
        print_images(in_img,out_img,nm='Perspective Transform',sav_path='output_images/Perspective.jpg',sav=1)

    if prn==1:
        print_images(in_img,out_img,nm)

    return warped

########################################################################################################
#### Find lanes in the image from scratch using sliding windows approach

def find_lanes(img,prn=0,nm='findlanes'):
    hist=np.sum(img[img.shape[0]//2:,:],axis=0)
    out_img=np.dstack((img,img,img))

    ## starting point of lanes
    midp=hist.shape[0]//2
    leftx=np.argmax(hist[:midp])
    rightx=np.argmax(hist[midp:])+midp

    left_curr=leftx
    right_curr=rightx

    ## parameters for sliding window
    margin=100
    nwindows=9
    minpix=50
    win_height=img.shape[0]//nwindows

    nonzero=img.nonzero()
    nonzerox=nonzero[1]
    nonzeroy=nonzero[0]

    left_lane_ind = [] # left lane pixel indices
    right_lane_ind = [] # right lane pixel indices

    ## Sliding windows implementation to find pixels associated with left and right lanes
    for i in range(nwindows):
    #i=0
        y_low=img.shape[0]-i*win_height
        y_high=img.shape[0]-(i+1)*win_height

        leftx_low=left_curr-margin
        leftx_high = left_curr +margin
        rightx_low=right_curr-margin
        rightx_high=right_curr+margin

        cv2.rectangle(out_img,(leftx_low,y_high),(leftx_high,y_low),color=(255,0,0),thickness=5)
        cv2.rectangle(out_img, (rightx_low,y_high), (rightx_high,y_low), color=(255, 0, 0),thickness=5)

        left_ind=np.where((nonzerox>=leftx_low)&(nonzerox<leftx_high)&(nonzeroy<y_low)&(nonzeroy>=y_high))
        right_ind = np.where((nonzerox >= rightx_low) & (nonzerox < rightx_high) & (nonzeroy < y_low) & (nonzeroy >= y_high))

        left_lane_ind.append(left_ind)
        right_lane_ind.append(right_ind)

        if len(left_ind[0])>minpix:
            left_curr=np.int(np.average(nonzerox[left_ind[0]]))
        if len(right_ind[0])>minpix:
            right_curr=np.int(np.average(nonzerox[right_ind[0]]))

    left_lane_ind=np.concatenate(left_lane_ind,axis=1)
    left.detected=True
    right_lane_ind=np.concatenate(right_lane_ind,axis=1)
    right.detected = True

    leftx=nonzerox[left_lane_ind]
    lefty=nonzeroy[left_lane_ind]
    rightx=nonzerox[right_lane_ind]
    righty=nonzeroy[right_lane_ind]

    left.allx = leftx[0,:]
    left.ally = lefty[0, :]
    right.allx = rightx[0, :]
    right.ally = righty[0, :]

    # coloring pixels from lanes in different colors
    out_img[left.ally,left.allx]=[255,0,0]
    out_img[right.ally,right.allx]=[0,0,255]

    # fit_poly function to fit polynomial to lane pixels
    left_poly,right_poly=fit_poly()

    left.recent_xfitted.append(left_poly)
    right.recent_xfitted.append(right_poly)

    # if len(left.recent_xfitted)>1:
    #     left.diffs.append(np.array(left.recent_xfitted[-1])-np.array(left.recent_xfitted[-2]))
    #     right.diffs.append(np.array(right.recent_xfitted[-1]) - np.array(right.recent_xfitted[-2]))
    #
    # if len(left.recent_xfitted)>5:
    #     left.average_fit=np.average(np.array(left.recent_xfitted)[-5:,:])

    return left_poly,right_poly

###################################################################################################
#### Search around previous polynomials to find lane pixels in next frames

def search_around_poly(warped):
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    binary_image=np.zeros_like(warped)
    image_half=np.zeros_like(warped)
    margin=100
    minpix=200

    left_poly=left.recent_xfitted[-1]
    right_poly=right.recent_xfitted[-1]

    left_p=np.poly1d(left_poly)
    leftx_plot=left_p(ploty)

    right_p=np.poly1d(right_poly)
    rightx_plot=right_p(ploty)

    pts_left=np.array([leftx_plot - margin, ploty,leftx_plot + margin, ploty]).T
    pts_left=pts_left.reshape(-1,2)

    pts_right=np.array([rightx_plot - margin, ploty,rightx_plot + margin, ploty]).T
    pts_right=pts_right.reshape(-1,2)

    ## creating mask around left and right lanes
    cv2.fillPoly(binary_image,np.int64([pts_left]),1)
    cv2.fillPoly(binary_image, np.int64([pts_right]),1)

    pts=np.array([(leftx_plot + rightx_plot)//2, ploty,np.zeros(ploty.shape), ploty]).T
    pts=pts.reshape(-1,2)

    # dividing the image in two halves from mid of the lanes
    cv2.fillPoly(image_half, np.int64([pts]),1)

    ## finding lane pixels using masking approach
    left_lane=(binary_image*warped*image_half).nonzero()
    right_lane=(binary_image*warped*(1-image_half)).nonzero()

    ## if sufficient lane pixels are found around both lanes the new polynomials are fitted
    if len(left_lane[0])>minpix & len(right_lane[0])>minpix:
        left.detected=True
        right.detected=True

        left.allx=left_lane[1]
        left.ally =left_lane[0]
        right.allx=right_lane[1]
        right.ally = right_lane[0]

        left_poly_curr,right_poly_curr=fit_poly()

        left_p_curr=np.poly1d(left_poly_curr)
        leftx_top_curr=left_p_curr(0)

        right_p_curr=np.poly1d(right_poly_curr)
        rightx_top_curr=right_p_curr(0)

        left.diffx=leftx_top_curr-leftx_plot[-1]
        right.diffx=rightx_top_curr-rightx_plot[-1]

        left.recent_xfitted.append(left_poly_curr)
        right.recent_xfitted.append(right_poly_curr)

    ## else switch to find lanes code to find lanes from scratch
    else:
        left_poly_curr,right_poly_curr=find_lanes(warped)

    return left_poly_curr,right_poly_curr

############################################################################################
#### Find radius of curvature and distance of car from the lane center

def rad_curvature(warped):
    ym_per_pix=30/720
    xm_per_pix=3.7/700

    y_eval=(warped.shape[0])*ym_per_pix

    left_poly=np.polyfit(left.ally*ym_per_pix,left.allx*xm_per_pix,2)
    lp=np.poly1d(left_poly)
    left_center=lp(y_eval)

    right_poly = np.polyfit(right.ally*ym_per_pix, right.allx*xm_per_pix, 2)
    rp=np.poly1d(right_poly)
    right_center=rp(y_eval)

    rad_left=np.abs(np.power((1+np.square(2*left_poly[0]*y_eval+left_poly[1])),1.5)/(2*left_poly[0]))
    rad_right = np.abs(np.power((1 + np.square(2 * right_poly[0] * y_eval + right_poly[1])), 1.5) / (2 * right_poly[0]))

    left.radius_of_curvature=rad_left
    right.radius_of_curvature=rad_right

    lane_center=(left_center+right_center)/2
    car_ln_center=(xm_per_pix*warped.shape[1]/2)-lane_center

    # positive value means car is on the right side of the center of the lanes
    left.line_base_pos=car_ln_center

###############################################################################################
#### used to fit polynomial around identified lane pixels

def fit_poly(prn=0,nm='fit_poly'):
    left_poly=np.polyfit(left.ally,left.allx,2)
    right_poly=np.polyfit(right.ally,right.allx,2)

    return left_poly,right_poly

###############################################################################################
#### Used to plot lane boundaries

def plot_poly(warped, left_poly, right_poly, prn=0, nm='fit_poly'):
    out_img=np.dstack((warped,warped,warped))*255
    margin=50

    ploty=np.linspace(0,warped.shape[0]-1,warped.shape[0])
    left_p=np.poly1d(left_poly)
    leftx_plot=left_p(ploty)

    right_p=np.poly1d(right_poly)
    rightx_plot=right_p(ploty)

    out_img[left.ally,left.allx]=[255,0,0]
    out_img[right.ally,right.allx]=[0,0,255]

    l_pts=np.array([leftx_plot, ploty]).T
    r_pts=np.array([rightx_plot, ploty]).T

    cv2.polylines(out_img,np.int64([l_pts]),isClosed=False,color=(0,255,255),thickness=5)
    cv2.polylines(out_img, np.int64([r_pts]), isClosed=False, color=(0, 255, 255), thickness=5)

    # f,ax=plt.subplots(1,1)
    # ax.imshow(out_img)
    # plt.show()
    # f.savefig('output_images/polylines.jpg')

    window_img=np.zeros_like(out_img)

    pts_left=np.array([leftx_plot - margin, ploty,leftx_plot + margin, ploty]).T
    pts_left=pts_left.reshape(-1,2)

    pts_right=np.array([rightx_plot - margin, ploty,rightx_plot + margin, ploty]).T
    pts_right=pts_right.reshape(-1,2)

    poly = np.array([leftx_plot, ploty,rightx_plot, ploty]).T
    poly = poly.reshape(-1, 2)

    cv2.fillPoly(window_img,np.int64([pts_left]),color=(0,0,255))
    cv2.fillPoly(window_img,np.int64([pts_right]), color=(255, 0, 0))
    cv2.fillPoly(window_img, np.int64([poly]), color=(0, 255, 0))

    result=img_weighted(out_img,window_img)

    if prn==1:
        print_images(out_img,result,nm)

    return result,window_img

######################################################################################
#### Find weighted image from two images

def img_weighted(main_img,overlay_img,a=1,b=0.3):
    return cv2.addWeighted(main_img, a, overlay_img, b, 0)

########################################################################################
#### Put text in the image

def image_text(img,string,coord=(100,100),type=4,scl=2,color=(255,255,255),thickness=2):
    return cv2.putText(img, string,coord, type, scl, (255, 255, 255), thickness)

###########################################################################################
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.average_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in x-coordinate at top
        self.diffx=[]
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

left=Line()
right=Line()

#############################################################################################
#### camera calibration

cal_images='camera_cal/calibration*.jpg'
image=cv2.imread('test_images/test5.jpg')
ret, mtx, dist, rvecs, tvecs=calibrate_camera(cal_images)

################################################################################################
#### defining source and destination points for perspective transform

src=np.float32([[190,720],[581,457],[703,457],[1140,720]])
dest=np.float32([[200,720],[200,50],[1080,50],[1080,720]])

###################################################################################################
#### FOR INDIVIDUAL IMAGE PROCESSING

# for name in glob.glob('test_images/*.jpg'):
# # name='test_images/test3.jpg'
#     image=cv2.imread(name)        #   Read image
#     undst=cal_undistort(image,mtx,dist)   # undistort image
#     sobx=sobel_threshold(undst, orient='x',kern=3, thresh=(30, 250))      # sobel threshold in X- direction
#     #soby=sobel_threshold(undst, orient='y',kern=3, thresh=(30, 250))
#     # sobm=sobel_threshold(undst, orient='m',kern=3, thresh=(70, 250))
#     # sobgr=sobel_threshold(undst, orient='gr',kern=3, thresh=(0.7, 1.1))
#     # color_threshold(lane_image,channel=0,thresh=(100,250),prn=1)
#     col_binary=color_threshold(undst,channel=2,thresh=(100,250))      # color threshold
#
#     combined=np.zeros_like(col_binary)
#     combined[(sobx>0) | (col_binary>0)]=1     # combining thresholds

#     warped=perspective_transform(combined,src,dest)       # perspective tranform Bird's eye view
#     left_poly,right_poly=find_lanes(warped)               # find lanes from scratch
#     result,window_img=plot_poly(warped,left_poly,right_poly)  # plot polynomial around lanes
#     rad_curvature(warped)     # radius of curvature of lanes
#
#     rad_left=left.radius_of_curvature
#     rad_right=right.radius_of_curvature
#     car_ln_center=left.line_base_pos
#
#     warped=perspective_transform(window_img,src,dest,inv=1)       # perspetive transform to original image frame
#     processed_img=img_weighted(image,warped.astype(np.uint8))     # projecting lanes to original image
#
#     rad_str='Radius of Curvature = '+str(int((rad_left+rad_right)/2))+'m'
#
#     if car_ln_center>0:
#         cen_str='Vehicle is '+str(np.abs(round(car_ln_center,2)))+'m right of center'
#     else:
#         cen_str = 'Vehicle is ' + str(np.abs(round(car_ln_center,2))) + 'm left of center'
#
#     final_img=image_text(processed_img,rad_str,coord=(warped.shape[1]//2-500,50))     # put text on the image
#     final_img=image_text(processed_img,cen_str,coord=(warped.shape[1]//2-500,125))
#
#     output_filepath='output_images/test_images_'+name[12:]
#     cv2.imwrite(output_filepath,final_img)

# FOR VIDEO FRAME PROCESSING
def img_proc(image):
    undst = cal_undistort(image, mtx, dist)     # undistort the image

    sobx = sobel_threshold(undst, orient='x', kern=3, thresh=(30, 250))     # sobel threshold in X-direction
    col_binary = color_threshold(undst, channel=2, thresh=(100, 250))       # color threshold

    combined = np.zeros_like(col_binary)
    combined[(sobx > 0) | (col_binary > 0)] = 1             # combined threshold

    warped = perspective_transform(combined, src, dest)     # perspective transform bird's eye view
    if len(left.recent_xfitted)<1:                          # finding lanes pixels from scratch
        left_poly, right_poly = find_lanes(warped)
    else:                                                   # finding lanes pixels around previous polynomials
        left_poly, right_poly = search_around_poly(warped)
    result, window_img = plot_poly(warped, left_poly, right_poly)   # plotting polynomials around lanes
    rad_curvature(warped)             # finding radius of curvature

    rad_left = left.radius_of_curvature
    rad_right = right.radius_of_curvature
    car_ln_center = left.line_base_pos

    warped = perspective_transform(window_img, src, dest, inv=1)    # reverse perspective transform to original frame
    processed_img = img_weighted(image, warped.astype(np.uint8))    # merge lanes onto original image

    rad_str = 'Radius of Curvature = ' + str(int((rad_left + rad_right) / 2)) + 'm'

    if car_ln_center > 0:
        cen_str = 'Vehicle is ' + str(np.abs(round(car_ln_center, 2))) + 'm right of center'
    else:
        cen_str = 'Vehicle is ' + str(np.abs(round(car_ln_center, 2))) + 'm left of center'

    final_img = image_text(processed_img, rad_str, coord=(warped.shape[1] // 2 - 500, 50))  # put text on the image
    final_img = image_text(processed_img, cen_str, coord=(warped.shape[1] // 2 - 500, 125))

    return final_img

# video paths
video1='harder_challenge_video.mp4'
output_filepath='trial/'+video1
clip1=VideoFileClip(video1)
mod=clip1.fl_image(img_proc)
mod.write_videofile(output_filepath, audio=False)


