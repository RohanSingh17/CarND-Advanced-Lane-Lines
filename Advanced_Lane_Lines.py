import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import cv2
import glob

def print_images(orig,mod,nm):
    f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(24, 12))
    f.tight_layout()

    if len(orig.shape)==3:
        ax1.imshow(orig[:,:,::-1])
    elif len(orig.shape)==2:
        ax1.imshow(orig,cmap='gray')
    ax1.set_title('Original Image', fontsize=50)

    if len(mod.shape)==3:
        ax2.imshow(mod[:,:,::-1])
    elif len(mod.shape)==2:
        ax2.imshow(mod,cmap='gray')
    ax2.set_title(nm,fontsize=50)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def calibrate_camera(cal_images,nx=9,ny=6,prn=0,nm='calibrate_camera'):
    imgpts = []
    objpts = []
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
    if prn==1:
        print_images(orig,img,nm)

    return ret, mtx, dist, rvecs, tvecs

def cal_undistort(dist_img,mtx,dist,prn=0,nm='cal_undistort'):
    orig=np.copy(dist_img)
    undst=cv2.undistort(dist_img,mtx,dist,None,mtx)

    if prn==1:
        print_images(orig,undst,nm)

    return undst

def color_threshold(img,channel,thresh=(30,100),prn=0,nm='color_threshold'):
    orig = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    binary_image=np.zeros((img.shape[0],img.shape[1]),np.float32)
    binary_image[(hls[:,:,channel]>=thresh[0]) & (hls[:,:,channel]<thresh[1])]=255

    if prn==1:
        print_images(orig,binary_image,nm+" - "+str(channel))

    return binary_image


# orient == 'm' for magnitude or 'gr' for gradient or 'x' or 'y'
def sobel_threshold(img, orient='m',kern=3, thresh=(30, 100),prn=0,nm='sobel_threshold'):
    orig=np.copy(img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sobx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=kern)
    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kern)

    if orient=='m':
        sob=np.sqrt((sobx**2)+(sobx**2))
    elif orient=='gr':
        abs_sobx=np.absolute(sobx)
        abs_soby = np.absolute(soby)

        sob=np.arctan2(abs_soby,abs_sobx)
    elif orient=='x':
        sob=sobx
    elif orient=='y':
        sob=soby
    else:
        print("Enter a valid value of orient")
        return gray

    binary_image = np.zeros_like(gray)
    binary_image[(sob>= thresh[0]) & (sob < thresh[1])]=1

    if prn==1:
        print_images(orig,binary_image,nm+orient)

    return binary_image

def perspective_transform(img,src,dst,inv=0,rect=0,prn=0,nm='perspective_transform'):
    orig=np.copy(img)

    if len(img.shape)==3:
        shp=img.shape[1::-1]
    elif len(img.shape)==2:
        shp=img.shape[::-1]

    if inv==1:
        temp=src
        src=dst
        dst=temp

    M=cv2.getPerspectiveTransform(src,dst)
    warped=cv2.warpPerspective(img,M,shp,flags=cv2.INTER_LINEAR)

    if rect==1:
        out_img=np.dstack((warped,warped,warped))
        in_img=np.dstack((img,img,img))

        tupl=map(tuple,src.astype(int))
        rect_cord1=tuple(tupl)
        cv2.line(in_img, rect_cord1[0], rect_cord1[1], color=(0, 0, 255), thickness=5)
        cv2.line(in_img, rect_cord1[2], rect_cord1[3], color=(0, 0, 255), thickness=5)

        tupl=map(tuple,dst.astype(int))
        rect_cord=tuple(tupl)
        cv2.rectangle(out_img, rect_cord[1], rect_cord[3], color=(0, 0, 255),thickness=5)

    #print_images(in_img, out_img, nm)
    if prn==1:
        print_images(in_img,out_img,nm)

    return warped

def find_lanes(img,prn=0,nm='findlanes'):
    hist=np.sum(img[img.shape[0]//2:,:],axis=0)
    out_img=np.dstack((img,img,img))

    midp=hist.shape[0]//2
    leftx=np.argmax(hist[:midp])
    rightx=np.argmax(hist[midp:])+midp
    #print(midp,leftx,rightx,sep='\t')

    left_curr=leftx
    right_curr=rightx

    margin=100
    nwindows=9
    minpix=50
    win_height=img.shape[0]//nwindows

    nonzero=img.nonzero()
    nonzerox=nonzero[1]
    nonzeroy=nonzero[0]

    left_lane_ind = []
    right_lane_ind = []

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
    right_lane_ind=np.concatenate(right_lane_ind,axis=1)

    leftx=nonzerox[left_lane_ind]
    lefty=nonzeroy[left_lane_ind]
    rightx=nonzerox[right_lane_ind]
    righty=nonzeroy[right_lane_ind]

    leftx = leftx[0,:]
    lefty = lefty[0, :]
    rightx = rightx[0, :]
    righty = righty[0, :]

    return leftx,lefty,rightx,righty

def rad_curvature(warped,leftx,lefty,rightx,righty):
    ym_per_pix=15/720
    xm_per_pix=3.7/900

    y_eval=(warped.shape[0])*ym_per_pix

    left_poly=np.polyfit(lefty*ym_per_pix,leftx*xm_per_pix,2)
    lp=np.poly1d(left_poly)
    left_center=lp(y_eval)

    right_poly = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    rp=np.poly1d(right_poly)
    right_center=rp(y_eval)

    rad_left=np.abs(np.power((1+np.square(2*left_poly[0]*y_eval+left_poly[1])),1.5)/(2*left_poly[0]))
    rad_right = np.abs(np.power((1 + np.square(2 * right_poly[0] * y_eval + right_poly[1])), 1.5) / (2 * right_poly[0]))

    left.radius_of_curvature=rad_left
    right.radius_of_curvature=rad_right

    lane_center=(left_center+right_center)/2
    car_ln_center=np.abs((xm_per_pix*warped.shape[1]/2)-lane_center)

    return rad_left,rad_right,car_ln_center

def fit_poly(warped,leftx,lefty,rightx,righty,prn=0,nm='fit_poly'):
    out_img=np.dstack((warped,warped,warped))*255
    margin=20

    left_poly=np.polyfit(lefty,leftx,2)
    right_poly=np.polyfit(righty,rightx,2)

    ploty=np.linspace(0,warped.shape[0]-1,warped.shape[0])
    left_p=np.poly1d(left_poly)
    leftx_plot=left_p(ploty)

    right_p=np.poly1d(right_poly)
    rightx_plot=right_p(ploty)

    out_img[lefty,leftx]=[255,0,0]
    out_img[righty,rightx]=[0,0,255]
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

    return result,window_img,left_poly,right_poly,ploty

def img_weighted(main_img,overlay_img,a=1,b=0.3):
    return cv2.addWeighted(main_img, a, overlay_img, b, 0)

def image_text(img,string,coord=(100,100),type=4,scl=2,color=(255,255,255),thickness=2):
    return cv2.putText(img, string,coord, type, scl, (255, 255, 255), thickness)

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
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

global i
i=0

left=Line()
right=Line()

cal_images='camera_cal/calibration*.jpg'
dist_img=cv2.imread('camera_cal/calibration1.jpg')
video1='harder_challenge_video.mp4'
ret, mtx, dist, rvecs, tvecs=calibrate_camera(cal_images)

src=np.float32([[190,720],[580,457],[705,457],[1140,720]])
dest=np.float32([[200,720],[200,50],[1080,50],[1080,720]])

#def img_proc(image):
for name in glob.glob('test_images/*.jpg'):
    image=cv2.imread(name)
    i=i+1
    undst=cal_undistort(image,mtx,dist)
    sobx=sobel_threshold(undst, orient='x',kern=3, thresh=(30, 250))
    #soby=sobel_threshold(undst, orient='y',kern=3, thresh=(30, 250))
    # sobm=sobel_threshold(undst, orient='m',kern=3, thresh=(70, 250))
    # sobgr=sobel_threshold(undst, orient='gr',kern=3, thresh=(0.7, 1.1))
    #color_threshold(lane_image,channel=0,thresh=(100,250),prn=1)
    col_binary=color_threshold(undst,channel=2,thresh=(100,250))

    combined=np.zeros_like(col_binary)
    combined[(sobx>0) | (col_binary>0)]=1
    # print_images(sobx,soby,sobm,sobgr,col_binary,combined, nm='transform')
    #color_threshold(lane_image,channel=2,thresh=(30,250),prn=1)
    #print_images(lane_image,combined,nm='combined')

    warped=perspective_transform(combined,src,dest)
    leftx,lefty,rightx,righty=find_lanes(warped)
    result,window_img,left_poly,right_poly,ploty=fit_poly(warped,leftx,lefty,rightx,righty)
    rad_left,rad_right,car_ln_center=rad_curvature(warped,leftx,lefty,rightx,righty)

    warped=perspective_transform(window_img,src,dest,inv=1)
    processed_img=img_weighted(image,warped.astype(np.uint8))

    rad_str='Radius of Curvature = '+str(int((rad_left+rad_right)/2))+'m'

    if car_ln_center>0:
        cen_str='Vehicle is '+str(round(car_ln_center,2))+'m left of center'
    else:
        cen_str = 'Vehicle is ' + str(round(car_ln_center,2)) + 'm right of center'

    final_img=image_text(processed_img,rad_str,coord=(warped.shape[1]//2-500,50))
    final_img=image_text(processed_img,cen_str,coord=(warped.shape[1]//2-500,125))

    # output_filepath='output_videos/project_video'+str(i)+'.jpg'
    # cv2.imwrite(output_filepath,final_img[:,:,::-1])
    #
    print_images(image,final_img,nm='lanes')
    print(left.radius_of_curvature,right.radius_of_curvature,sep='\t')
    # return final_img

# output_filepath='output_videos/harder_challenge_video.mp4'
# clip1=VideoFileClip(video1)
# mod=clip1.fl_image(img_proc)
# mod.write_videofile(output_filepath, audio=False)


