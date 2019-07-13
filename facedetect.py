

from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

# local module
import video
from video import presets
from video import create_capture
from common import clock, draw_str

#define the class for the predict model
class LetterStatModel(object):
    class_n = 26
    train_ratio = 0.

    def load(self, fn):
        self.model = self.model.load(fn)

    def unroll_samples(self, samples):
        sample_n, var_n = samples.shape
        new_samples = np.zeros((sample_n * self.class_n, var_n+1), np.float32)
        new_samples[:,:-1] = np.repeat(samples, self.class_n, axis=0)
        new_samples[:,-1] = np.tile(np.arange(self.class_n), sample_n)
        return new_samples

    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*self.class_n, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*self.class_n )
        new_responses[resp_idx] = 1
        return new_responses

#define the MLP model for predicting
class MLP(LetterStatModel):
    def __init__(self):
        self.model = cv.ml.ANN_MLP_create()

    def predict(self, samples):
        _ret, resp = self.model.predict(samples)
        return resp.argmax(-1)

#show the density distribution of the selection area
def show_hist(hist):
    bin_count = hist.shape[0]
    bin_w = 24
    img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
    for i in xrange(bin_count):
        h = int(hist[i])
        cv.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    cv.imshow('hist', img)

#the algorithm for the actual catch the hand area and predict what character it should be
def camshiftrun(cam,selection,track_window):
    show_backproj = False
    flag=0
    while True:
        _ret, frame = cam.read() 
        vis = frame.copy()
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        #mask--(0,1) check the hsv values and find the three elements if is in the value range
        mask = cv.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        if selection:
            x0, y0, x1, y1 = selection
            # set up the ROI for tracking
            hsv_roi = hsv[y0:y1, x0:x1]
            mask_roi = mask[y0:y1, x0:x1]
            hist = cv.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 300] )
            cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
            hist = hist.reshape(-1)
            show_hist(hist)

            vis_roi = vis[y0:y1, x0:x1]
            cv.bitwise_not(vis_roi, vis_roi)
            vis[mask == 0] = 0

            #get the area to be blocking 
            #the face area
            begY = int(round(selection[1] -80)) 
            endY = int(round(selection[3] +20))
            begX = int(round(selection[0] -40)) 
            endX = int(round(selection[2] +40))
            
            if( begX < 0 ): 
                begX = 0;

            if( endX > vis.shape[1]):
                endX = vis.shape[1]

            if( begY < 0 ): 
                begY = 0;

            if( endY > vis.shape[0]):
                endY = vis.shape[0];

        if track_window and track_window[2] > 0 and track_window[3] > 0:
            
            prob = cv.calcBackProject([hsv], [0], hist, [0, 300], 1)
            prob &= mask
            # make the prob of selection area to be zero
            # blocking the face,hair and shoulder etc.. 
            for y in range(begY, endY):
                for x in range (begX , endX):
                    prob[y][x] = 0

            for y in range(endY, vis.shape[0]):
                for x in range (0 , vis.shape[1]):
                    prob[y][x] = 0

            # Setup the termination criteria, either 10 iteration or move by at least 1 pt
            term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
            #track_box - the center and the parameters for the ellipse area
            #perform the detection for hand on the entire image
            entire_window = (0, 0, np.shape(vis)[0], np.shape(vis)[1])
            track_box, track_window = cv.CamShift(prob, entire_window, term_crit)
            
            pts = cv.boxPoints(track_box)
            pts = np.int0(pts)
            #for the first time into this loop, initial the hisbox parameter
            #hisbox is to stored the previous trackbox 
            if selection:
                selection=None
                hisbox=pts

            cv.polylines(vis,[pts],True, 255,0)

            #when I turn to look at the backproj image, start to capture
            #cut the hand area and save it. I use the right hand only.
            if show_backproj:
                vis[:] = prob[...,np.newaxis]
                #if the current trackbox and the previous trackbox is similiar
                #the flag is to defined that this pattern save or not;1- saved,0-not save
                if ((np.any((hisbox==pts))) and flag==0):
                    index = 0
                    (y1,x1),(y2,x2),(y3,x3),(y4,x4)=pts
                    #calculate the truly hand area
                    ty1 = min(y1,y3)+30
                    ty2 = max(y1,y3)+30
                    tx1 = min(x2,x4)-40
                    tx2 = tx1+(ty2-ty1)
                    if (ty1 <0):
                        ty1=0
                    if (tx1 <0):
                        tx1=0
                    if (ty2 >vis.shape[0]):
                        ty2=vis.shape[0]                                            
                    if (tx2 >vis.shape[1]):
                        tx2=vis.shape[1]
                    #get the area for prediction
                    dect=vis[tx1:tx2,ty1:ty2]
                    
                    #resize the image and get the gray flatten-array to fit the model
                    pic1 = cv.resize(dect,(16, 16), interpolation=cv.INTER_CUBIC)
                    gray = cv.cvtColor(pic1, cv.COLOR_BGR2GRAY)
                    gray =[float(x) for x in np.array(gray).flatten()]
                    print(chr(model.predict(np.array([gray]))+ord('A')))
                    #change the saving option to saved
                    flag=1
                    cv.rectangle(vis, (ty1, tx1),
                                     (ty2, tx2),
                                     (0, 255, 0), 2)
                img2 = cv.polylines(vis,[pts],True, 255,0)
            # if the current trackbox and the previous trackbox is not similiar
            # change the saving option
            if (np.any((hisbox-pts)>8)):
                flag=0

            hisbox=pts

        cv.imshow('facedetect-camshift', vis)

        ch = cv.waitKey(5)
        if ch == 27:
            break
        if ch == ord('b'):
            show_backproj = not show_backproj

def detect(img, cascade): # load the image and the classification

    #use the classification to get the face area
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    # important parameters 
    #scaleFactor stands for the step between the searching windows 
    #minNeighbors: control that if the area works for several times so it can the target area
    #Size minSize = Size()/Size maxSize = Size()
    #flags: zoom the image not the filter; 

    if len(rects) == 0: # if detect nothing
        return []

    rects[:,2:] += rects[:,:2] # adjust the indexes for vertices properly  
    return rects

def draw_rects(img, rects, color): #draw a rectangle form according to the area we captured
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt # get command line options
    #print(__doc__) # print the head markdown

    # Get user supplied values
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args) # make it to a map

    ##make a flag for the starter
    rects=[];

    # We use the already done classification.Before that,we need to get the source address
    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    #nested_fn  = args.get('--nested-cascade', "data/haarcascades/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn)) # get the classification for the face
    #nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn)) # get the classification for eyes

    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('samples/data/lena.jpg')))
    #read the video and ready for the detecting

    # Make a loop so that make up-to-date dections automatically
    while rects==[]:
        ret, img = cam.read() # read the image
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #make the image to grey
        gray = cv.equalizeHist(gray) # make the histogram balance

        t = clock() # mark down the starting time
        rects = detect(gray, cascade) # capture the face area
        dt = clock() - t # the time for the detecting
        #make a copy of image and then can show the new one
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))  # show the image with frame

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000)) # show the timing
        cv.imshow('facedetect-camshift', vis) # show the plot

        if cv.waitKey(5) == 27: # delay for the vesion things and control the breakpoint here
            break

    # get the tracking window's value and set for the cam-shift algorithm
    for xx1, yy1, xx2, yy2 in rects:
        x0=xx1
        y0=yy1
        x1=xx2
        y1=yy2
    selection = (x0, y0, x1, y1)
    track_window = (x0, y0, x1 - x0, y1 - y0)
    
    #load the predict model
    model = MLP()
    args.setdefault('--load', 'fn')
    if '--load' in args:
        fn = args['--load']
    print('loading model from %s ...' % fn)
    model.load(fn)

    #the cam-shift runs
    while True:
        camshiftrun(cam,selection,track_window)
    file_write_obj.close()   
    cv.destroyAllWindows() # to end up all the windows
