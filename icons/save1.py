import cv2
import imutils
import numpy as np
from webcolors import name_to_rgb
def running_average(bg_img, image, aweight):
    if bg_img is None:
        bg_img = image.copy().astype("float")
    else:
        cv2.accumulateWeighted(image, bg_img, aweight)    
    return bg_img
def get_contours(bg_img, image, threshold=10):    
    diff = cv2.absdiff(bg_img.astype("uint8"), image)    # abs diff betn img and bg    
    _, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    _,cnts,_ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
    if len(cnts) == 0:
        return None
    else:
        max_cnt = max(cnts, key=cv2.contourArea)
        return th, max_cnt

drawing = True
aweight = 0.5 #accumulate weight variable
cam = cv2.VideoCapture(0) # strat the camera
top, right, bottom, left = 250, 400, 480, 640 # ROI box
num_frames=0 # count frame
canvas = None # writing canvas
t=3 # thickness
draw_color = (0, 255, 0) # draw color(ink color)
pointer_color = (255, 255, 0) # pointer color
erase = False # mode flag
take_average=True # flag to indicate take average
bg_img=None #bg image
while True: # loop while everything is true  
    (ret, frame) = cam.read() # read the camera result    
    if ret: # if camera has read frame        
        key = cv2.waitKey(1) & 0xFF # wait for 1ms to key press
        frame = imutils.resize(frame, width=700)        
        frame = cv2.flip(frame, 1) # flip to remove mirror effect        
        clone = frame.copy() # clone it to not mess with real frame
        h, w = frame.shape[:2]        
        roi = frame[top:bottom, right:left] # take roi, to send it onto contour/average extraction       
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # roi to grayscale        
        gray = cv2.GaussianBlur(gray, (7, 7), 0) # add GaussianBlur to eliminate some noise  
       
        if num_frames < 100 and take_average == True: # if to take average and num frames on average taking is lesser 
            bg_img = running_average(bg_img, gray, aweight) # perform running average            
            cv2.putText(clone, str(num_frames), (100, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 5) # put frame number on frame
            num_frames += 1        
        else: 
            num_frames=0                                   
            hand = get_contours(bg_img, gray)  
            take_average=False
            if hand is not None:
                thresholded, segmented = hand
                cv2.drawContours(clone, [segmented+(right,top)], -1, (0, 0, 255))                   
                tshape = thresholded.shape
                sshape = segmented.shape
                new_segmented = segmented.reshape(sshape[0], sshape[-1])
                m = new_segmented.min(axis=0)  
                if type(canvas) == type(None):
                    canvas = np.zeros((tshape[0], tshape[1], 3))+255                
                c = np.zeros(canvas.shape, dtype=np.uint8)
                cv2.circle(c, (m[0], m[1]), 5, pointer_color, -3)
                cv2.circle(clone, (right+m[0], top+m[1]), 5, pointer_color, -3)   
                if(drawing==True):     
                    cv2.circle(canvas, (m[0], m[1]), 5, draw_color, -3)
                    e = cv2.erode(canvas, (3, 3), iterations=5)
                    drawn = cv2.resize(e, (clone.shape[1], clone.shape[0]))
                    c = cv2.resize(c, (clone.shape[1], clone.shape[0]))
                    drawn_new = drawn+c
                    cv2.imshow("Drawing", drawn+c)
                else:
                    c = cv2.resize(c, (clone.shape[1], clone.shape[0]))
                    c = drawn+c
                    cv2.imshow("Drawing", drawn_new)
                    cv2.imshow("c",c)
                    # cv2.imshow("drawn",drawn)
                    # cv2.imshow("tot",drawn+c)


        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)       
        cv2.imshow("Feed", clone) 
        if(key==9):
            drawing = not drawing
        if(key==32):
            color = input("Enter the color: ")
            m = name_to_rgb(color)
            draw_color = (m.blue,m.green,m.red)
            print(draw_color)       
        if key==27: 
            break
cam.release()
cv2.destroyAllWindows()