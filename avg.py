import cv2
import imutils
import numpy as np
def running_average(bg_img, image, aweight):
    if bg_img is None:
        bg_img = image.copy().astype("float")
    else:
        cv2.accumulateWeighted(image, bg_img, aweight)    
    return bg_img
def get_contours(bg_img, image, threshold=10):    
    diff = cv2.absdiff(bg_img.astype("uint8"), image)    # abs diff betn img and bg    
    _, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    cnts,_ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
    if len(cnts) == 0:
        return None
    else:
        max_cnt = max(cnts, key=cv2.contourArea)
        return th, max_cnt
class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
       self.top = 250
       self.bottom = 480
       self.left = 640
       self.right = 400
       self.t=5
       self.draw_color = (0, 255, 0)
       self.pointer_color = (255, 0, 0)
       self.erase = False
       self.bg_img = None
       self.canvas = None
       self.num_frames = 0
       self.take_average = True
       self.draw = True
       self.er = False
    def close(self):
        #releasing camera
        self.video.release()

    def get_frame(self,aweight,take_average):
        ret, frame = self.video.read()
        top = self.top
        bottom = self.bottom
        right = self.right
        left = self.left
        bg_img = self.bg_img
        num_frames = self.num_frames
        if ret:
        # wait for 1ms to key press
            #key = cv2.waitKey(1) & 0xFF
            frame = imutils.resize(frame, width=700)
        # flip to remove mirror effect
            frame = cv2.flip(frame, 1)
                # clone it to not mess with real frame
            clone = frame.copy()
            h, w = frame.shape[:2]
                # take roi, to send it onto contour/average extraction
            roi = frame[top:bottom, right:left]
                # roi to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # add GaussianBlur to eliminate some noise
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            if num_frames<100 and take_average==True:
                # perform running average
                bg_img = running_average(bg_img, gray, aweight)
                # put frame number on frame
                cv2.putText(clone, str(num_frames), (100, 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 5)
                num_frames+=1
            else:
                cv2.putText(clone, "done with the average", (100, 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 5)
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
            ret, jpeg = cv2.imencode('.jpg', clone)
            bytes_clone = jpeg.tobytes()
            self.bg_img = bg_img
            self.num_frames = num_frames
            return num_frames,bg_img,bytes_clone

    def write(self): 
        ret, frame = self.video.read()
        draw_this_run = np.zeros((525,700,3))
        top = self.top
        bottom = self.bottom
        right = self.right
        left = self.left
        t = self.t
        canvas = self.canvas
        draw_color = self.draw_color 
        pointer_color = self.pointer_color
        erase = self.draw
        bg_img = self.bg_img
        er = self.er
        if ret:
        # wait for 1ms to key press
            key = cv2.waitKey(1) & 0xFF
            frame = imutils.resize(frame, width=700)
        # flip to remove mirror effect
            frame = cv2.flip(frame, 1)
                # clone it to not mess with real frame
            clone = frame.copy()
            h, w = frame.shape[:2]
                # take roi, to send it onto contour/average extraction
            roi = frame[top:bottom, right:left]
                # roi to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # add GaussianBlur to eliminate some noise
            gray = cv2.GaussianBlur(gray, (7, 7), 0)                        
                # take our segmented hand
            hand = get_contours(bg_img, gray)
            take_average=False
            if hand is not None:
                
                # #if pressed x, erase
                # if chr(key) == "x":
                #     draw_color = (255, 255, 255)
                #     pointer_color = (0, 0, 255)
                #     erase = True
                # if chr(key) == "c":
                #     draw_color = (0, 255, 0)
                #     pointer_color = (255, 0, 0)
                #     erase = False
                # #idle
                # if chr(key) == "z":
                #     erase = None
                #     pointer_color = (0, 0, 0)                   
                # # restart system
                # if chr(key) == "r":
                #     take_average=True
                #     canvas = None
                # if chr(key) == "e":
                #     canvas = None
                #     drawn = np.zeros(drawn.shape)+255
                thresholded, segmented = hand
                cv2.drawContours(clone, [segmented+(right,top)], -1, (0, 0, 255))   
                tshape = thresholded.shape
                sshape = segmented.shape
                new_segmented = segmented.reshape(sshape[0], sshape[-1])
                m = new_segmented.min(axis=0)
                if type(canvas) == type(None):
                    canvas = np.zeros((tshape[0], tshape[1], 3))+255
                c = np.zeros(canvas.shape, dtype=np.uint8)
                cv2.circle(c, (m[0], m[1]), t-2, pointer_color, -3)
                cv2.circle(clone, (right+m[0], top+m[1]), t, pointer_color, -3)
                
                if er==True:
                    cv2.circle(canvas, (m[0], m[1]), t+2, (255,255,255), -3)
                    erimg=cv2.circle(canvas.copy(), (m[0], m[1]), t+2, (0, 0, 0), -3)            
                    cv2.circle(c, (m[0], m[1]), t+2, (0, 0, 255), -3)
                    e = cv2.erode(erimg, (3, 3), iterations=5)
                    drawn = cv2.resize(e, (clone.shape[1], clone.shape[0]))
                    c = cv2.resize(c, (clone.shape[1], clone.shape[0]))
                    drawn_new = drawn+c
                    draw_this_run = drawn_new.copy()
                    #cv2.imshow("Drawing", drawn+c)
                    erimg=cv2.circle(canvas.copy(), (m[0], m[1]), t+2, (255, 255, 255), -3)            
                    cv2.circle(c, (m[0], m[1]), t+2, (0, 0, 255), -3)
                    e = cv2.erode(erimg, (3, 3), iterations=5)
                    drawn = cv2.resize(e, (clone.shape[1], clone.shape[0])) 
                    self.drawn = drawn                   
                elif erase==True:
                    cv2.circle(canvas, (m[0], m[1]), t-2, draw_color, -3)
                    canvas_shape = canvas.shape
                    clone_shape = clone.shape
                    eshape = (clone_shape[0]/canvas_shape[0], clone_shape[1]/canvas_shape[1])
                    m[0] = int(eshape[1]*m[0])
                    m[1] = int(eshape[0]*m[1])
                    e = cv2.erode(canvas, (3, 3), iterations=5)
                    drawn = cv2.resize(e, (clone.shape[1], clone.shape[0]))
                    c = cv2.resize(c, (clone.shape[1], clone.shape[0]))
                    drawn_new = drawn+c
                    x = drawn_new.copy()
                    cv2.circle(x, (m[0], m[1]), t, pointer_color, -3)
                    draw_this_run = x.copy()
                    self.drawn = drawn
                    #cv2.imshow("Drawing", drawn+c)
                elif erase == False:
                    drawn = self.drawn
                    canvas_shape = canvas.shape
                    clone_shape = clone.shape
                    eshape = (clone_shape[0]/canvas_shape[0], clone_shape[1]/canvas_shape[1])
                    m[0] = int(eshape[1]*m[0])
                    m[1] = int(eshape[0]*m[1])
                    drawn = cv2.resize(drawn, (clone.shape[1], clone.shape[0]))
                    dc = drawn.copy()  
                    cv2.circle(dc, (m[0], m[1]), t, pointer_color, -3)
                    draw_this_run = dc.copy()
                #cv2.imshow("Drawing",draw_this_run)
                #print("shape:",clone.shape)
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow("Feed", clone)
        ret, jpeg = cv2.imencode('.jpg', draw_this_run)
        bytes_clone = jpeg.tobytes()
        self.canvas = canvas
        return canvas,bytes_clone



    
