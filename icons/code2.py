import imutils
import time
import cv2
import numpy as np
import os
def running_average(bg_img, image, aweight):
    if bg_img is None:
        bg_img = image.copy().astype("float")
    else:
        cv2.accumulateWeighted(image, bg_img, aweight)    
    return bg_img

def get_contours(bg_img, image, threshold=10):
    
    # abs diff betn img and bg
    diff = cv2.absdiff(bg_img.astype("uint8"), image)    
    _, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    (cnts, _) = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) == 0:
        return None
    else:
        max_cnt = max(cnts, key=cv2.contourArea)
        return th, max_cnt

def run_system(count_mode=5, avg_frames=100, hroi=[250, 400, 480, 681],
               groi=[150, 400, 240, 681], icon_dir="icons/", color_dir="colors/"
              ):
    #accumulate weight variable
    aweight = 0.5
    # strat the camera
    cam = cv2.VideoCapture(0)
    # ROI box
    top, right, bottom, left = hroi
    # move ROI
    mtop, mright, mbottom, mleft =  100, 10, 250, 160
    
    # ROI for GUI

    #show(icons_holder)

    # read colors
    

    # count frame
    num_frames=0
    # writing canvas
    canvas = None
    # thickness
    t=3
    # draw color(ink color)
    draw_color = (0, 255, 0)
    # pointer color
    pointer_color = (255, 0, 0)
    # mode flag
    erase = False
    # flag to indicate take average
    take_average=True
    #bg image
    draw_bg=None
    gui_bg = None
    move_bg = None
    
    previous_mode = "move"
    running_mode = "move"
    drawn = None
    count_modes = 0
    vui_canvas = None
    current_color = (20, 100, 50)
    colors_array = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255)]).tolist()
    # loop while everything is true
    while True:
        # read the camera result
        (ret, frame) = cam.read()
        # if camera has read frame
        if ret:
            # wait for 1ms to key press
            key = cv2.waitKey(1) & 0xFF
            frame = imutils.resize(frame, width=700)
            # flip to remove mirror effect
            frame = cv2.flip(frame, 1)
            # clone it to not mess with real frame
            clone = frame.copy()
            gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]
            # take roi, to send it onto contour/average extraction
            draw_roi = frame[top:bottom, right:left]
            # roi to grayscale
            #draw_gray = cv2.cvtColor(draw_roi, cv2.COLOR_BGR2GRAY)
            draw_gray = gray[top:bottom, right:left]
            # add GaussianBlur to eliminate some noise
            draw_gray = cv2.GaussianBlur(draw_gray, (7, 7), 0)


            gui_roi = frame[gtop:gbottom, gright:gleft]
            #gui_gray = cv2.cvtColor(gui_roi, cv2.COLOR_BGR2GRAY)
            gui_gray = gray[gtop:gbottom, gright:gleft]
            gui_gray = cv2.GaussianBlur(gui_gray, (7, 7), 0)
            
            move_gray = gray[mtop:mbottom, mright:mleft]
            
            if vui_canvas is None:
                gshape = gray.shape
                vui_canvas = cv2.resize(icons_holder, (gshape[1], 100)).astype(np.uint8)
                vshape = vui_canvas.shape
                vui = np.zeros_like(frame)
                vui[:100, :] = vui_canvas

            # if to take average and num frames on average taking is lesser than 
            if num_frames<avg_frames and take_average==True:
                # perform running average
                draw_bg = running_average(draw_bg, draw_gray, aweight)                
                gui_bg = running_average(gui_bg, gui_gray, aweight=aweight)
                move_bg = running_average(move_bg, move_gray, aweight=aweight)
                # put frame number on frame
                cv2.putText(clone, str(num_frames), (100, 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
                num_frames+=1
            # if not to take average
            else:
                num_frames=0                        
                # take our segmented hand
                gui_hand = get_contours(gui_bg, gui_gray)
                draw_hand = get_contours(draw_bg, draw_gray)
                move_hand = get_contours(move_bg, move_gray)
                take_average=False

                if gui_hand is not None:
                    # get the position of contours
                    gthresholded, gsegmented = gui_hand
                    cv2.drawContours(clone, [gsegmented+(gright,gtop)], -1, (0, 0, 255))   
                    original_gcontour = gsegmented + (gright, gtop)

                    tshape = gthresholded.shape
                    sshape = gsegmented.shape
                    new_segmented = original_gcontour.reshape(sshape[0], sshape[-1])
                    m = new_segmented.min(axis=0)
                    # check if the y axis lies on any of gui boxes
                    current_box = [box for box in gui_boxes if box[0] <= m[0] <= box[1]][0] 
                    current_mode = [mode for mode, pos in gui_modes_position.items() if current_box==pos][0]
                    
                    ## Change color on current box
                    ccolor = np.array([[5, 5, 8]]).astype(np.uint8)
                    sf = 0.2
                    if current_mode == previous_mode:
                        count_modes += 1
                        ccolor = count_modes * ccolor
                        sf  *= count_modes 
                    if count_modes > count_mode:
                        running_mode = current_mode
                        count_modes = 0
                    previous_mode = current_mode
                    
                    # to make animate like features, we will have to find part where cursor is on canvas
                    mind = modes.index(current_mode)
                    # to make 9 boxes, we need 10 llines on left and right
                    vboxes = np.linspace(0, vui.shape[1], len(icons)+1).astype(np.int64)
                    vboxes = [(vboxes[i], vboxes[i+1]) for i in range(len(icons))]
                    vbox = vboxes[mind]
                    vui_canvas_anim = vui_canvas.copy()
                    
                    ## ANim 1, change color##
                    # increase brightness?
                    vui_canvas_anim[:, vbox[0]:vbox[1]] += ccolor
                    
                    ## Anim 2 scale##
                    ### Scale down to up and repeat
                    icon_box = vui_canvas_anim[:, vbox[0]:vbox[1]].copy()
                    zeros_icon = np.zeros_like(icon_box)
                    icshape = icon_box.shape
                    icon_box = cv2.resize(icon_box.astype(np.uint8), (int(icshape[1]*sf), int(icshape[0]*sf)))
                    # take only part of new icon that fits to its original part
                    
                    
                    if sf > 1:
                        rd = int((icon_box.shape[0] - icshape[0])/2)
                        cd = int((icon_box.shape[1] - icshape[1])/2)
                        #print(icon_box.shape, icshape, rd, icon_box.shape[0]-rd, cd,icon_box.shape[1]-cd)
                        # edit here.....
                        zeros_icon[:, :] = icon_box[rd:icshape[0]+rd, cd:icshape[1]+cd] 
                    else:
                        rd = int((icshape[0] - icon_box.shape[0])/2)
                        cd = int((icshape[1] - icon_box.shape[1])/2)
                        
                        #print(icon_box.shape, icshape, rd, icon_box.shape[0]-rd, cd,icon_box.shape[1]-cd)
                        zeros_icon[rd:abs(icon_box.shape[0]-rd), cd:abs(icon_box.shape[1]-cd)] = icon_box[rd:abs(icon_box.shape[0]-rd), cd:abs(icon_box.shape[1]-cd)] 
                    
                    vui_canvas_anim[:, vbox[0]:vbox[1]] = zeros_icon
                    vui_canvas_anim[:, vbox[0]:vbox[1]] += ccolor
                    if vui_canvas is not None:
                        gshape = gray.shape
                        dummy = np.zeros_like(clone)
                        dummy_copy = dummy.copy()
                        dummy[:100, :] += vui_canvas_anim
                        cv2.circle(dummy_copy, (m[0], m[1]), 10, (250, 250, 150), -3)
                        d = dummy_copy[gtop:gbottom, gright:gleft]                                                
                        d = cv2.resize(d, (gshape[1], vshape[0]))                        
                        dummy_temp = dummy[:100, :]
                        dummy_temp[d!=[0, 0, 0]] = 100
                        dummy[:100, :] += dummy_temp
                        vui = dummy
                        
                    
                    cv2.circle(clone, (m[0], m[1]), 10, pointer_color, -3)
                
                if move_hand is not None:
                    mthresholded, msegmented = move_hand
                    sshape = msegmented.shape
                    new_segmented = msegmented.reshape(sshape[0], sshape[-1])
                    m = new_segmented.min(axis=0)
                    cv2.drawContours(clone, [msegmented+(mright,mtop)], -1, (0, 0, 255))   
                    if len(msegmented)>50:
                        if m[0]+mright > int((mright+mleft)/2):
                            running_mode = "draw"
                            vui[:100] = vui_canvas
                            vui[100:] = 0
                        else:
                            running_mode = "move"
                            vui[:100] = vui_canvas
                            vui[100:] = 0
                if draw_hand is not None:
                    dthresholded, dsegmented = draw_hand
                    cv2.drawContours(clone, [dsegmented+(right,top)], -1, (0, 0, 255))   

                    tshape = dthresholded.shape
                    sshape = dsegmented.shape
                    new_segmented = dsegmented.reshape(sshape[0], sshape[-1])
                    m = new_segmented.min(axis=0)

                    
                    # if pressed x, erase
                    if chr(key) == "x" or running_mode=="erase":
                        draw_color = (255, 255, 255)
                        pointer_color = (0, 0, 255)
                        erase = True
                    if chr(key) == "c" or running_mode=="draw":
                        draw_color = current_color
                        pointer_color = (255, 0, 0)
                        erase = False
                    #idle
                    if chr(key) == "z" or running_mode=="move":
                        erase = None
                        pointer_color = (0, 0, 0)                   
                    # restart system
                    if chr(key) == "r" or running_mode == "restart":
                        take_average=True
                        running_mode = "move"
                        canvas = None
                    if chr(key) == "e" or running_mode == "clear":
                        canvas = None
                        drawn = np.zeros(drawn.shape)+255
                        running_mode="move"
                    if chr(key) == "v" or running_mode == "color":
                        # add color region on VUI
                        # only use few portion of vui for dropdown
                        xs = vui.shape[0]-300
                        # what will be the length of each box on modes panel?
                        # since our colors mode lies on 1st box
                        ys = int(vshape[1]/len(icons)) * 2 - int(vshape[1]/len(icons))
                        # length of each box
                        c1 = int(vshape[1]/len(icons))
                        # col. num. of 1st box's end
                        c2 = c1 * 2
                        # for every row below the modes region up to xs, for only 1st box 
                        vui[100:100+xs, c1:c2] = cv2.resize(colors_holder, (ys, xs))
                        
                        # we don't want yet to draw so we will move cursor instead
                        erase = None
                        
                        # out of xs length on vui, how much region will a single color can hold?
                        cboxes = int(xs/len(colors))
                        
                        # take parts of rows from the draw ROI because our ROI for color lies on Draw ROI part. 
                        # if gshape[0] rows contains xs parts for colors then what must contain for bottom-top?
                        crb = int(xs/gshape[0] * (bottom-top))
                        
                        # divide that crb rows into num colors part
                        cboxes = np.linspace(0, crb, len(colors)+1).astype(np.int64)
                        # find each box's extreme columns on Draw ROI   
                        cboxes = [(cboxes[i], cboxes[i+1]) for i in range(len(cboxes)-1)]
                        #print(len(cboxes))
                        
                        # in which box is pointer now?
                        # check the rows, on which rows among colors ROI is pointer now?
                        cpointer = [cbox for cbox in cboxes if (cbox[0] < m[1] <= cbox[1])]
                        
                        # check the rows
                        cdb = (left-right)/len(icons) # draw box, divide original col draw ROI into num icons pcs
                        if  len(cpointer) > 0 and cdb<=m[0]<=cdb*2: # since color lies on 2nd box check if pointer is on that col
                            current_color = colors_array[cboxes.index(cpointer[-1])]
                            
                    if type(canvas) == type(None):
                        canvas = np.zeros((tshape[0], tshape[1], 3))+255
                    c = np.zeros(canvas.shape, dtype=np.uint8)
                    cv2.circle(c, (m[0], m[1]), 5, pointer_color, -3)
                    cv2.circle(clone, (right+m[0], top+m[1]), 5, pointer_color, -3)
                    #print(right+m[0], top+m[1])
                    if erase==True:
                        cv2.circle(canvas, (m[0], m[1]), 5, draw_color, -3)
                        erimg=cv2.circle(canvas.copy(), (m[0], m[1]), 5, (0, 0, 0), -3)            
                        cv2.circle(c, (m[0], m[1]), 5, (0, 0, 255), -3)
                        e = cv2.erode(erimg, (3, 3), iterations=5)
                        drawn = cv2.resize(e, (clone.shape[1], clone.shape[0]))
                        c = cv2.resize(c, (clone.shape[1], clone.shape[0]))
                        drawn_new = drawn+c
                        new_canvas = np.vstack([vui[:100], drawn_new]).astype(np.uint8)
                        cv2.imshow("Drawing", new_canvas)
                        #cv2.imshow("Drawing", drawn+c)
                        erimg=cv2.circle(canvas.copy(), (m[0], m[1]), 5, (255, 255, 255), -3)            
                        cv2.circle(c, (m[0], m[1]), 5, (0, 0, 255), -3)
                        e = cv2.erode(erimg, (3, 3), iterations=5)
                        drawn = cv2.resize(e, (clone.shape[1], clone.shape[0])) 
                        dc = drawn.astype(np.uint8)
                        #cv2.imshow("dc", drawn.astype(np.uint8))
                        #exception from here, xs not defined
                        try:
                            new_canvas[100:100+xs] += vui[100:100+xs]
                        except:
                            pass
                        #show(vui[100:100+xs])
                    elif erase==False:
                        cv2.circle(canvas, (m[0], m[1]), 5, draw_color, -3)
                        #print(mm)
                        e = cv2.erode(canvas, (3, 3), iterations=5)
                        drawn = cv2.resize(e, (clone.shape[1], clone.shape[0]))
                        c = cv2.resize(c, (clone.shape[1], clone.shape[0]))
                        drawn_new = drawn+c
                        new_canvas = np.vstack([vui[:100], drawn_new]).astype(np.uint8)
                        try:
                            new_canvas[100:100+xs] += vui[100:100+xs]
                        except:
                            pass
                        #show(vui[100:100+xs])
                        
                        cv2.imshow("Drawing", new_canvas)
                        dc = drawn.astype(np.uint8)
                        #cv2.imshow("D", drawn.astype(np.uint8))
                        #cv2.imshow("c", c)
                    elif erase == None:
                        canvas_shape = canvas.shape
                        clone_shape = clone.shape
                        eshape = (clone_shape[0]/canvas_shape[0], clone_shape[1]/canvas_shape[1])
                        m[0] = int(eshape[1]*m[0])
                        m[1] = int(eshape[0]*m[1])

                        if drawn is None:
                            e = cv2.erode(canvas, (3, 3), iterations=5)
                            drawn = cv2.resize(e, (clone.shape[1], clone.shape[0]))

                        drawn = cv2.resize(drawn, (clone.shape[1], clone.shape[0]))
                        dc = drawn.copy()  
                        cv2.circle(dc, (m[0], m[1]), 10, pointer_color, -3)
                        drawn_new = dc
                        new_canvas = np.vstack([vui[:100], drawn_new]).astype(np.uint8)
                        if running_mode == "color":
                            new_canvas[100:100+xs, c1:c2] = vui[100:100+xs, c1:c2]
                        #show(vui[100:100+xs])
                        
                        cv2.imshow("Drawing", new_canvas)
                        #cv2.imshow("Drawing", dc)

            if chr(key) == "s" or running_mode=="detect":
                if drawn is not None:
                    show(drawn)
                    d = drawn.copy().astype(np.uint8) 
                    r = recognition(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), 'show')
                    cv2.imshow("Detection", r)
                    running_mode="move"

            # draw a ROI for Draw
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(clone, str("GUI ROI"), (400, 140),
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.putText(clone, f"Curr. Mode: {running_mode}", (400, 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(clone, str("Move"), (int((mright)/1), int((mtop+mbottom)/2)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(clone, str("Draw"), (int((mleft + mright)/2), int((mtop+mbottom)/2)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(clone, (mleft, mtop), (int((mleft + mright)/2), mbottom), (0, 255, 0), 2)
            cv2.rectangle(clone, (int((mleft + mright)/2), mtop), (mright, mbottom), (0, 255, 0), 2)
            
            # draw a ROI boxes for GUI
            for i in range(len(gb_indices)-1):
                _gleft = gb_indices[i]
                _gright = gb_indices[i+1]
                cv2.rectangle(clone, (_gleft, gtop), (_gright, gbottom), gcolors[i], 3)
                cv2.putText(clone, modes[i][:2], (_gleft+2, int((gtop+gbottom)/2)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, gcolors[i], 2)
            # show live feed
            cv2.imshow("Feed", clone)
            if key == 32 or running_mode=="save":
                cv2.imwrite("Captured.png", dc)
                cv2.imshow("captured", dc)
                running_mode = "move"
            # if pressed  escape, loop out and stop processing
            if running_mode=="exit" or key==27:
                break
    cam.release()
    cv2.destroyAllWindows()
run_system(avg_frames=30)