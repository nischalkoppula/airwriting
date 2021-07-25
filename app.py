from flask import Flask,make_response, render_template, request, redirect, url_for, Response
from avg import *
from webcolors import name_to_rgb
app = Flask(__name__)

from pynput import keyboard
s = VideoCamera()
def on_press(key):
    global s
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys
    if (k=='space'):  # keys of interest
        # self.keys.append(k)  # store it in global-like variable
        
        s.draw = not s.draw
    if (k=="backspace"):
        s.er = not s.er
          # stop listener; remove this if want more keys
listener = keyboard.Listener(on_press=on_press)
listener.start()
def gen(camera):
    canvas = None
    bg_img=None
    aweight = 0.5
    # strat the camera
    cam = cv2.VideoCapture(0)
    # ROI box
    top, right, bottom, left = 250, 400, 480, 640
    # count frame
    num_frames=0
    # writing canvas
    canvas = None
    # thickness
    t=5
    # draw color(ink color)
    draw_color = (0, 255, 0)
    # pointer color
    pointer_color = (255, 0, 0)
    # mode flag
    erase = False
    # flag to indicate take average
    take_average=True
    #bg image
    bg_img=None
    data = {}
    data["aweight"] = aweight
    data["num_frames"] = 0
    data["t"] = t
    data["draw_color"] = draw_color
    data["pointer_color"] = pointer_color
    while True:
        frame = camera.get_frame(canvas,bg_img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def sc_data_collect(camera):
    aweight = 0.5
    take_average=True
    bg_img=None
    canvas = None
    while True:
        if(camera.num_frames<100):
            num_frames,bg_img,clone = camera.get_frame(aweight,take_average)
        else:
            canvas,clone = camera.write()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + clone + b'\r\n')


@app.route('/',methods=['GET', 'POST'])
def home1():
    global s
    if request.method == 'POST':
        thickness = request.form["thick"]
        color = request.form["colors"]
        p_c = request.form["pointer_color"]
        dc_m = name_to_rgb(color)
        pc_m = name_to_rgb(p_c)
        s.draw_color = (dc_m.blue,dc_m.green,dc_m.red)
        s.pointer_color = (pc_m.blue,pc_m.green,pc_m.red)
        s.t = int(thickness)
    return render_template("open_page.html")

@app.route('/viewcamera')
def viewcamera():

    return Response(sc_data_collect(s),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/close',methods=['GET', 'POST'])
def close():
    global s
    s.close()
    return render_template("out.html")


if __name__ == '__main__':
    app.run()