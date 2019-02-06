import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import winsound
#.darkflow.net.build
#from app import app
# enable debugging

#Testing class

options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',#'cfg/yolov2-tiny.cfg',#'cfg/yolov2.cfg',#
    'load': 'bin/yolov2-tiny.weights',#'bin/yolov2.weights',
    #REPLACE WITH GENERATED PEDESTRIAN WEIGHTS
    'thre`shold': 0.5045, #low for weaker model, higher for stronger model
    'gpu': 0.0
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

#Video9114
#pedtest.MP4
capture = cv2.VideoCapture(0)
#"http://10.0.0.134:8081"
#capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

count = 0

while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:

        results = tfnet.return_predict(frame)
        print(results)

        if(len(results) != 0):
            if(results):
                winsound.PlaySound('./sound/pop.wav', winsound.SND_FILENAME)
        #draw prediction boxes

        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, (0, 255, 0), 15)
            #frame = cv2.putText(
            #frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)

        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
