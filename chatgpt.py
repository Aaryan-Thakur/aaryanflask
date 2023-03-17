from flask import Flask, request,Response, render_template
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

handler = insightface.model_zoo.get_model('model.onnx') 
handler.prepare(ctx_id=0) 




app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # If the request method is GET, display the upload form to the user
    return render_template('index2.html')

@app.route('/video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
def gen_frames():
    cap = cv2.VideoCapture(0)
    fa = FaceAnalysis() #This is used to DETECT faces
    fa.prepare(ctx_id=0,det_thresh=0.5, det_size=(320, 240)) 
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        face = fa.get(frame)
        
        if(len(face)!=1):
            frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue
        
        bbox = face[0].bbox
        

        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        # Convert the frame to bytes and yield it
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)
