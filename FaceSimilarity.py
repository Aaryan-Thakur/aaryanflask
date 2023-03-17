from flask import Flask, request, render_template
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

handler = insightface.model_zoo.get_model('model.onnx') 
handler.prepare(ctx_id=0) 


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded files from the user
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        # Save the uploaded files to disk
        file1_path = 'uploads/' + file1.filename
        file2_path = 'uploads/' + file2.filename
        file1.save(file1_path)
        file2.save(file2_path)
        
        fa = FaceAnalysis() #This is used to DETECT faces
        fa.prepare(ctx_id=0,det_thresh=0.5, det_size=(640, 640)) 


        img1 = cv2.imread(file1_path) #First Image 
        img2 = cv2.imread(file2_path) #Second Image
        
        faces1 = fa.get(img1) # Get the face mappings of faces in the First Image
        faces2 = fa.get(img2) # Get the face mappings of faces in the Second Image
        
        emb1 = handler.get(img1,faces1[0]) # Get the facial features of face in the First Image
        emb2 = handler.get(img2,faces2[0]) # Get the facial features of face in the Second Image
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))*100


        # Display the results to the user
        return render_template('result.html', sim=similarity)
    
    # If the request method is GET, display the upload form to the user
    return render_template('index.html')

@app.route('/camera')
def camera():
        cap = cv2.VideoCapture(0)
        fa = FaceAnalysis() #This is used to DETECT faces
        fa.prepare(ctx_id=0,det_thresh=0.5, det_size=(640, 640)) 
        while True:
            ret,frame = cap.read()
            faces = fa.get(frame)
            if faces:
                print("Face detected. Picture taken.")
                break
        return render_template('camera.html')


if __name__ == '__main__':
    app.run(debug=True)
