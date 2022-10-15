# from asyncio.format_helpers import extract_stack
# from contextlib import redirect_stderr
# from fileinput import filename
# from importlib.metadata import files
# from wsgiref.util import request_uri
import tensorflow as tf 
from flask import Flask,request,render_template,redirect
import librosa
import librosa.display
import numpy as np
import soundfile
from scipy.io.wavfile import write


app = Flask(__name__)

loaded_model_cnn = tf.keras.models.load_model('model_cnn.h5')


def extract_features(file,ZCR,stft,mfcc,rms,mel):            
    with soundfile.SoundFile(file) as sound_file:
        data = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        # ZCR
        result = np.array([])
        if ZCR:
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result=np.hstack((result, zcr)) # stacking horizontally

        # Chroma_stft
        if stft:
            stft = np.abs(librosa.stft(data))
            chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma_stft)) # stacking horizontally

        # MFCC
        if mfcc:
            mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc)) # stacking horizontally

        # Root Mean Square Value
        if rms:
            rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        result = np.hstack((result, rms)) # stacking horizontally

        # MelSpectogram
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel)) # stacking horizontally
        
    return result

emotions={
  1:'neutral',
  2:'calm',
  3:'surprised',
  4:'angry',
  5:'sad',
  6:'fearful',
  7:'disgust',
  8:'happy'
}

@app.route('/predict',methods=["GET","POST"])
def predict(file):
        ans =[]
        new_feature  = extract_features(file,ZCR=True,stft=True,mfcc=True,rms=True,mel=True)
        
        ans.append(new_feature)
        ans = np.array(ans)
        prediction = loaded_model_cnn.predict([ans])
        return emotions[np.argmax(prediction[0])+1]


@app.route("/",methods=["GET","POST"])
def index():
    prediction = ""
    if request.method == "POST":
        print("Data received.")

        file = request.files["file"]

        if "file" not in request.files:
            return redirect(request.url)


        if file.filename =="":
            return redirect(request.url)

        if file:
            if request.method == 'POST':
               
                file = request.files['file']
                
                file_path = "static/" + file.filename

                
                file.save(file_path)
                
                prediction = "The predicted Emotion is : " + predict(file_path)

    return render_template('index.html',prediction_html=prediction)
                    


if __name__ == "__main__":

    app.run(debug=True,host='0.0.0.0',port=9999,threaded=True)
