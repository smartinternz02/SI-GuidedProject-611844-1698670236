import numpy as np
import os
import tensorflow as tf
from PIL import Image 
from flask import Flask,render_template,request,jsonify,url_for,redirect
from tensorflow.keras.preprocessing.image import load_img,img_to_array

app=Flask(__name__)
model=tf.keras.models.load_model('dogbreed_vgg19.h5')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/header')
def header():
    return render_template("header.html")
# @app.route('/predict',methods=['GET','POST'])

# def predict():
#     return render_template("predict.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/output',methods=['GET','POST'])

def output():
    if request.method=='POST':
        f=request.files['img']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=load_img(filepath,target_size=(224,224))

        image_array=np.array(img)

        image_array=np.expand_dims(image_array,axis=0)

        pred=np.argmax(model.predict(image_array),axis=1)
        # index=['affenpinscher','beagle','appenzeller','basset','buletick','boxer','cairn','doberman','german_shepherd','golden_retriever']
        index = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff	', 'cairn	', 'cardigan', 'chesapeake_bay_retriever ', 'chihuahua	', 'chow	', 'clumber', 'cocker_spaniel', 'collie	', 'curly-coated_retriever', 'dandie_dinmont	', 'dhole	', 'dingo	', 'doberman', 'english_foxhound', 'english_setter	', 'english_springer	', 'entlebucher	', 'eskimo_dog	', 'flat-coated_retriever', 'french_bulldog	', 'german_shepherd	', 'german_short-haired_pointer', 'giant_schnauzer	', 'golden_retriever	', 'gordon_setter	', 'great_dane	', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael	', 'ibizan_hound	', 'irish_setter	', 'irish_terrier	', 'irish_water_spaniel', 'irish_wolfhound	', 'italian_greyhound	', 'japanese_spaniel	', 'keeshond	', 'kelpie	', 'kerry_blue_terrier', 'komondor	', 'kuvasz	', 'labrador_retriever', 'lakeland_terrier	', 'leonberg	', 'lhasa	', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle	', 'miniature_schnauzer', 'newfoundland	', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier	', 'old_english_sheepdog', 'otterhound	', 'papillon	', 'pekinese	', 'pembroke	', 'pomeranian', 'pug	', 'redbone', 'rhodesian_ridgeback', 'rottweiler	', 'saint_bernard	', 'saluki	', 'samoyed	', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier	', 'shetland_sheepdog	', 'shih-tzu	', 'siberian_husky', 'silky_terrier	', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier	', 'standard_poodle	', 'standard_schnauzer', 'sussex_spaniel	', 'tibetan_mastiff	', 'tibetan_terrier	', 'toy_poodle	', 'toy_terrier	', 'vizsla	', 'walker_hound', 'weimaraner	', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet	', 'wire-haired_fox_terrier', 'yorkshire_terrier']
        
        predicition=index[int(pred)]
        print("predicition")
        return render_template("output.html",prediction_text=predicition)


    

if __name__ == "__main__":
    app.run(debug=True)
