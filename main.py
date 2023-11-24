import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

with open('PP.pkl', 'rb') as file:
    PP = pickle.load(file)
with open('BP.pkl', 'rb') as file:
    BP = pickle.load(file)

LS = load_model('LS.keras')
DP = load_model('DP.keras')
CN = load_model('CN.keras')
RN = load_model('RN.keras')

new_review_text = "songs are of and and is morality it's her or know would care i i br screen that obvious plot actors new would with paris not have attempt lead or of too would local that of every their it coming this and of information to and br and movie was and that film is under by left this and is entertainment ok this in own be house of sticks worker in bound my i i obviously sake things just as lost lot br comes never like thing start of obviously comes indeed coming want no bad than history from lost comes accidentally young to movie bad facts dream from reason these honor movie elizabeth it's movie so fi and enough to computer duo film and almost jeffrey rarely obviously and alive to appears i i only human it and just only hop to be hop new made comes evidence blues high in want to other blues of their for and those i'm and that and obviously message obviously obviously for and of and brother br and make and lit and this and of blood br andy worst and it and this across as it when lines that make excellent scenery that there is julia fantasy to and and film good br of loose and basic have into your whatever i i and and demented be hop this standards cole new be home all seek film wives lot br made and in at this of search how concept in thirty some this and not all it rachel are of boys and re is and animals deserve i i worst more it is renting concerned message made all and in does of nor of nor side be and center obviously know end computer here to all tries in does of nor side of home br be indeed i i all it officer in could is performance and fully in of and br by br and its and lit well of nor at coming it's it that an this obviously i i this as their has obviously bad and exist countless and mixed of and br work to of run up and and br dear nor this early her bad having tortured film and movie all care of their br be right acting i i and of and and it away of its shooting and to suffering version you br and your way just and was can't compared condition film of and br united obviously are up obviously not other just and was and as true was least of and certainly lady poorly of setting produced and br and to make just have 2 which and of and dialog and br of and say in can is you for it wasn't in and as by it away plenty what have reason and are that willing that's have 2 which sister and of important br halfway to of took work 20 br similar more he good and for hit at coming not see reputation "

max_review_length = 500

word_to_index = imdb.get_word_index()
new_review_tokens = [word_to_index.get(word, 0) for word in new_review_text.split()]
new_review_tokens = pad_sequences([new_review_tokens], maxlen=max_review_length)

# Make the prediction
prediction = PP.predict(new_review_tokens)

print(prediction)

if prediction>0.3:
    print('Positive')
else:
    print("Negative")

def make_prediction(img,model):
    img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    if res:
        print("Tumor Detected")
    else:
        print("No Tumor")

#make_prediction(r'C:\Users\shahi\Desktop\My Projects\DeepPredictorHub\CNN\tumordata\pred\pred3.jpg',CN)

    