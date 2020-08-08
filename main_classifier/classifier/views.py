from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings
import os

from .forms import ClassifierForm
from .models import Classifier

# import for deep learning
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2



# Path to input image
media_path = os.path.join(os.path.dirname(settings.BASE_DIR), 'media_cdn/images')

# Model names
smile_model= './models/trained_model.h5'
facemask_model= './models/model.h5'

# Create your views here.

def index(request):
    return render(request, 'classifier/index.html')


def upload_img(request):
    form = ClassifierForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        m = Classifier()
        m.image = form.cleaned_data['image']
        print(type(form.cleaned_data['image']))
        print("TYPE: " + form.cleaned_data['category'])
        m.save()
        
        category = form.cleaned_data['category']
        return HttpResponseRedirect('/classifier/predict/?category=' + category)

    return render(request, 'classifier/upload_img.html', {'form': form})


def predict(request):
    # Preprocess image
    img_path = os.path.join(media_path, os.listdir(media_path)[0])

    category = request.GET.get('category')
    print("IMAGE PATH: " + img_path)
    print("CATEGORY: " + request.GET.get('category'))

    if category == 'Face Mask':
        #Predict for Pnuemonia
        img = preprocess_img(img_path, category='Face Mask')
        model = load_model(facemask_model)
        
        (mask, withoutMask) = model.predict(img)[0]

       
        
        label = int(1) if mask > withoutMask else int(0)
        
        label = get_label_name(label, "Face Mask")
       
        context = {
            'label': label,
            'imagepath': img_path
        }
        

    else:
        #Predict for Pnuemonia
        img = preprocess_img(img_path, category='Smile')
        model = load_model(smile_model)
        print("Making Predictions.....")

        (NotSmiling, Smiling) = model.predict(img)[0]
       
        
        label = int(1) if Smiling > NotSmiling else int(0)
      

        label = get_label_name(label, "Smile")
      
        context = {
            'label': label,
            'imagepath': img_path
        }
        

    return render(request, 'classifier/result.html', context)

def preprocess_img(img, category):
    image = cv2.imread(img)
    
    
    if category == 'Face Mask':

        face = cv2.resize(image,(224,224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        
    else:
        
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(image,(28,28))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)


    return face


def get_label_name(label, category):
    if category == 'Face Mask':
        if label == 1:
            return "Mask"
        else:
            return "No Mask"
    else:
        if label == 1:
            return "Smiling"
        else:
            return "Not Smiling"


def clean_path(request):
    '''Cleans up image path'''
    # Delete image instance from model
    Classifier.objects.all().delete()

    # Delete image from media directory
    for img in os.listdir(media_path):
        os.remove(os.path.join(media_path, img))

    return HttpResponseRedirect('/')


