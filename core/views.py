from django.shortcuts import render,redirect
from .models import Profile,LastFace
from .forms import ProfileForm
from django.db.models import Q
from django.http import HttpResponse
import numpy as np
import cv2
import os
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
import winsound


# Create your views here.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def index(request):
    scanned = LastFace.objects.all().order_by('date').reverse()
    present = Profile.objects.filter(present=True).order_by('updated').reverse()
    absent = Profile.objects.filter(present=False).order_by('shift')
    context = {
        'scanned': scanned,
        'present': present,
        'absent': absent,
    }
    return render(request, 'core/index.html', context)

def profiles(request):
    profiles = Profile.objects.all()
    context = {
        'profiles': profiles
    }
    return render(request, 'core/profiles.html', context)

from django.shortcuts import render
from .models import LastFace, Profile

# def details(request):
#     try:
#         last_face = LastFace.objects.last()
#         recognized_name = last_face.last_face

#         # Search for the profile with the recognized name
#         profile = Profile.objects.get(Q(first_name__icontains=recognized_name.split()[0]) &
#                                       Q(last_name__icontains=recognized_name.split()[1]))
#     except Profile.DoesNotExist:
#         last_face = None
#         profile = None

#     context = {
#         'profile': profile,
#         'last_face': last_face
#     }
#     return render(request, 'core/details.html', context)

def details(request):
    last_face = LastFace.objects.last()
    profile = None  # Initialize profile as None

    if last_face is not None:
        recognized_name = last_face.last_face

        try:
            # Search for the profile with the recognized name
            profile = Profile.objects.get(Q(first_name__icontains=recognized_name.split()[0]) &
                                          Q(last_name__icontains=recognized_name.split()[1]))
        except Profile.DoesNotExist:
            pass  # Profile not found, profile remains None

    context = {
        'profile': profile,
        'last_face': last_face
    }
    return render(request, 'core/details.html', context)


def add_profile(request):
    form = ProfileForm()
    if request.method == 'POST':
        form = ProfileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('profiles')
    context = {'form':form}
    return render(request,'core/add_profile.html',context)

def edit_profile(request,id):
    profile = Profile.objects.get(id=id)
    form = ProfileForm(instance = profile)
    if request.method == 'POST':
        form = ProfileForm(request.POST,request.FILES,instance=profile)
        if form.is_valid():
            form.save()
            return redirect('profiles')
    context = {'form':form}
    return render(request,'core/add_profile.html',context)

def delete_profile(request,id):
    profile = Profile.objects.get(id=id)
    profile.delete()
    return redirect('profiles')

def reset(request):
    profiles = Profile.objects.all()
    for profile in profiles:
        if profile.present == True:
            profile.present = False
            profile.save()
        else:
            pass
    return redirect('index')

def clear_history(request):
    history = LastFace.objects.all()
    history.delete()
    return redirect('index')

# sound = os.path.join(BASE_DIR,'core','sound','beep.wav')
# last_face = ""
# def scan(request):
#     global last_face
    
#     face_cascade_path = os.path.join(BASE_DIR,'core','haarcascades','haarcascade_frontalface_default.xml')
#     face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
#     known_face_names = [f'{profile.image}'[:-4] for profile in Profile.objects.all()]

#     # Start capturing video from the webcam
#     video_capture = cv2.VideoCapture(0)

#     # Loop to continuously process video frames
#     while True:
#         ret, frame = video_capture.read()

#         # Convert frame to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect faces in the frame
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         # Iterate through detected faces
#         for (x, y, w, h) in faces:
#             face_name = None

#             # Compare detected face with known face names
#             for known_name in known_face_names:
#                 if known_name in frame:
#                     face_name = known_name
#                     break

#             if face_name:
#                 # Update the presence status of recognized faces
#                 profile = Profile.objects.get(Q(image__icontains=face_name))
#                 if profile.present == True:
#                     pass
#                 else:
#                     profile.present = True
#                     profile.save()

#                 # Save last recognized face and play a sound
#                 if last_face != face_name:
#                     last_face = face_name
#                     last_face_obj = LastFace(last_face=last_face)
#                     last_face_obj.save()
#                     winsound.PlaySound(sound, winsound.SND_ASYNC)
#                 else:
#                     pass

#                 # Draw rectangle and text for recognized face
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#                 cv2.putText(frame, face_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#         # Display the video frame
#         cv2.imshow('Video', frame)

#         # Break the loop if the Enter key is pressed
#         if cv2.waitKey(1) & 0xFF == 13:
#            video_capture.release()
#            cv2.destroyAllWindows()
#            return JsonResponse({'message':'Scanner closed','last_face': last_face})   

    # Release the video capture and close the OpenCV windows
    #video_capture.release()
    #cv2.destroyAllWindows()

    # Return a response indicating that the scanner is closed
    #return HttpResponse('Scanner closed', last_face)
# def scan(request):
#     global last_face
#     sound = os.path.join(BASE_DIR, 'core', 'sound', 'beep.wav')
    
#     face_cascade_path = os.path.join(BASE_DIR, 'core', 'haarcascades', 'haarcascade_frontalface_default.xml')
#     face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
#     # Start capturing video from the webcam
#     video_capture = cv2.VideoCapture(0)
    
#     recognized_employees = []

#     while True:
#         ret, frame = video_capture.read()

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        

#         for (x, y, w, h) in faces:
#             face_roi = gray[y:y+h, x:x+w]
#             face_name = None
            
#             # Compare detected face with known face names
#             for profile in Profile.objects.all():
#                 if f'{profile.image}'[:-4] in face_roi:
#                     face_name = f'{profile.first_name} {profile.last_name}'
#                     recognized_employees.append({
#                         'name': face_name,
#                         'profession': profile.profession,
#                         'ranking': profile.ranking
#                     })
#                     break
            
#             if face_name:
#                 # Update the presence status of recognized faces
#                 profile = Profile.objects.get(image__icontains=face_name)
#                 if not profile.present:
#                     profile.present = True
#                     profile.save()
                
#                 # Play a sound and update last recognized face
#                 if last_face != face_name:
#                     last_face = face_name
#                     last_face_obj = LastFace(last_face=last_face)
#                     last_face_obj.save()
#                     winsound.PlaySound(sound, winsound.SND_ASYNC)

#                 # Draw rectangle and text for recognized face
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#                 cv2.putText(frame, face_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#         # Display the video frame
#         cv2.imshow('Video', frame)

#         # Break the loop if the Enter key is pressed
#         if cv2.waitKey(1) & 0xFF == 13:
#             video_capture.release()
#             cv2.destroyAllWindows()
#             return JsonResponse({'message': 'Scanner closed', 'recognized_employees': recognized_employees})

# def scan(request):
#     global last_face
#     sound = os.path.join(BASE_DIR, 'core', 'sound', 'beep.wav')
    
#     face_cascade_path = os.path.join(BASE_DIR, 'core', 'haarcascades', 'haarcascade_frontalface_default.xml')
#     face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    
#     # Start capturing video from the webcam
#     video_capture = cv2.VideoCapture(0)
    
#     recognized_employees = []
    
#     last_face = ""

#     while True:
#         ret, frame = video_capture.read()

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#         print("Number of detected faces:", len(faces))
        
#         for (x, y, w, h) in faces:
#             face_roi = gray[y:y+h, x:x+w]
#             face_name = None
#             print('Detected face ROI:',face_roi)            # Compare detected face with known face names
#             for profile in Profile.objects.all():
#                 profile_name = f'{profile.first_name} {profile.last_name}'
#                 print("comparing with profile name:",profile_name)
#                 if profile_name in face_roi:
#                     face_name = profile_name
#                     recognized_employees.append({
#                         'name': face_name,
#                         'profession': profile.profession,
#                         'ranking': profile.ranking
#                     })
#                     print("Match found!")
#                     break
            
#             if face_name:
#                 # Update the presence status of recognized faces
#                 profile = Profile.objects.get(first_name=profile.first_name, last_name=profile.last_name)
#                 if not profile.present:
#                     profile.present = True
#                     profile.save()
                
#                 # Play a sound and update last recognized face
#                 if last_face != face_name:
#                     last_face = face_name
#                     last_face_obj = LastFace(last_face=last_face)
#                     last_face_obj.save()
#                     winsound.PlaySound(sound, winsound.SND_ASYNC)

#                 # Draw rectangle and text for recognized face
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#                 cv2.putText(frame, face_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#         # Display the video frame
#         cv2.imshow('Video', frame)

#         # Break the loop if the Enter key is pressed
#         if cv2.waitKey(1) & 0xFF == 13:
#             video_capture.release()
#             cv2.destroyAllWindows()
#             print("scanner closed")
#             print("Recognized employees:",recognized_employees)
#             return JsonResponse({'message': 'Scanner closed', 'recognized_employees': recognized_employees})

def scan(request):
    global last_face
    sound = os.path.join(BASE_DIR, 'core', 'sound', 'beep.wav')
    
    face_cascade_path = os.path.join(BASE_DIR, 'core', 'haarcascades', 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Create the LBPHFaceRecognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Train the recognizer with profile images and labels
    profiles = Profile.objects.all()
    faces = []
    labels = []
    for profile in profiles:
        profile_image = cv2.imread(profile.image.path, cv2.IMREAD_GRAYSCALE)
        faces.append(profile_image)
        labels.append(profile.id)
    face_recognizer.train(faces, np.array(labels))
    
    # Start capturing video from the webcam
    video_capture = cv2.VideoCapture(0)
    
    recognized_employees = []
    
    last_face = ""

    while True:
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print("Number of detected faces:", len(faces))
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_id, confidence = face_recognizer.predict(face_roi)
            print('Detected face ROI:', face_roi)
            print("Predicted face ID:", face_id, "Confidence:", confidence)
    
            if confidence < 100:
                profile = Profile.objects.get(id=face_id)
                face_name = f'{profile.first_name} {profile.last_name}'
                recognized_employees.append({
                    'name': face_name,
                    'profession': profile.profession,
                    'ranking': profile.ranking
                })
                print("Match found:", face_name)
                
                # Update the presence status of recognized faces
                if not profile.present:
                    profile.present = True
                    profile.save()
                
                # Play a sound and update last recognized face
                if last_face != face_name:
                    last_face = face_name
                    last_face_obj = LastFace(last_face=last_face)
                    last_face_obj.save()
                    winsound.PlaySound(sound, winsound.SND_ASYNC)

                # Draw rectangle and text for recognized face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, face_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                print("No match found.")
                
        # Display the video frame
        cv2.imshow('Video', frame)

        # Break the loop if the Enter key is pressed
        if cv2.waitKey(1) & 0xFF == 13:
            video_capture.release()
            cv2.destroyAllWindows()
            print("Scanner closed")
            print("Recognized employees:", recognized_employees)
            return JsonResponse({'message': 'Scanner closed', 'recognized_employees': recognized_employees})


def ajax(request):
    last_face_obj = LastFace.objects.last()
    last_face_value = last_face_obj.last_face if last_face_obj else None
    context = {
        'last_face': last_face_value
    }
    return JsonResponse (context)