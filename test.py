import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

img = cv2.imread('images/testingHappyBoy.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
print("You should see me")
analysis = DeepFace.analyze(img, actions=["emotion"])
print(analysis)

##emotion = predictions['dominant_emotion']
##txt = str(emotion)
##print(txt)

##CODE IS UNUSABLE AT THE MOMENT | RUN CHATBOT.PY FOR CURRENT BUILD