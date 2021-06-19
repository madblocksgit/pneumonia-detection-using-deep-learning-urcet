from keras.models import load_model
import cv2

import numpy as np

model = load_model('model.h5')

model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
img = cv2.imread('mad_p5.jpeg',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(150,150))
img = np.reshape(img,[-1,150,150,1])
classes = model.predict_classes(img)
classes = classes.reshape(1,-1)[0]
print (classes)

img=cv2.imread('mad_normal2.jpeg',cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img,(150,150))
img=np.reshape(img,[-1,150,150,1])
classes=model.predict_classes(img)
classes=classes.reshape(1,-1)[0]
print(classes)
