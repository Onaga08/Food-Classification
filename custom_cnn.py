import os
import numpy as np 
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# Set paths
parent_dir = 'food-101' 
train_file = os.path.join(parent_dir, 'meta', 'train.txt')
test_file = os.path.join(parent_dir, 'meta', 'test.txt')  
img_dir = os.path.join(parent_dir, 'images')

# Fixed params    
num_classes = 101    
input_shape = (64, 64, 3) 
batch_size = 64
epochs = 15

# Load and preprocess data
def load_data(txt_file):
    
    X = []
    y = []
    with open(txt_file) as file:
        for line in file:
            cls, img_name = line.strip().split('/')   
            img_path = os.path.join(img_dir, cls, img_name+'.jpg') 
            img = cv2.imread(img_path)
            img = cv2.resize(img, input_shape[:2])
            X.append(img)
            y.append(cls)
            
    X = np.array(X)/255.0
    y = to_categorical(y, num_classes)   
    return X,y
    
X_train, y_train = load_data(train_file) 
X_test, y_test = load_data(test_file)   

# Train test split on training data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1) 

# Model definition and training  
model = Sequential()
model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(64, 64, 3)))    
model.add(MaxPooling2D())

model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, 3, padding='same', activation='relu')) 
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size)

# Evaluate on test set  
y_pred = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Plot training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()