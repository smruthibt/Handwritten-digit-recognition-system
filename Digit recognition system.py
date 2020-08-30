from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#Load the data and splitting it into train and test set

(X_train,y_train), (X_test,y_test) = mnist.load_data()


"""#To take a look at the first image (at index=0) in the training dataset,and to show the image as apicture
plt.imshow(X_train[0])"""

#We reshape and process the data to fit the model
X_train=X_train.reshape(60000,28,28,1)
X_test=X_test.reshape(10000,28,28,1)

#OneHotEncoding
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

"""#If we print the new label
print(y_train_one_hot[0])"""

#build the cnn model
model=Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))

#Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#train
hist = model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot),epochs=10)

#To visualise the model's accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc='upper left')
plt.show()

#show predictions as probablilties for first four images in the test set
predictions = model.predict(X_test[:4])
#print predictions as number labels for the first 4 images
print(np.argmax(predictions, axis=1))

#print the pictures is images
for i in range(0,4):
    image=X_test[i]
    image=np.array(image, dtype='float')
    pixels=image.reshape((28,28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


#print the actual labels
print(y_test[:4])

















