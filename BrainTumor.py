
from tkinter import messagebox
from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import pickle

main = tkinter.Tk()
main.title("Identifying Brain Tumor using X-Ray Images") #designing main screen
main.geometry("1300x1200")

global filename
global accuracy
X = []
Y = []
global classifier
disease = ['No Tumor Detected','Tumor Detected']
classifier = None

with open('Model/segmented_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    segmented_model = model_from_json(loaded_model_json)
json_file.close()    
segmented_model.load_weights("Model/segmented_weights.h5")

def edgeDetection():
    img = cv2.imread('myimg.png')
    orig = cv2.imread('test1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    min_area = 0.95*180*35
    max_area = 1.05*180*35
    result = orig.copy()
    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(result, [c], -1, (0, 0, 255), 10)
        if area > min_area and area < max_area:
            cv2.drawContours(result, [c], -1, (0, 255, 255), 10)
    return result    

def tumorSegmentation(filename):
    global segmented_model
    img = cv2.imread(filename,0)
    img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(1,64,64,1)
    img = (img-127.0)/127.0
    preds = segmented_model.predict(img)
    preds = preds[0]
    print(preds.shape)
    orig = cv2.imread(filename,0)
    orig = cv2.resize(orig,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("test1.png",orig)    
    segmented_image = cv2.resize(preds,(300,300),interpolation = cv2.INTER_CUBIC)
    segmented_image = (segmented_image * 255).astype(np.uint8)
    cv2.imwrite("myimg.png",segmented_image)
    edge_detection = edgeDetection()
    return segmented_image*255, edge_detection
    

def uploadDataset(): #function to upload dataset
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def datasetPreprocessing():
    global X
    global Y
    X.clear()
    Y.clear()
    if os.path.exists('Model/myimg_data.txt.npy'):
        X = np.load('Model/myimg_data.txt.npy')
        Y = np.load('Model/myimg_label.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename+"/no"):
            for i in range(len(directory)):
                name = directory[i]
                img = cv2.imread(filename+"/no/"+name,0) #reading images
                ret2, th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU) #processing and normalization images
                img = cv2.resize(img, (128,128)) #resizing images
                im2arr = np.array(img) #extract features from images
                im2arr = im2arr.reshape(128,128,1)
                X.append(im2arr)
                Y.append(0)
                print(filename+"/no/"+name)

        for root, dirs, directory in os.walk(filename+"/yes"):
            for i in range(len(directory)):
                name = directory[i]
                img = cv2.imread(filename+"/yes/"+name,0)
                ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                img = cv2.resize(img, (128,128))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(128,128,1)
                X.append(im2arr)
                Y.append(1)
                print(filename+"/yes/"+name)
                
        X = np.asarray(X)
        X = X.astype(np.float32) / 255.0
        Y = np.asarray(Y)            
        np.save("Model/myimg_data.txt",X)
        np.save("Model/myimg_label.txt",Y)
    print(X.shape)
    print(Y.shape)
    print(Y)
    cv2.imshow('ss',X[20])
    cv2.waitKey(0)
    text.insert(END,"Total number of images found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total number of classes : "+str(len(set(Y)))+"\n\n")
    text.insert(END,"Class labels found in dataset : "+str(disease))       
        
 
def trainTumorDetectionModel():
    global accuracy
    global classifier

    YY = to_categorical(Y)

    # Shuffle dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    x_train = X[indices]
    y_train = YY[indices]

    # Split into training and testing sets
    X_trains, X_tests, y_trains, y_tests = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=0
    )

    if os.path.exists('Model/model.json'):
        # Load existing model
        with open('Model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()

        classifier = model_from_json(loaded_model_json)
        classifier.load_weights("Model/model_weights.h5")

        classifier.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    else:
        # Build CNN
        classifier = Sequential()

        classifier.add(
            Conv2D(
                filters=32,
                kernel_size=(3,3),
                activation='relu',
                input_shape=(128,128,1)
            )
        )

        classifier.add(MaxPooling2D(pool_size=(2,2)))

        classifier.add(
            Conv2D(
                filters=32,
                kernel_size=(3,3),
                activation='relu'
            )
        )

        classifier.add(MaxPooling2D(pool_size=(2,2)))

        classifier.add(Flatten())

        classifier.add(Dense(128, activation='relu'))

        classifier.add(Dense(2, activation='softmax'))

        print(classifier.summary())

        classifier.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train using ONLY training data
        hist = classifier.fit(
            X_trains,
            y_trains,
            validation_data=(X_tests, y_tests),
            batch_size=16,
            epochs=10,
            shuffle=True,
            verbose=2
        )

        # Save model
        classifier.save_weights('Model/model_weights.h5')

        model_json = classifier.to_json()

        with open("Model/model.json", "w") as json_file:
            json_file.write(model_json)

        # Save training history
        with open('Model/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)

    # Evaluate model on TEST data
    loss, test_accuracy = classifier.evaluate(
        X_tests,
        y_tests,
        verbose=0
    )

    accuracy = test_accuracy * 100

    print(f"\nTest Accuracy : {accuracy:.2f}%")

    # Display in GUI
    text.insert(
        END,
        "\n\nCNN Brain Tumor Model Generated. See black console to view layers of CNN\n\n"
    )

    text.insert(
        END,
        "CNN Brain Tumor Prediction Accuracy on Test Images : {:.2f}%\n".format(accuracy)
    )

    # Optional: show training & validation accuracy
    if os.path.exists("Model/history.pckl"):
        with open("Model/history.pckl", "rb") as f:
            history = pickle.load(f)

        if "accuracy" in history:
            text.insert(
                END,
                "Final Training Accuracy : {:.2f}%\n".format(
                    history["accuracy"][-1] * 100
                )
            )

        if "val_accuracy" in history:
            text.insert(
                END,
                "Final Validation Accuracy : {:.2f}%\n".format(
                    history["val_accuracy"][-1] * 100
                )
            )
       


def tumorClassification():
    global classifier

    # Check whether model is loaded/trained
    if classifier is None:
        messagebox.showerror(
            "Error",
            "Please train or load the CNN model first."
        )
        return

    # Select image
    filename = filedialog.askopenfilename(
        initialdir="testImages",
        title="Select Brain MRI Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All Files", "*.*")
        ]
    )

    # User pressed Cancel
    if filename == "":
        return

    # Read grayscale image
    img = cv2.imread(filename, 0)

    # Check image
    if img is None:
        messagebox.showerror(
            "Error",
            "Unable to read the selected image."
        )
        return

    # Resize and normalize
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0

    XX = img.reshape(1, 128, 128, 1)

    # Predict
    predicts = classifier.predict(XX, verbose=0)

    print("\nPrediction Probabilities:", predicts)

    cls = np.argmax(predicts)

    confidence = predicts[0][cls] * 100

    print("Predicted Class :", cls)
    print("Confidence      : {:.2f}%".format(confidence))

    # Read original image for display
    display_img = cv2.imread(filename)

    if display_img is None:
        messagebox.showerror(
            "Error",
            "Unable to load original image."
        )
        return

    display_img = cv2.resize(display_img, (800, 500))

    cv2.putText(
        display_img,
        "Prediction : {}".format(disease[cls]),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display_img,
        "Confidence : {:.2f}%".format(confidence),
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # No Tumor
    if cls == 0:

        cv2.imshow(
            "Classification Result",
            display_img
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Tumor
    elif cls == 1:

        segmented_image, edge_image = tumorSegmentation(filename)

        cv2.imshow(
            "Classification Result",
            display_img
        )

        cv2.imshow(
            "Tumor Segmented Image",
            segmented_image
        )

        cv2.imshow(
            "Edge Detected Image",
            edge_image
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss,color='red')
    plt.plot(accuracy,color='green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('Brain Tumor CNN Model Training Accuracy & Loss Graph')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Identifying Brain Tumor using X-Ray Images')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Tumor X-Ray Images Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Dataset Preprocessing & Features Extraction", command=datasetPreprocessing)
preprocessButton.place(x=430,y=550)
preprocessButton.config(font=font1) 

cnnButton = Button(main, text="Trained CNN Brain Tumor Detection Model", command=trainTumorDetectionModel)
cnnButton.place(x=810,y=550)
cnnButton.config(font=font1) 

classifyButton = Button(main, text="Brain Tumor Segmentation & Classification", command=tumorClassification)
classifyButton.place(x=50,y=600)
classifyButton.config(font=font1)

graphButton = Button(main, text="Training Accuracy Graph", command=graph)
graphButton.place(x=430,y=600)
graphButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()
