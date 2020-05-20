import tensorflow as tf
import numpy as np
import cv2
import os
import random
import time
from multiprocessing import Process, Queue, current_process
import matplotlib.pyplot as plt
from mss import mss
from PIL import Image
import pyautogui
import pynput
from music21 import *
import ast

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 0.6
fontColor              = (255,0,255)
lineType               = 2

allNotes = [
"A#0","C#1","D#1","F#1","G#1","A#1",
"C#2","D#2","F#2","G#2","A#2",
"C#3","D#3","F#3","G#3","A#3",
"C#4","D#4","F#4","G#4","A#4",
"C#5","D#5","F#5","G#5","A#5",
"C#6","D#6","F#6","G#6","A#6",
"C#7","D#7","F#7","G#7","A#7",
"A0","B0",
"C1","D1","E1","F1","G1","A1","B1",
"C2","D2","E2","F2","G2","A2","B2",
"C3","D3","E3","F3","G3","A3","B3",
"C4","D4","E4","F4","G4","A4","B4",
"C5","D5","E5","F5","G5","A5","B5",
"C6","D6","E6","F6","G6","A6","B6",
"C7","D7","E7","F7","G7","A7","B7","C8"]

def display_img(img):
    cv2.imshow('Img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_labels_and_images():
    start_time = time.time()
    filepath = "keysData/"
    # Read the label file, mapping image filenames to (x, y) pairs for UL corner
    with open(filepath + "keysLabel") as label_file:
        label_string = label_file.readline()
        label_map = eval(label_string)
    print("Read label map with", len(label_map), "entries")

    # Read the directory and make a list of files
    filenames = []
    for filename in os.listdir(filepath):
        if random.random() < -0.7: continue # to select just some files
        if filename.find("png") == -1: continue # process only images
        filenames.append(filename)
    print("Read", len(filenames), "images.")

    # Extract the features from the images
    print("Extracting features")
    train, train_labels, predict, predict_labels = [], [], [], []
    processes = []     # Hold (process, queue) pairs
    num_processes = 8  # Process files in parallel

    # Launch the processes
    for start in range(num_processes):
        q = Queue()
        files_for_one_process = filenames[start:len(filenames):num_processes]
        file_process = Process(target=process_files, \
                        args=(filepath, files_for_one_process, label_map, q))
        processes.append((file_process, q))
        file_process.start()

    # Wait for processes to finish, combine their results
    for p, q in processes: # For each process and its associated queue
        result = q.get()            # Blocks until the item is ready
        train += result[0]          # Get training features from child
        train_labels += result[1]   # Get training labels from child
        predict += result[2]        # Get prediction features from child
        predict_labels += result[3] # Get prediction labels from child
        p.join()                    # Wait for child process to finish
    print("Done extracting features from images. Time =", time.time() - start_time)

    return (train, train_labels, predict, predict_labels)


# Child process, intended to be run in parallel in several copies
#  files      our portion of the files to process
#  label_map  contains the right answers for each file
#  q          a pipe for sending results back to the parent process
def process_files(filepath, files, label_map, q):
    np.random.seed(current_process().pid) # Each child gets a different RNG
    t, tl, p, pl = [], [], [], []
    checkpoint, check_interval, num_done = 0, 5, 0 # Just for showing progress
    for filename in files:
        if 100*num_done > (checkpoint + check_interval) * len(files):
            checkpoint += check_interval
            print((int)(100 * num_done / len(files)), "% done")
        num_done += 1
        img = cv2.imread(filepath + filename,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (640, 24))
        img = img/255.0
        img = tf.reshape(img, [24, 640, 1])
        for contrast in [1, 1.2]:
            img2 = tf.image.adjust_contrast(img, contrast_factor=contrast)
            for bright in [0.0, 0.1, 0.2]:
                img3 = tf.image.adjust_brightness(img2, delta=bright)
                img3 = img3.numpy()
                img3 = img3.reshape(-1, 640, 24)
                img3 = np.float32(img3)
                if np.random.random() < 0.8: # 80% of images
                    t.append(img3)
                    tl.append(label_map[filename])
                else:
                    p.append(img3)
                    pl.append(label_map[filename])
                            
    q.put((t, tl, p, pl))


def build_model():
    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, input_shape=(1, 640, 24), kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(88, activation='sigmoid')])
    print("Shape of output", model.compute_output_shape(input_shape=(None, 24, 640, 1)))
    '''model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['mse'])'''
    model.compile(optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy'])
    print("Done building the network topology.")

    # Read training and prediction data
    train, train_labels, predict, predict_labels = read_labels_and_images()
    checkpoint = tf.keras.callbacks.ModelCheckpoint("models/keyboardDetect/checkpoint_{epoch}")

    # Train the network
    print("Starting to train the network, on", len(train), "samples.")
    history = model.fit(np.asarray(train, dtype=np.float32), np.asarray(train_labels), \
              epochs=20, batch_size=64, verbose=2, callbacks=[checkpoint])
    model.save("models/keyboardDetect")
    print("Done training network.")

    # Test Accuracy
    validation_x = np.asarray(predict)
    validation_y = np.asarray(predict_labels)
    test_loss, test_acc = model.evaluate(validation_x,  validation_y, verbose=2)
    print(test_acc)
    return model

# Saves coordinates at mouse press and release
# to determine area of screen to use as input
coords = []
def on_click(x, y, button, pressed):
    print('{0} at {1}'.format('Pressed' if pressed else 'Released', (x, y)))
    coords.append((x,y))
    if not pressed:
        coords.append((x,y))
        # Stop listener
        if len(coords) == 2:
            return False
        
# V1 visualizer of notes that the NN thinks are being played
def show_sheet(img, notes):
    global prevChords
    c = set()
    color1 = (0, 0, 0)
    color2 = (0, 0, 255)
    radius = 5
    thresh = 0.1
    for i in range(36):
        if notes[i] > thresh:
            img = cv2.circle(img, (int(img.shape[1]/2), (int(img.shape[0]*i/36))), radius, color1, -1)
            c.add(allNotes[i])
    for i in range(52):
        if notes[i+36] > thresh:
            img = cv2.circle(img, (int(img.shape[1]/2), (int(img.shape[0]*i/52))), radius, color2, -1)
            c.add(allNotes[i+36])
    stream1.append(chord.Chord(c-prevChords))
    prevChords = c
    return img
    
# V2 visualizer of notes that the NN thinks are being played
def show_sheet2(img, notes):
    global prevChords
    c = set()
    color1 = (0, 0, 0)
    color2 = (0, 0, 255)
    radius = 5
    i = np.argmax(notes)
    print(allNotes[i])
    c.add(allNotes[i])
    if i == 0:
        stream1.append(note.Rest())
    if i < 36 and i > 0:
        img = cv2.circle(img, (int(img.shape[1]/2), (int(img.shape[0]*i/36))), radius, color1, -1)
    if i >= 36:
        img = cv2.circle(img, (int(img.shape[1]/2), (int(img.shape[0]*(i-36)/52))), radius, color2, -1)
    stream1.append(chord.Chord(c-prevChords))
    prevChords = c
    return img
    
# Shift over notes on visualizer
def shift(img, size):
    img = img[:,size:img.shape[1]]
    arr = [(255,255,255)]*img.shape[0]
    c = np.array([arr])
    c = np.swapaxes(c,0,1)
    # Adding column to numpy array
    for i in range(size):
        img = np.column_stack((img, c))
    img = img.astype(np.uint8)
    return img
    
# Reads model saved in filename on disk, runs it using screen as input
def load_and_run_model(filename):
    model = tf.keras.models.load_model(filename)
    listener = pynput.mouse.Listener(
        on_click=on_click)
    listener.start()
    while len(coords) < 2:
        continue
    mon = {'top': coords[0][1], 'left': coords[0][0], 'width': coords[1][0] - coords[0][0], 'height': coords[1][1] - coords[0][1]}
    sct = mss()
    sheet = np.zeros((240, 320, 3), np.uint32)
    sheet[:] = (255, 255, 255)
    while(True):
        img = np.array(sct.grab(mon))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (640, 24))
        img = img/255.0
        cv2.imshow("piano", img)
        tf_img = img.reshape(-1, 640, 24)
        tf_img = np.float32(tf_img)
        p = model.predict(np.asarray([tf_img], dtype=np.float32))[0]
        sheet = shift(sheet,25)
        sheet = show_sheet(sheet,p)
        cv2.imshow("sheet", sheet)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    stream1.show()
    cv2.destroyAllWindows()
    
prevChords = set()
stream1 = stream.Stream()
#build_model()
load_and_run_model("models/keyboardDetect")


