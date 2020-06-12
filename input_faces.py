from tkinter import *
from tkinter.ttk import *
# loading Python Imaging Library
from PIL import ImageTk, Image

# To get the dialog box to open when required
from tkinter import filedialog
from imutils import paths
import face_recognition
import argparse
import pickle
import os
import cv2
import time


def openfilename():
    filenames = filedialog.askopenfilename(title='"pen', multiple=True)
    return filenames


global filenames


def open_img():
    global filenames
    filenames = openfilename()

    for (index, filename) in enumerate(filenames):
        print(filename)
        img = Image.open(filename)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(root, image=img)
        panel.image = img
        panel.grid(row=int((index+4)/2), column=int(int(index+4) % 2 + 1))


def submit():
    global filenames
    username = username_entry.get()
    if not os.path.exists("dataset/{}".format(username)):
        os.makedirs("dataset/{}".format(username))
    infile = open("encodings.pickle", "rb")
    loadedData = pickle.load(infile)
    infile.close()
    knownEncodings = loadedData["encodings"]
    knownNames = loadedData["names"]
    for (index, filename) in enumerate(filenames):
        basename = os.path.basename(filename)
        image = cv2.imread(filename)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb,
                                                model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(username)
        cv2.imwrite("dataset/{}/{}".format(username, basename), image)
    progress['value'] = int((index + 1)/len(filenames) * 100)
    time.sleep(1)
    root.update_idletasks()
    data = {"encodings": knownEncodings, "names": knownNames}

    f = open("encodings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()


global username_entry

root = Tk()
name = ""
# Set Title as Image Loader
root.title("Image Loader")

# Set the resolution of window
root.geometry('%dx%d+%d+%d' % (700, 3000, 300, 150))

# Allow Window to be resizable
root.resizable(width=True, height=True)
Label(master=root, text="Username").grid(row=0)
username_entry = Entry(root)
username_entry.grid(row=0, column=1)
progress = Progressbar(root, orient=HORIZONTAL,
                       length=100, mode='determinate')
btn = Button(root, text='open image', command=open_img).grid(
    row=1, columnspan=1)


submit = Button(root, text='Submit', command=submit).grid(
    row=1, columnspan=2)

root.mainloop()
