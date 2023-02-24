import os
from pathlib import Path
from tkinter import filedialog

from filedialogs import open_file_dialog, open_folder_dialog, save_file_dialog
from flask import Flask
from SV2TTS_master import *

app = Flask(__name__)


# Members API rout

#http://localhost:5000/Members
@app.route("/Members")
def members():
    x = ModelMaster()
    x._upload_audio(Path("../SV2TTS/samples/p240_00000.mp3"))
    x._compute_embedding()
    x.synthesize("I can now make this say whatever I want, in almost real time")
    x._save_to_file("./test.wav")
    return{"members":["Member1","Member2","Member3"]}


#http://localhost:5000/Members
@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"    





if __name__ == "__main__":
    app.run(debug=True)
    