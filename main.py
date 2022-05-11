# import required modules
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from src.detect import process_image
import os
# create flask app
app = Flask(__name__)

# main route (show html page)
@app.route('/')
def index():
    return render_template('index.html')


# api endpoint for image upload
@app.route('/api/upload', methods=['POST'])
def upload():
    # receive the file from the client
    file = request.files['file']
    filepath = f'static/temp/{file.filename}'
    file.save(filepath) # save to directory
    process_image(filepath)
    # return server url to client
    return f"{request.url_root}{filepath}"
# Run flask server
if __name__ == '__main__':
    app.run(debug=True) # set debug true to load reload server auto on changes