import face_recognition
import cv2
import numpy as np
import os
import urllib.request
from flask import (
    Flask,
    flash,
    request,
    redirect,
    url_for,
    render_template,
    make_response,
)
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "static/uploads/"
DOWNLOAD_FOLDER = "static/downloads/"

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DOWNLOAD_FOLDER"] = DOWNLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def upload_form():
    return render_template("upload.html")


@app.route("/", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        flash("No image selected for uploading")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        flash("Image identified as:")
        return render_template("upload.html", filename=filename)
    else:
        flash("Allowed image types are -> png, jpg, jpeg, gif")
        return redirect(request.url)


@app.route("/display/<filename>")
def display_image(filename):
    # print('display_image filename:' + filename)
    path = "static/database"
    database = []
    className = []
    myList = os.listdir(path)
    for x, cl in enumerate(myList):
        curImg = cv2.imread(f"{path}/{cl}")
        database.append(curImg)
        className.append(os.path.splitext(cl)[0])
    encodeList = []
    for dt in database:
        dt = cv2.cvtColor(dt, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(dt)[0]
        encodeList.append(encode)
    img = face_recognition.load_image_file("static/uploads/" + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceLoc = face_recognition.face_locations(img)[0]
    encodeImg = face_recognition.face_encodings(img)[0]
    matches = face_recognition.compare_faces(encodeList, encodeImg)
    faceDis = face_recognition.face_distance(encodeList, encodeImg)
    matchIndex = np.argmin(faceDis)
    if faceDis[matchIndex] < 0.50:
        name = className[matchIndex].upper()
    else:
        name = "Unknown"
    # print(name)
    y1, x2, y2, x1 = faceLoc
    newimg = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    newimg = cv2.putText(newimg, name, (x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite('static/images/k.jpg', newimg)
    retval, buffer = cv2.imencode('.jpg', newimg)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpg'
    return response


if __name__ == "__main__":
    app.run()
