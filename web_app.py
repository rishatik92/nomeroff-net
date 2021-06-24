# Specify device
import os

# Specify device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Import all necessary libraries.
import numpy as np
import sys
import cv2

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('../')

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from NomeroffNet.YoloV5Detector import Detector

detector = Detector()
detector.load()

from NomeroffNet.BBoxNpPoints import NpPointsCraft, getCvZoneRGB, convertCvZonesRGBtoBGR, reshapePoints

npPointsCraft = NpPointsCraft()
npPointsCraft.load()

from NomeroffNet.OptionsDetector import OptionsDetector
from NomeroffNet.TextDetector import TextDetector

from NomeroffNet import TextDetector
from NomeroffNet import textPostprocessing

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

# load models
optionsDetector = OptionsDetector()
optionsDetector.load("latest")

textDetector = TextDetector.get_static_module("ru")
textDetector.load("latest")

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file and (file.content_type.rsplit('/', 1)[1] in ALLOWED_EXTENSIONS).__bool__():
            f  # Detect numberplate
            img = cv2.imdecode(file.read())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            targetBoxes = detector.detect_bbox(img)
            all_points = npPointsCraft.detect(img, targetBoxes, [5, 2, 0])

            # cut zones
            zones = convertCvZonesRGBtoBGR([getCvZoneRGB(img, reshapePoints(rect, 1)) for rect in all_points])

            # predict zones attributes
            regionIds, countLines = optionsDetector.predict(zones)
            regionNames = optionsDetector.getRegionLabels(regionIds)

            # find text with postprocessing by standart
            textArr = textDetector.predict(zones)
            textArr = textPostprocessing(textArr, regionNames)
            print(textArr)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
