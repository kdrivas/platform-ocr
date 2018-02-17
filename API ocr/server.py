import os
import cv2
import numpy as np
from flask import Flask, jsonify, request
from werkzeug.datastructures import ImmutableMultiDict
from Recognizer_model import Recognizer

app = Flask(__name__)

def read_image(file):
	img = file.stream.read()
	nparr = np.fromstring(img, np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

	return img

@app.route('/get_sentence', methods=['POST'])
def apicall():

	try:
		file = request.files['imgProcessing']
		img = read_image(file)

		recog = Recognizer(50, load_model='model.h5')
		pred_word = recog.eval_image(img)

		print(pred_word)
	except Exception as e:
		raise e

	responses = jsonify(word=pred_word)
	responses.status_code = 200

	print(responses)
	return responses

@app.route('/get_words_bbox', methods=['POST'])
def apicall():

	try:
		file = request.files['imgProcessing']
		img = read_image(file)

		bbox=[]

		recog = Recognizer(50, load_model='model.h5')
		pred_word = recog.eval_image(img)

		print(pred_word)
	except Exception as e:
		raise e

	responses = jsonify(word=pred_word, bbox=pred_bbox)
	responses.status_code = 200

	print(responses)
	return responses