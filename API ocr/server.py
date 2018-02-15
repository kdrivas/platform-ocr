import os
import pandas as pd 
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():

	try:
		var = request.get_json()
		print(var)
	except Exception as e:
		raise e

	return var