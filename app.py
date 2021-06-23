import os

from flask import Flask, jsonify, request
import flask
from release import ReleaseModel
import numpy as np
import torch
import json
from flask_cors import CORS, cross_origin

num_items = 3500

app = Flask(__name__)
CORS(app)
modelClass = ReleaseModel(model_path='./model.ckpt', num_items=num_items)

fmti = open('model_to_id.json')
model_to_id = json.load(fmti)

fitm = open('id_to_model.json')
id_to_model = json.load(fitm)



@app.route('/predict', methods=['POST'])
@cross_origin(supports_credentials=True)
def predict():
    if request.method == 'POST':
        print("OK")
        ids = request.json['ids']
        num = int(request.json['num'])

        model_input = torch.zeros((num_items,))

        l = [id_to_model[str(i)] for i in ids]
        model_input[l] = 1
    
        ret = modelClass.predict(model_input)

        ret[l] = 0

        s = torch.argsort(ret, descending=True)
        s = s[:num]

        l2 = [model_to_id[str(i.item())] for i in s]

        obj = {
            'rets': l2
        }

        response = jsonify(obj)
        
        return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)