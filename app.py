from flask import Flask,request,render_template
import os
import pickle
import numpy as np
# create the instance of the class

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")
def predictor(predictlist):
    to_predict = np.array(predictlist).reshape(1,12)
    model = pickle.load(open("model_pkl","rb"))
    result = model.predict(to_predict)
    return result[0]
@app.route("/result",methods = ["POST"])
def result():
    if request.method == "POST":
        to_dicts = request.form.to_dict()
        to_list = list(to_dicts.values())
        to_lists = list(map(int,to_list))
        result = predictor(to_lists)
        if result == 1:
            prediction = "more then 50k"
        else:
            prediction = "less then 50k"
        return render_template("result.html",prediction = prediction)
if __name__ == "__main__":
    app.run(debug = True)










