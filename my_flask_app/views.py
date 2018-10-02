from flaskexample import app

from flask import request
from flask import render_template
from flask import json
from flask import jsonify
from flaskexample import app
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import json
import pickle

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      return render_template("result.html",result = result)
@app.route('/cat',methods = ['POST', 'GET'])
def cat():
   if request.method == 'POST':
      cat = request.form
      return render_template("cat.html",result = result)

if __name__ == '__main__':
   app.run(debug = True)
