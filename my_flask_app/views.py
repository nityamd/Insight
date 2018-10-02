from flask import request
from flask import render_template
from flask import json
from flask import jsonify
from my_flask_app import app
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
@app.route('/insight',methods = ['POST', 'GET'])
def insight():
   if request.method == 'POST':
      insight = request.form
      return render_template("insight.html",result = result)

if __name__ == '__main__':
   app.run(debug = True)
