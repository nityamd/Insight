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
from ytapi_extractor import videoId_data, video_query
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import logging

afile = open(r'tfidf_model_description.pkl', 'rb')
des_model = pickle.load(afile, encoding = 'latin1')
afile.close()

afile = open(r'tfidf_vectorizer_description.pkl', 'rb')
des_vectorizer = pickle.load(afile, encoding = 'latin1')
afile.close()

afile = open(r'tfidf_model_title.pkl', 'rb')
ti_model = pickle.load(afile, encoding = 'latin1')
afile.close()

afile = open(r'tfidf_vectorizer_title.pkl', 'rb')
ti_vectorizer = pickle.load(afile, encoding = 'latin1')
afile.close()



@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      earl = result['URL']
      df = videoId_data(video_query(earl))
      des_vector = des_vectorizer.transform(df['description'])
      des_rating = des_model.predict(des_vector)[0]
      ti_vector = ti_vectorizer.transform(df['title'])
      ti_rating = ti_model.predict(ti_vector)[0]

      rating = des_rating + ti_rating
      if rating == 2.0:
          output = 'Shady!'
      elif rating == 1.0:
          output = 'Shady? Clean?'
      else:
          output = 'Clean!'
      logging.warning("I'm so successful!")
      logging.warning(rating)
      return render_template("result.html",
                             result = output,
                             title = df['title'][0],
                             description = df['description'][0])

@app.route('/insight',methods = ['POST', 'GET'])
def insight():
   if request.method == 'POST':
      insight = request.form
      return render_template("insight.html",result = result)

if __name__ == '__main__':
   app.run(debug = True)
