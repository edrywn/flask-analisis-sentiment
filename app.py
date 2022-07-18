from ast import Break
from cProfile import label
from random import sample
from unittest import result
from flask import Flask, render_template, request, url_for, redirect, jsonify
from flask_mysqldb import MySQL
import pandas as pd
from var_dump import var_dump

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'datatweet'
mysql = MySQL(app)


@app.route('/')
def home():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM datasentimen")
    rv = cur.fetchall()

    cur.close()
    var_dump(rv)
    return render_template('index.html', value=rv)


@app.route('/preprocessing')
def stemming():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM datasentimen")
    rv = cur.fetchall()
    cur.close()
    return render_template('preprocessing.html', value=rv)


@app.route('/sentiment')
def sentiment():
    cur = mysql.connection.cursor()
    pos = mysql.connection.cursor()
    net = mysql.connection.cursor()
    neg = mysql.connection.cursor()
    aku = mysql.connection.cursor()

    cur.execute("SELECT * FROM datasentimen")
    pos.execute("SELECT * FROM datasentimen WHERE sentiment='Positive'")
    net.execute("SELECT * FROM datasentimen WHERE sentiment='Netral'")
    neg.execute("SELECT * FROM datasentimen WHERE sentiment='Negative'")
    aku.execute("SELECT english, sentiment FROM datasentimen")

    result = cur.fetchall()
    resultPositif = pos.fetchall()
    resultNetral = net.fetchall()
    resultNegatif = neg.fetchall()
    resultAkurasi = aku.fetchall()

    df = pd.DataFrame(resultAkurasi, columns=["text", "sentiment"])

    df = df.astype({'text': 'string'})
    df = df.astype({'sentiment': 'category'})

    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(df['text'].astype('U'))

    X_train, X_test, y_train, y_test = train_test_split(
        text_tf, df['sentiment'], test_size=0.4, random_state=100)

    clf = MultinomialNB().fit(X_train, y_train)
    predicted = clf.predict(X_test)

    akurasi = round(accuracy_score(y_test, predicted), 4)
    presisi = round(precision_score(y_test, predicted,
                                    average="macro", pos_label="Positive"), 4)
    recall = round(recall_score(y_test, predicted,
                                average="macro", pos_label="Positive"), 4)

    cur.close()
    pos.close()
    net.close()
    neg.close()
    aku.close()
    return render_template('sentiment.html', value=result, panjangPositif=len(resultPositif), panjangNetral=len(resultNetral), panjangNegatif=len(resultNegatif), hasilAkurasi=akurasi, hasilPresisi=presisi, hasilRecall=recall)


@app.route('/visualisasi')
def visualisasi():

    pos = mysql.connection.cursor()
    net = mysql.connection.cursor()
    neg = mysql.connection.cursor()
    aku = mysql.connection.cursor()

    pos.execute("SELECT * FROM datasentimen WHERE sentiment='Positive'")
    net.execute("SELECT * FROM datasentimen WHERE sentiment='Netral'")
    neg.execute("SELECT * FROM datasentimen WHERE sentiment='Negative'")
    aku.execute("SELECT english, sentiment FROM datasentimen")

    resultPositif = pos.fetchall()
    resultNetral = net.fetchall()
    resultNegatif = neg.fetchall()
    resultAkurasi = aku.fetchall()

    df = pd.DataFrame(resultAkurasi, columns=["text", "sentiment"])

    df = df.astype({'text': 'string'})
    df = df.astype({'sentiment': 'category'})

    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(df['text'].astype('U'))

    X_train, X_test, y_train, y_test = train_test_split(
        text_tf, df['sentiment'], test_size=0.4, random_state=100)

    clf = MultinomialNB().fit(X_train, y_train)
    predicted = clf.predict(X_test)

    akurasi = round(accuracy_score(y_test, predicted), 4)

    lenPos = len(resultPositif)
    lenNet = len(resultNetral)
    lenNeg = len(resultNegatif)
    if lenPos > lenNeg:
        hasil = "Di rekomendasikan untuk kuliah tatapmuka"
    elif lenPos < lenNeg:
        hasil = "Tidak di rekomendasikan untuk kuliah tatap muka"
    else:
        hasil = "Belum bisa di ambil keputusan"

    values = [lenPos, lenNet, lenNeg]

    pos.close()
    net.close()
    neg.close()

    return render_template('visualisasi.html', values=values, panjangPositif=len(resultPositif), panjangNetral=len(resultNetral), panjangNegatif=len(resultNegatif), hasilAkurasi=akurasi, hasil=hasil)


if __name__ == '__main__':
    app.run(debug=True)
