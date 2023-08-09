from flask import Flask, render_template
import sys
application = Flask(__name__)

@application.route("/")
def main():
    return render_template("main.html")

@application.route("/info")
def info():
    return render_template("info.html")
    
@application.route("/loading")
def loading():
    return render_template("loading.html")

@application.route("/report")
def report():
    return render_template("report.html")

if __name__ == "__main__" : 
    application.run(host="0.0.0.0", port=9900)