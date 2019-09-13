import os
import requests
import time
import main
import urllib
from flask import Flask, render_template, request
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./static/uploads"
THUMBNAIL_PATH="./static/img/main.png"

@app.route("/", methods=["GET"])
def acsess_main_page():
    labels=""
    thumbnail_path=urllib.parse.urljoin(request.url_root,THUMBNAIL_PATH)
    for i in main.LABELS:
        labels+=main.LABELS[i]+"か"
    return render_template("index.html", labels=labels, thumbnail_path=thumbnail_path)

@app.route("/", methods=["POST"])
def upload_file():
    if request.files["file"]:
        file = request.files["file"]
        ip = request.remote_addr;
        ut=time.time()
        img_name="%s.dat"%(ip+"_"+str(ut)).replace(".","_")
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], str(img_name))
        file.save(img_path)
        result=main.judge(img_path)
        RANK_TOP=0;
        answer=result[RANK_TOP]["name"]
        result_texts=[];
        for r in result:
            result_texts.append("%s: %.1f％"%(r["name"],r["rate"]))
        detail=" / ".join(result_texts)
    return render_template("result.html",img_path=img_path,answer=answer,detail=detail)

if __name__ == "__main__":
    app.run()
