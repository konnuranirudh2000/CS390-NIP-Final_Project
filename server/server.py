import activation
import os
import time
from flask import *  
PEOPLE_FOLDER = os.path.join('static', 'images')
app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
import saliency

@app.route('/')  
def upload():
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        output_a = activation.activation(f.filename)
 #       time.sleep(12)
        print("FILE OUTPUT:", output_a)
        output_a = os.path.join(app.config['UPLOAD_FOLDER'],output_a)
        print("FILE OUTPUT:", output_a)
        output_s = saliency.get_saliency(f.filename)
        output_s = os.path.join(app.config['UPLOAD_FOLDER'],output_s)
        return render_template("success.html", activation_name = output_a, saliency_name=output_s)  
  
if __name__ == '__main__':  
    app.run(debug = True)
