import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flask import Flask,request
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import json
from flask_cors import CORS


#Inicializar Flask
app = Flask(__name__)
CORS(app, support_credentials=True)


#Cargar modelo preentrenado
longitud, altura = 150, 150
modelo = './modelo_cnn/modelo.h5'
pesos_modelo = './modelo_cnn/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)


@app.route('/')
def index():
    return "Api clasificación de patologías de aguacate hass"

@app.route('/upload',methods = ['POST'])
def upload():
    if request.method == 'POST':  
        #Verificar si se cargó una imágen
        f = request.files['file']  
        if f:
            
            f.save(f.filename)  

            ruta = f.filename


           

            x = load_img(ruta, target_size=(longitud, altura))  #Cargar imágen y cambiar tamaño
            x = img_to_array(x)              #Transformar imágen a números
            x = np.expand_dims(x, axis=0)
            array = cnn.predict(x)              # Se realiza la predicción con el modelo
            result = array[0]
            answer = np.argmax(result)
            

            if answer == 0:
                print("pred: Rojizo")
            elif answer == 1:
                print("pred: Marceño")
            elif answer == 2:
                print("pred: Roña")
            elif answer == 3:
                print("pred: Trips")


            json_pred=json.dumps(result.tolist())
            print(json_pred)
            return json_pred

        else:
            return("No subiste nada")
    



if __name__ == "__main__":
    app.run(debug=True)
