
from fastapi import FastAPI
from fastapi import encoders
import pandas as pd
import pickle
import uvicorn ## ASGI
from starlette import responses

# Create the app object
app = FastAPI()

data = pd.read_feather("data/train_data_smote_feather")
#label_data= pd.read_feather("y_data_feather")

pickle_in = open('data/model_final.pkl', 'rb') # importation du modèle
model_final = pickle.load(pickle_in)

def predict_col(data,index,model):
    score_client = model.predict_proba(data[data.index == int(index)])[:,1]

    return score_client



# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
   return {'message': "Implémentez un modèle de scoring GEV"}



@app.get('/credit/{id_client}')
def credit(id_client : int):
    pred=predict_col(data,id_client,model_final)

    dict_final = {
        'proba':pred[0],
        }
    json_item=encoders.jsonable_encoder(dict_final)
    return responses.JSONResponse(content=json_item)

#lancement de l'application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

#uvicorn app:app --reload     