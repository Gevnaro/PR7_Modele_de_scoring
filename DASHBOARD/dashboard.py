import json

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
#import plotly.express as px
import numpy as np
#from BankFeatures import BankFeature
import streamlit.components.v1 as components
from lime.lime_tabular import LimeTabularExplainer 
from sklearn.cluster import KMeans



plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')





data = pd.read_feather("data/train_data_smote_feather")
label_data= pd.read_feather("data/y_data_feather")

pickle_in = open('data/model_final.pkl', 'rb') # importation du modèle
model_final = pickle.load(pickle_in)

data_mini_scaled = pd.read_feather('data/df_final_scaled')

data_original = pd.read_feather('data/X_mini_feather')
data_neighboard = pd.read_feather('data/data_neighboard_feather')

def neiboards_client(data_neighboard, number_id):



    cls = data_neighboard[data_neighboard.index == int(number_id)]['class']

    class_num = data_neighboard['class'][data_neighboard['class'] == cls.values[0]].sample(5)

    affiche_voisin = data_neighboard[['TARGET','DAYS_BIRTH', 'AMT_CREDIT','AMT_INCOME_TOTAL', 'AMT_ANNUITY','CODE_GENDER','EXT_SOURCE_1']]
    affiche_voisin['DAYS_BIRTH']=np.round(affiche_voisin['DAYS_BIRTH'],0)
    affiche_voisin['CODE_GENDER'] = affiche_voisin['TARGET'].map({0:'Men',1:'Women'})
    affiche_voisin['TARGET'] = affiche_voisin['TARGET'].map({0.0:'Accepted',1.0:'Refused'}) 

    for i in range(5):
        st.write(affiche_voisin[affiche_voisin.index == class_num.index[i]])

    



@st.cache
def load_infos_gen(data):
    lst_infos = [data.shape[0],
                    round(np.abs(data["AMT_INCOME_TOTAL"]).mean(), 2),
                    round(np.abs(data["AMT_CREDIT"].mean()), 2)]

    nb_clients = lst_infos[0] #Nombre de clients
    rev_moy = lst_infos[1] #Revenus moyen
    credits_moy = lst_infos[2] # montant moyen de crédit

    
    return nb_clients, rev_moy, credits_moy

@st.cache
def chargement_ligne_data(index, data):
    return data[data.index == int(index)]

def identite_client(data, id):
    data_client = data[data.index == int(id)]
    return data_client
@st.cache
def load_age_population(data):
    data_age = round((data["DAYS_BIRTH"]), 2)
    return data_age

@st.cache
def load_income_population(data):
    df_income = pd.DataFrame(data["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return df_income

#Loading selectbox ==> choisir l'identifiant
id_client = data_original.index.values
chk_id = st.sidebar.selectbox("Identifiant du client", id_client)
#Loading general info ==> Calculs de quelques informations générales
nb_clients, rev_moy, credits_moy = load_infos_gen(data_original)

st.sidebar.markdown("<u>**Nombre totale de client :**</u>", unsafe_allow_html=True)
st.sidebar.text(nb_clients)

# Average income
st.sidebar.markdown("<u>**Revenu moyen (USD) :**</u>", unsafe_allow_html=True)
st.sidebar.text(rev_moy)

# AMT CREDIT
st.sidebar.markdown("<u>**Montant de crédit moyen (USD) :**</u>", unsafe_allow_html=True)
st.sidebar.text(credits_moy)

#######################################
# Page d'accueil
#######################################
#Identifiant et données du client


#Prédiction
st.header("**Implémentez un modèle de scoring**")
if st.checkbox("Prédiction"):
    
    response=requests.get("https://pr7-fastapi-gev.herokuapp.com/credit/"+str(chk_id))
                             
    #response = requests.get('http://127.0.0.1:8000/credit/'+str(chk_id))
    prediction=json.loads(response.content)
    
    proba=prediction['proba']

    if proba >0.4:
        etat='Refuser'
    else:
        etat='Accepter'
    score=round(proba*100,2)

    chaine=etat + ':  avec  ' + str(score) +'% de risque de défaut de paiement'
    st.write(chaine)

if st.button("Explain Results"):
    with st.spinner('Calculating...'):
        explainer = LimeTabularExplainer(data,training_labels=label_data,
                      feature_names=data.columns, 
                      categorical_names=[1,11,26,35],        
                      class_names=['accepter','refuser'], 
                      discretize_continuous=False, 
                      verbose=True)
        # Display explainer HTML object
        lime = explainer.explain_instance(data.loc[chk_id].values, model_final.predict_proba)
        components.html(lime.as_html(), height=800)

if st.button("5 nearest neighbors"):

    neiboards_client(data_neighboard,chk_id)

#quelques informations générales
st.header("**Informations du client**")


if st.button("Détails"):
    infos_client = identite_client(data_original, chk_id) #Identité
    #st.write(infos_client)
    st.write("**Genre :**",infos_client["CODE_GENDER"].values[0]) #Genre
    st.write("**Age:** {:.0f} ans".format(abs(int(infos_client["DAYS_BIRTH"]))))
    st.write("**Nombre de personnes dans le foyer :**", infos_client["CNT_FAM_MEMBERS"].values[0])
    st.write("**Nombre d'enfants:** {:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))

# Age distribution plot
    data_age = load_age_population(data_original)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(abs(data_age), edgecolor='k', color="red", bins=20)
    ax.axvline(abs(int(infos_client["DAYS_BIRTH"].values)), color="green", linestyle='--')
    ax.set(title='Age des clients', xlabel='Age (Années)', ylabel='')
    st.pyplot(fig)

    st.subheader("*Revenus (USD)*")
    st.write("**Revenus total du client :** {:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
    st.write("**Montant du crédit du client :** {:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
    st.write("**Annuités:** {:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
    st.write("**Montant des biens pour crédit de consommation:** {:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))

    # Income distribution plot
    data_income = load_income_population(data_original)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor='k', color="red", bins=10)
    ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
    ax.set(title='Revenu du client', xlabel='Revenu (USD)', ylabel='')
    st.pyplot(fig)




















