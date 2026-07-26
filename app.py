import random
import joblib
import numpy as np
import spacy
import streamlit as st


# Cargar spaCy y el modelo directamente
@st.cache_resource
def cargar_nlp_y_modelo():
  nlp = spacy.load("es_core_news_sm")
  modelo = joblib.load("modelo_rap.pkl")
  return nlp, modelo


nlp, modelo = cargar_nlp_y_modelo()


# Misma función que usaste para entrenar el modelo
def obtener_vector(texto):
  doc = nlp(texto)
  if doc.has_vector:
    return doc.vector
  else:
    return np.zeros(nlp.vocab.vectors_length)


# Caja de texto para que el usuario escriba
frase_prueba = st.text_input(
    "Escribe un objeto, acción o persona:", "Ergo Pro"
)

# Diccionarios de frases
frases_rap = {
    "alto": [
        "🔥 Más rap que gastarse el primer sueldo en una cadena.",
        (
            "🔥 Más rap como escucharse una instrumental de Midas Alonso a las"
            " tres de la mañana."
        ),
        "🔥 Das real rap",
    ],
    "medio_alto": [
        "🔥 Wat a time to be alive. Da shi es rap.",
        "🔥 Lo haría el yonki de mi barrio, asi que si es rap.",
    ],
    "medio": [
        "🔥 ¡Es rap! Pero Cecilio G te diría que no lo suficiente.",
        "🔥 Lleva rap en la sangre, aunque le falta calle.",
    ],
    "bajo_medio": [
        (
            "🔥 Es más que rap comer en el kebab de Omar Montes, pero tampoco"
            " por mucho."
        ),
        (
            "🔥 Es rap pero muy tomado con pinzas, como un 5.0 en una"
            " recuperación."
        ),
    ],
    "limbo": [
        (
            "Si me apuntas con una pistola diría que es rap, pero tengo mis"
            " dudas. Depende la persona."
        ),
        "Podría ser rap si tienes mucha fe y poca cultura.",
    ],
}

frases_no_rap = {
    "alto": [
        "❌ Menos rap que ir a clase de tutoría en la ESO.",
        "❌ Cero rap. Es más típico de familia de la Moraleja.",
        "❌ No rap. Ofende que hayas pensado que podía serlo",
    ],
    "medio_alto": [
        (
            "❌ Igual de rap que llamarse Bizarrap y hacer una session con"
            " Maluma. NADA."
        ),
        "❌ No es rap ni para un cayetano.",
    ],
    "medio": [
        "❌ Por mucho que quieras, no es para nada rap.",
        "❌ Típico de profesor de instituto que va de guay. Muy poco rap",
    ],
    "bajo_medio": [
        "❌ Poco rap. Como mucho se podría considerar una copia barata.",
        "❌ Lo intenta pero no es rap.",
    ],
    "limbo": [
        "❌ No es rap rap, pero tampoco el antirap. Está en el limbo.",
        "❌ Diría que no es rap. Pero algún loco igual te dice que si.",
    ],
}

if st.button("Analizar"):
  if frase_prueba.strip() == "":
    st.warning("No has puesto nada, escribe algo puto vago.")
  else:
    vector_prueba = obtener_vector(frase_prueba).reshape(1, -1)
    prediccion = modelo.predict(vector_prueba)
    probabilidades = modelo.predict_proba(vector_prueba)

    if prediccion[0] == 1:
      prob = probabilidades[0][1]
      if prob > 0.9:
        st.write(random.choice(frases_rap["alto"]))
      elif prob > 0.8:
        st.write(random.choice(frases_rap["medio_alto"]))
      elif prob > 0.7:
        st.write(random.choice(frases_rap["medio"]))
      elif prob > 0.6:
        st.write(random.choice(frases_rap["bajo_medio"]))
      else:
        st.write(random.choice(frases_rap["limbo"]))
    else:
      prob = probabilidades[0][0]
      if prob > 0.9:
        st.write(random.choice(frases_no_rap["alto"]))
      elif prob > 0.8:
        st.write(random.choice(frases_no_rap["medio_alto"]))
      elif prob > 0.7:
        st.write(random.choice(frases_no_rap["medio"]))
      elif prob > 0.6:
        st.write(random.choice(frases_no_rap["bajo_medio"]))
      else:
        st.write(random.choice(frases_no_rap["limbo"]))