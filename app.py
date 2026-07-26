import os
import joblib
import numpy as np
import spacy
import streamlit as st


# Función para asegurar que el modelo de spaCy está disponible en el servidor
@st.cache_resource
def cargar_recursos():
  try:
    nlp = spacy.load("es_core_news_md")
  except OSError:
    # Si Streamlit Cloud no lo tiene instalado, lo descarga automáticamente al arrancar
    os.system("python -m spacy download es_core_news_md")
    nlp = spacy.load("es_core_news_md")

  modelo = joblib.load("modelo_rap_semantico.pkl")
  return nlp, modelo


nlp, modelo = cargar_recursos()

# Configuración de la página
st.title("🎤 ¿Es RAP o no?")
st.write(
    "Introduce lo que quieres saber si es rap o no en opinión de un experto."
)


# Cargar spaCy y el modelo (usamos st.cache_resource para que cargue rápido)
@st.cache_resource
def cargar_recursos():
  nlp = spacy.load("es_core_news_md")
  modelo = joblib.load("modelo_rap.pkl")
  return nlp, modelo


nlp, modelo = cargar_recursos()

# Caja de texto para que el usuario escriba
frase_prueba = st.text_input(
    "Escribe un objeto, acción o persona:", "Ergo Pro"
)

if st.button("Analizar"):
  if frase_prueba.strip() == "":
    st.warning("No has puesto nada, escribe algo puto vago.")
  else:
    # Procesar y predecir
    doc = nlp(frase_prueba)
    vector_prueba = doc.vector.reshape(1, -1)

    prediccion = modelo.predict(vector_prueba)
    probabilidades = modelo.predict_proba(vector_prueba)

   frases_rap = {
    "alto": [
        "🔥 Más rap que gastarse el primer sueldo en una cadena.",
        "🔥 Más rap como escucharse una instrumental de Midas Alonso a las tres de la mañana.",
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
        "🔥 Es más que rap comer en el kebab de Omar Montes, pero tampoco por mucho.",
        "🔥 Es rap pero muy tomado con pinzas, como un 5.0 en una recuperación.",
    ],
    "limbo": [
        "Si me apuntas con una pistola diría que es rap, pero tengo mis dudas. Depende la persona.",
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
	"❌ Igual de rap que llamarse Bizarrap y hacer una session con Maluma. NADA.",
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

if prediccion[0] == 1:
  prob = probabilidades[0][1]
  if prob > 0.9:
    print(random.choice(frases_rap["alto"]))
  elif 0.9 >= prob > 0.8:
    print(random.choice(frases_rap["medio_alto"]))
  elif 0.8 >= prob > 0.7:
    print(random.choice(frases_rap["medio"]))
  elif 0.7 >= prob > 0.6:
    print(random.choice(frases_rap["bajo_medio"]))
  elif 0.6 >= prob > 0.5:
    print(random.choice(frases_rap["limbo"]))
else:
  prob = probabilidades[0][0]
  if prob > 0.9:
    print(random.choice(frases_no_rap["alto"]))
  elif 0.9 >= prob > 0.8:
    print(random.choice(frases_no_rap["medio_alto"]))
  elif 0.8 >= prob > 0.7:
    print(random.choice(frases_no_rap["medio"]))
  elif 0.7 >= prob > 0.6:
    print(random.choice(frases_no_rap["bajo_medio"]))
  elif 0.6 >= prob > 0.5:
    print(random.choice(frases_no_rap["limbo"]))