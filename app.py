import random
import os
import joblib
import numpy as np
import spacy
import streamlit as st

# Configuración de la página (¡Debe ir al principio!)
st.set_page_config(
    page_title="RapGPT | Verificador de Real Rap",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados para un acabado más limpio
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput input {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# Cargar spaCy y el modelo de forma robusta
@st.cache_resource
def cargar_nlp_y_modelo():
  try:
    nlp = spacy.load("es_core_news_md")
  except OSError:
    try:
      os.system("python -m spacy download es_core_news_sm")
      nlp = spacy.load("es_core_news_sm")
    except OSError:
      nlp = spacy.load("es_core_news_md")

  modelo = joblib.load("modelo_rap.pkl")
  return nlp, modelo


with st.spinner("Cargando el flow y los modelos de IA... 🎧"):
  nlp, modelo = cargar_nlp_y_modelo()


# Misma función que usaste para entrenar el modelo
def obtener_vector(texto):
  doc = nlp(texto)
  if doc.has_vector:
    return doc.vector
  else:
    return np.zeros(nlp.vocab.vectors_length)


# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
  st.header("Sobre RapGPT")
  st.write(
      "Herramienta impulsada por Machine Learning (Regresión Logística +"
      " spaCy) para medir científicamente si algo respira cultura de la calle"
      " o no."
  )
  st.markdown("---")
  st.markdown("### 🛠️ Tecnologías")
  st.markdown("- Python & Streamlit\n- Scikit-Learn\n- spaCy (NLP)")
  st.markdown("---")
  st.caption("Hecho con pasión por el rap 🎤🔥")


# --- CABECERA PRINCIPAL ---
st.title("🎤 RapGPT")
st.markdown("#### *¿Es esto Real Rap o una absoluta milonga?*")
st.write("")


# Diccionarios de frases
frases_rap = {
    "alto": [
        "🔥 Más rap que gastarse el primer sueldo en una cadena.",
        (
            "🔥 Más rap que escucharse una instrumental de Midas Alonso a las"
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


# --- CUERPO PRINCIPAL ---
with st.container():
  st.markdown("### Introduce un concepto para evaluar:")
  frase_prueba = st.text_input(
      "Escribe un objeto, acción o persona:",
      value="Ergo Pro",
      label_visibility="collapsed",
  )

  st.write("")
  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
    boton_analizar = st.button(
        "🚀 Analizar Flow", type="primary", use_container_width=True
    )

if boton_analizar:
  if frase_prueba.strip() == "":
    st.warning("⚠️ No has puesto nada, escribe algo puto vago.")
  else:
    with st.spinner("Analizando códigos de barras y métricas de calle..."):
      vector_prueba = obtener_vector(frase_prueba).reshape(1, -1)
      prediccion = modelo.predict(vector_prueba)
      probabilidades = modelo.predict_proba(vector_prueba)

    st.markdown("---")
    st.subheader("📊 Veredicto Final")

    if prediccion[0] == 1:
      prob = probabilidades[0][1]
      if prob > 0.9:
        nivel, frase_elegida = (
            "Nivel: Puro Underground (Alto)",
            random.choice(frases_rap["alto"]),
        )
      elif prob > 0.8:
        nivel, frase_elegida = (
            "Nivel: Bastante Calle (Medio-Alto)",
            random.choice(frases_rap["medio_alto"]),
        )
      elif prob > 0.7:
        nivel, frase_elegida = (
            "Nivel: Aprobado por los pelos (Medio)",
            random.choice(frases_rap["medio"]),
        )
      elif prob > 0.6:
        nivel, frase_elegida = (
            "Nivel: Dudoso (Bajo-Medio)",
            random.choice(frases_rap["bajo_medio"]),
        )
      else:
        nivel, frase_elegida = (
            "Nivel: En el Limbo",
            random.choice(frases_rap["limbo"]),
        )

      st.success(
          f"### ¡SÍ ES RAP! 🎤\n**{nivel}** (Confianza:"
          f" `{prob:.0%}`)\n\n> *{frase_elegida}*"
      )
    else:
      prob = probabilidades[0][0]
      if prob > 0.9:
        nivel, frase_elegida = (
            "Nivel: Anti-Rap Absoluto (Alto)",
            random.choice(frases_no_rap["alto"]),
        )
      elif prob > 0.8:
        nivel, frase_elegida = (
            "Nivel: Cero Calle (Medio-Alto)",
            random.choice(frases_no_rap["medio_alto"]),
        )
      elif prob > 0.7:
        nivel, frase_elegida = (
            "Nivel: Flojito (Medio)",
            random.choice(frases_no_rap["medio"]),
        )
      elif prob > 0.6:
        nivel, frase_elegida = (
            "Nivel: Bastante Lejos (Bajo-Medio)",
            random.choice(frases_no_rap["bajo_medio"]),
        )
      else:
        nivel, frase_elegida = (
            "Nivel: En la cuerda floja",
            random.choice(frases_no_rap["limbo"]),
        )

      st.error(
          f"### NO ES RAP ❌\n**{nivel}** (Confianza:"
          f" `{prob:.0%}`)\n\n> *{frase_elegida}*"
      )

    st.metric(
        label="Grado de certeza del modelo", value=f"{max(probabilidades[0]):.1%}"
    )