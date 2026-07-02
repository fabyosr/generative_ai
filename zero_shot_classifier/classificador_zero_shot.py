import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(
    page_title="Zero-Shot Classifier",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Classificador Zero-Shot")
st.write("Classifique frases utilizando modelos Zero-Shot da Hugging Face.")

# -------------------------------
# Cache do modelo
# -------------------------------

@st.cache_resource
def carregar_modelo(nome_modelo):

    return pipeline(
        "zero-shot-classification",
        model=nome_modelo
    )

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.header("Configurações")

modelo = st.sidebar.text_input(
    "Modelo",
    "facebook/mbart-large-50-mnli-multilingual"
)

multilabel = st.sidebar.checkbox(
    "Permitir múltiplas categorias",
    False
)

template = st.sidebar.text_input(
    "Template da hipótese",
    "Este texto é sobre {}."
)

# -------------------------------
# Categorias
# -------------------------------

st.subheader("Categorias")

categorias_txt = st.text_area(
    "Digite uma categoria por linha",
    """economia
tecnologia
esportes
saúde""",
    height=150
)

categorias = [
    c.strip()
    for c in categorias_txt.split("\n")
    if c.strip()
]

# -------------------------------
# Frases
# -------------------------------

st.subheader("Frases")

frases_txt = st.text_area(
    "Digite uma frase por linha",
    """O Banco Central anunciou hoje uma nova redução na taxa Selic.

A nova atualização do sistema operacional corrigiu o bug.

O atacante marcou três gols na partida.

O hospital inaugurou uma nova ala de pediatria.""",
    height=220
)

frases = [
    f.strip()
    for f in frases_txt.split("\n")
    if f.strip()
]

# -------------------------------
# Botão
# -------------------------------

if st.button("🚀 Classificar"):

    with st.spinner("Carregando modelo..."):

        classificador = carregar_modelo(modelo)

    with st.spinner("Classificando..."):

        resultados = classificador(
            frases,
            candidate_labels=categorias,
            hypothesis_template=template,
            multi_label=multilabel
        )

    linhas = []

    if isinstance(resultados, dict):
        resultados = [resultados]

    for r in resultados:

        linhas.append(
            {
                "Frase": r["sequence"],
                "Categoria": r["labels"][0],
                "Confiança (%)": round(
                    r["scores"][0] * 100,
                    2
                )
            }
        )

    df = pd.DataFrame(linhas)

    st.success("Classificação concluída!")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Confiança")

    for _, row in df.iterrows():

        st.write(f"**{row['Categoria']}**")

        st.progress(
            row["Confiança (%)"] / 100
        )

        st.caption(row["Frase"])

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Baixar CSV",
        csv,
        "resultado.csv",
        "text/csv"
    )