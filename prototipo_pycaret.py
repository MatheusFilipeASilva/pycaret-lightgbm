import pandas as pd
import streamlit as st
import pickle
from pathlib import Path
from pycaret.classification import load_model, predict_model
#from pycaret.classification import *

# Diretório do script (usado para localizar arquivos relativos ao projeto)
PROJECT_DIR = Path(__file__).resolve().parent


def test_run():
    st.write("Insira os seus dados na upbox abaixo. Se não tiver dados para inserir, marque sim na box abaixo para carregar um dataset de exemplo.")
    use_example = st.checkbox("Carregar data-set de exemplo")
    df = None
    example_path = PROJECT_DIR / "credit_scoring.ftr"
    if use_example:
        if example_path.exists():
            try:
                df = pd.read_feather(example_path)
                st.info(f"Lendo dataset de exemplo em: {example_path}")
            except Exception as e:
                st.error(f"Erro ao ler arquivo de exemplo: {e}")
        else:
            st.error(f"Arquivo de exemplo 'credit_scoring.ftr' não encontrado em: {example_path}")
    else:
        uploaded_file = st.file_uploader("Insira seu arquivo .ftr")
        if uploaded_file is not None:
            try:
                df = pd.read_feather(uploaded_file)
            except Exception as e:
                st.error(f"Erro ao ler o arquivo enviado: {e}")
    return use_example, df


def coletar_modelo():
    modelo_pickle = PROJECT_DIR/"Tuned_LightGBM"
    return load_model(modelo_pickle)


def get_predict(modelo, df):
    return predict_model(modelo, data=df)


#@st.cache_data
def main():
    st.title("Prototipo de Classificação com PyCaret")
    show_diag = st.checkbox("Mostrar informações de diagnóstico")
    if show_diag:
        st.write("CWD:", Path.cwd())
        st.write("Script dir:", PROJECT_DIR)
        st.write("Arquivos na pasta do projeto:", [p.name for p in PROJECT_DIR.iterdir()])
        st.write("Arquivos na CWD:", [p.name for p in Path.cwd().iterdir()])

    a, df = test_run()
    if df is not None:
        st.write(df.head())
    else:
        st.write("Aguardando dados para carregar.")
    

    modelo = coletar_modelo()
    
    # permite upload de modelo se não for encontrado
    if modelo is None:
        uploaded_model = st.file_uploader("Carregar modelo (.pkl)", type="pkl")
        if uploaded_model is not None:
            try:
                modelo = coletar_modelo()
                st.success("Modelo carregado com sucesso a partir do upload.")
            except Exception as e:
                st.error(f"Erro ao carregar modelo do upload: {e}")
    if modelo is not None:
        st.success("Modelo pronto para uso.")
    else:
        st.info("Modelo não carregado. Coloque 'Tuned_LightGBM.pkl' no diretório do app ou faça upload para habilitar previsões.")
    
    st.write("### Previsões")
    st.write("Deseja rodar as previsões do modelo?")
    prever_agora = st.checkbox("Prever agora")

    if prever_agora:
        previsoes = get_predict(modelo, df)
        st.write("O problema talvez esteja aqui, no predict")
        st.write(previsoes)


if __name__ == "__main__":
    main()