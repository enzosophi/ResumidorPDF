import pymupdf
import streamlit as st
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv
import time
import random
import os
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")
client = genai.Client()

MODELOS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]
TENTATIVA_POR_MODELO = 2
LIMITTE_CARACTERES = 500_000

SISTEMA = "Você é um analista que resume documentos em tópicos, de forma clara e objetiva, com base no conteúdo do documento fornecido. Você deve fornecer um resumo conciso e informativo, destacando os pontos principais e as informações mais relevantes do documento. Evite incluir informações irrelevantes ou detalhes desnecessários. O resumo deve ser estruturado de maneira lógica e fácil de entender, facilitando a compreensão do conteúdo do documento."
INSTRUCOES = """Gere um relatorio deste documento com:
1. Resumo do conteúdo
2. Principais pontos abordados
3. Informações relevantes e conclusões
4 valores, dados e estatísticas importantes
5. Sugestões de melhorias ou próximos passos, se aplicável
Se alguma informação não estiver presente no documento, indique que não foi possível encontrar a informação.
"""


@st.cache_data(show_spinner=False)
def leitor_pdf(pdf_bytes: bytes) -> str | None:
    try:
        with pymupdf.open(stream=pdf_bytes.read(), filetype="pdf") as doc:
            if doc.needs_pass:
                return None
    except Exception as e:
        st.error(f"Erro ao ler o PDF: {e}")
        return None


TENTATIVA_MAXIMA = 3


def relatorio_ia(conteudo):
    if isinstance(conteudo, str):
        entrada = f"<documento>{conteudo}\n</documento>"
    else:
        entrada = types.Part.from_bytes(data=conteudo, mime_type="application/pdf")

    ultimo_erro = None
    for modelo in MODELOS:
        for tentativa in range(1, TENTATIVA_POR_MODELO + 1):
            try:
                resposta = client.models.generate_content(
                    model=modelo,
                    contents=[entrada, INSTRUCOES],
                    config=types.GenerateContentConfig(
                        system_instruction=SISTEMA, temperature=0.2
                    ),
                )
                st.caption(f"Modelo usado: {modelo} (tentativa {tentativa})")
                return resposta.text
            except errors.ServerError as e:
                ultimo_erro = e
                espera = 2**tentativa + random.uniform(0, 1)
                st.warning(
                    f"Servidor Google instavel {e.code}. Tentando novamente em {espera} segundos..."
                    f"({tentativa}/{TENTATIVA_MAXIMA})"
                )
                time.sleep(espera)
        st.info(
            f"Modelo {modelo} não conseguiu gerar o relatório. Tentando próximo modelo..."
        )

    raise ultimo_erro


# Interface do Streamlit
st.title("Resumo de Documentos PDF com IA")

arquivo = st.file_uploader("Faça upload do seu arquivo PDF", type=["pdf"])

if arquivo and st.button("Gerar relatório"):
    texto = leitor_pdf(arquivo)

    if texto is None:
        st.stop()  # o erro já foi mostrado pelo leitor_pdf

    if texto.strip():
        conteudo = texto  # modo econômico: só o texto
        st.caption(f"Texto extraído: {len(texto):,} caracteres")
    else:
        conteudo = arquivo.getvalue()  # PDF escaneado: manda o arquivo inteiro
        st.info(
            "Não encontrei texto no PDF (pode ser escaneado). Enviando o arquivo completo para a IA."
        )

    with st.spinner("Gerando relatório..."):
        try:
            relatorio = relatorio_ia(conteudo)
        except errors.ClientError as e:
            if e.code == 429:
                st.error(
                    "Limite gratuito da API atingido. Espere um pouco e tente de novo."
                )
            else:
                st.error(f"Erro na Requisição: {e}")
            st.stop()
        except errors.ServerError as e:
            st.error("O servidor do Google está instável. Tente novamente mais tarde.")
            st.stop()
        except Exception as e:
            st.error(f"Erro inesperado: {e}")
            st.stop()

    st.markdown(relatorio)
    st.download_button("⬇️ Baixar relatório (.md)", relatorio, file_name="relatorio.md")
