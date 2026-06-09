# =====================================================
# sarah_model.py
# Parte da Sarah: Rede Neural com TensorFlow/Keras
# =====================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# Nome do arquivo onde o modelo treinado será salvo
NOME_MODELO = "modelo_risco.keras"


# =====================================================
# 1. Classificar o tipo de ativo
# =====================================================

def classificar_tipo_ativo(ticker):
    """
    Classifica o ativo em uma categoria.

    Isso é importante porque renda fixa, ações e criptomoedas
    têm comportamentos de risco diferentes.
    """

    ticker = str(ticker).upper()

    if ticker == "^IRX":
        return "renda_fixa"

    elif "BTC" in ticker or "ETH" in ticker or "-USD" in ticker:
        return "cripto"

    elif ticker.endswith(".SA"):
        return "acao"

    else:
        return "outro"


# =====================================================
# 2. Criar dados de treino simulados
# =====================================================

def criar_dataset_treino():
    """
    Cria uma base simulada para treinar a rede neural.

    Como ainda não temos uma base real com scores prontos,
    criamos exemplos artificiais coerentes:

    - renda fixa: score baixo
    - ações: score médio
    - cripto: score alto
    """

    np.random.seed(42)

    linhas = []

    # -------------------------------------------------
    # Classe 1: Renda fixa / baixo risco
    # -------------------------------------------------
    for _ in range(300):
        volatilidade = np.random.uniform(0.01, 0.15)
        drawdown = np.random.uniform(0.01, 0.40)
        sharpe = np.random.uniform(-5.0, 2.0)
        retorno = np.random.uniform(-0.25, 0.20)

        score = 5
        score += volatilidade * 60
        score += drawdown * 20

        if sharpe < 0:
            score += 3

        if retorno < 0:
            score += 3

        score = max(0, min(30, score))

        linhas.append([
            volatilidade,
            drawdown,
            sharpe,
            retorno,
            1,  # eh_renda_fixa
            0,  # eh_acao
            0,  # eh_cripto
            score
        ])

    # -------------------------------------------------
    # Classe 2: Ações / risco médio
    # -------------------------------------------------
    for _ in range(300):
        volatilidade = np.random.uniform(0.15, 0.50)
        drawdown = np.random.uniform(0.10, 0.60)
        sharpe = np.random.uniform(-1.5, 2.0)
        retorno = np.random.uniform(-0.30, 0.40)

        score = 30
        score += volatilidade * 60
        score += drawdown * 50

        if sharpe < 0:
            score += 10
        elif sharpe < 0.5:
            score += 5
        elif sharpe > 1:
            score -= 5

        if retorno < 0:
            score += 5

        score = max(35, min(75, score))

        linhas.append([
            volatilidade,
            drawdown,
            sharpe,
            retorno,
            0,  # eh_renda_fixa
            1,  # eh_acao
            0,  # eh_cripto
            score
        ])

    # -------------------------------------------------
    # Classe 3: Criptomoedas / alto risco
    # -------------------------------------------------
    for _ in range(300):
        volatilidade = np.random.uniform(0.30, 1.30)
        drawdown = np.random.uniform(0.30, 0.90)
        sharpe = np.random.uniform(-2.0, 2.0)
        retorno = np.random.uniform(-0.60, 1.00)

        score = 55
        score += volatilidade * 40
        score += drawdown * 30

        if sharpe < 0:
            score += 10
        elif sharpe > 1:
            score -= 5

        if retorno < 0:
            score += 5

        score = max(70, min(100, score))

        linhas.append([
            volatilidade,
            drawdown,
            sharpe,
            retorno,
            0,  # eh_renda_fixa
            0,  # eh_acao
            1,  # eh_cripto
            score
        ])

    colunas = [
        "Volatilidade_Anual",
        "Drawdown_Maximo",
        "Sharpe_Ratio",
        "Retorno_Medio_Anual",
        "Eh_Renda_Fixa",
        "Eh_Acao",
        "Eh_Cripto",
        "Score"
    ]

    df_treino = pd.DataFrame(linhas, columns=colunas)

    return df_treino


# =====================================================
# 3. Treinar a rede neural
# =====================================================

def treinar_modelo():
    """
    Treina uma rede neural para prever o score de risco.
    """

    print("Criando base de treino simulada...")

    df_treino = criar_dataset_treino()

    X = df_treino[
        [
            "Volatilidade_Anual",
            "Drawdown_Maximo",
            "Sharpe_Ratio",
            "Retorno_Medio_Anual",
            "Eh_Renda_Fixa",
            "Eh_Acao",
            "Eh_Cripto"
        ]
    ].values

    y = df_treino["Score"].values

    print("Treinando rede neural com TensorFlow...")

    normalizador = layers.Normalization()
    normalizador.adapt(X)

    modelo = keras.Sequential([
        normalizador,
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="linear")
    ])

    modelo.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    modelo.fit(
        X,
        y,
        epochs=200,
        batch_size=32,
        verbose=0
    )

    modelo.save(NOME_MODELO)

    print(f"Modelo treinado e salvo como {NOME_MODELO}")

    return modelo


# =====================================================
# 4. Preparar os dados do Gabriel
# =====================================================

def preparar_dados_do_gabriel(df):
    """
    Recebe o DataFrame do Gabriel e transforma no formato
    que a rede neural espera.
    """

    ticker = str(df["Ticker"].iloc[0])
    tipo_ativo = classificar_tipo_ativo(ticker)

    volatilidade = abs(float(df["Volatilidade_Anual"].iloc[0]))
    drawdown = abs(float(df["Drawdown_Maximo"].iloc[0]))
    sharpe = float(df["Sharpe_Ratio"].iloc[0])
    retorno = float(df["Retorno_Medio_Anual"].iloc[0])

    eh_renda_fixa = 1 if tipo_ativo == "renda_fixa" else 0
    eh_acao = 1 if tipo_ativo == "acao" else 0
    eh_cripto = 1 if tipo_ativo == "cripto" else 0

    dados = np.array([
        [
            volatilidade,
            drawdown,
            sharpe,
            retorno,
            eh_renda_fixa,
            eh_acao,
            eh_cripto
        ]
    ])

    return dados


# =====================================================
# 5. Função principal da Sarah
# =====================================================

def get_risk_score(df):
    """
    Recebe o DataFrame do Gabriel e retorna um score de risco entre 0 e 100.
    """

    if os.path.exists(NOME_MODELO):
        modelo = keras.models.load_model(NOME_MODELO)
    else:
        print("Modelo ainda não existe. Treinando agora...")
        modelo = treinar_modelo()

    dados = preparar_dados_do_gabriel(df)

    score = modelo.predict(dados, verbose=0)[0][0]

    # Garante que o score fique entre 0 e 100
    if score < 0:
        score = 0

    if score > 100:
        score = 100

    return float(score)


# =====================================================
# 6. Teste simples
# =====================================================

if __name__ == "__main__":
    treinar_modelo()
    print("sarah_model.py funcionando corretamente.")