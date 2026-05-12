# =====================================================
# Importando as bibliotecas necessárias
# =====================================================
import yfinance as yf       # Para baixar dados de ações
import pandas as pd         # Para trabalhar com tabelas
import numpy as np          # Para cálculos matemáticos


# =====================================================
# A FUNÇÃO PRINCIPAL
# =====================================================
def get_asset_data(ticker):
    """
    Baixa dados históricos e calcula indicadores de risco.
    
    ticker = código do ativo, exemplos:
        "PETR4.SA"  → Petrobras (ação brasileira)
        "BTC-USD"   → Bitcoin
        "IVVB11.SA" → ETF do S&P500 na B3
    """

    # -------------------------------------------------
    # PASSO 1 — Baixar os dados históricos
    # -------------------------------------------------
    print(f"Baixando dados de {ticker}...")

    ativo = yf.Ticker(ticker)
    dados = ativo.history(period="2y")  # 2 anos de histórico

    # Pega só a coluna de preço de fechamento
    preco = dados["Close"]


    # -------------------------------------------------
    # PASSO 2 — Calcular a Volatilidade
    # -------------------------------------------------
    # Retorno diário = variação percentual de um dia para o outro
    # Exemplo: se ontem fechou em R$10 e hoje em R$11 → retorno = 10%
    retornos_diarios = preco.pct_change().dropna()
    # dropna() = remove os valores vazios (o primeiro dia não tem retorno)

    # Volatilidade = desvio padrão dos retornos × raiz de 252
    # 252 = número de dias úteis no ano
    # Multiplicar pela raiz de 252 converte para volatilidade anual
    volatilidade = retornos_diarios.std() * np.sqrt(252)


    # -------------------------------------------------
    # PASSO 3 — Calcular o Drawdown Máximo
    # -------------------------------------------------
    # Drawdown = quanto o ativo caiu em relação ao seu pico
    # Exemplo: se subiu até R$100 e depois caiu para R$70
    #          drawdown = -30%

    # pico_historico = maior preço visto ATÉ cada data
    # cummax() = vai guardando o valor máximo acumulado
    pico_historico = preco.cummax()

    # Drawdown em cada dia = (preço atual - pico) / pico
    drawdown = (preco - pico_historico) / pico_historico

    # Pega o pior drawdown (o mais negativo)
    drawdown_maximo = drawdown.min()


    # -------------------------------------------------
    # PASSO 4 — Calcular o Sharpe Ratio
    # -------------------------------------------------
    # Sharpe mede: quanto de retorno você teve por unidade de risco
    # Quanto MAIOR o Sharpe, MELHOR (mais retorno com menos risco)
    #
    # Fórmula: (Retorno médio anual - Taxa livre de risco) / Volatilidade
    #
    # Taxa livre de risco = rendimento de um investimento "seguro"
    # Usamos 0.1065 = 10.65% (referência ao CDI/Selic brasileiro)

    taxa_livre_risco = 0.1065

    # Retorno médio diário → convertido para anual
    retorno_medio_anual = retornos_diarios.mean() * 252

    # Cálculo final do Sharpe
    sharpe = (retorno_medio_anual - taxa_livre_risco) / volatilidade


    # -------------------------------------------------
    # PASSO 5 — Montar e retornar o DataFrame
    # -------------------------------------------------
    # Organiza todos os resultados em uma tabela
    resultado = pd.DataFrame({
        "Ticker":            [ticker],
        "Volatilidade (ao ano)": [f"{volatilidade:.2%}"],   # Ex: 35.20%
        "Drawdown Máximo":   [f"{drawdown_maximo:.2%}"],    # Ex: -42.10%
        "Sharpe Ratio":      [f"{sharpe:.2f}"],             # Ex: 0.85
        "Retorno Médio Anual": [f"{retorno_medio_anual:.2%}"]
    })

    return resultado


# =====================================================
# TESTANDO A FUNÇÃO
# =====================================================
if __name__ == "__main__":

    # Testando com diferentes tipos de ativos
    ativos = ["PETR4.SA", "BTC-USD", "IVVB11.SA", "IVVB11.SA"]

    for ticker in ativos:
        print("\n" + "="*50)
        resultado = get_asset_data(ticker)
        print(resultado.to_string(index=False))

    print("\n" + "="*50)
    print("Análise concluída!") 
