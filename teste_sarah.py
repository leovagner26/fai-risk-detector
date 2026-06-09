from gabriel_data import get_asset_data
from sarah_model import get_risk_score, treinar_modelo

# Primeiro treinamos a rede neural
treinar_modelo()

# Depois testamos com os ativos do plano
ativos = ["PETR4.SA", "BTC-USD", "^IRX"]

for ativo in ativos:
    print("\n" + "="*50)
    print(f"Analisando {ativo}...")

    df = get_asset_data(ativo)

    print("\nDados recebidos do Gabriel:")
    print(df.to_string(index=False))

    score = get_risk_score(df)

    print(f"\nScore de risco de {ativo}: {score:.2f}/100")