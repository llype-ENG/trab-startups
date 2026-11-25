from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DADOS_PRF = ROOT.parent / "analise-acidentes-prf" / "Dados"
DADOS = ROOT / "data" / "processed"
ARQUIVO_PRF = DADOS_PRF / "datatran2026.csv"
ARQUIVO_PREVISAO = DADOS / "previsao_janeiro_2026.csv"
ARQUIVO_HISTORICO = DADOS / "acidentes_por_trecho_mes.csv"
TRECHO = "SC_BR101_KM200"


def main():
    if not ARQUIVO_PRF.exists():
        raise FileNotFoundError(f"Coloque o CSV oficial de 2026 em: {ARQUIVO_PRF}")
    df = pd.read_csv(ARQUIVO_PRF, sep=";", encoding="latin1", usecols=["id", "data_inversa", "uf", "br", "km"], dtype="string", low_memory=False)
    df["data"] = pd.to_datetime(df["data_inversa"], errors="coerce")
    df["br_num"] = pd.to_numeric(df["br"].str.extract(r"(\d{1,3})", expand=False), errors="coerce")
    df["km_num"] = pd.to_numeric(df["km"].str.replace(",", ".", regex=False), errors="coerce")
    segmento = df[
        (df["data"] >= "2026-01-01") & (df["data"] < "2026-02-01")
        & (df["uf"].str.upper() == "SC") & (df["br_num"] == 101)
        & df["km_num"].ge(200) & df["km_num"].lt(210)
    ]
    previsoes = pd.read_csv(ARQUIVO_PREVISAO)
    linha = previsoes[previsoes["trecho_id"] == TRECHO]
    if linha.empty:
        raise ValueError(f"Trecho {TRECHO} não encontrado nas previsões.")
    estimativa = float(linha.iloc[0]["acidentes_estimados"])
    historico = pd.read_csv(ARQUIVO_HISTORICO)
    hist_2025 = historico[(historico["trecho_id"] == TRECHO) & historico["ano_mes"].between("2025-01", "2025-12")]
    real = len(segmento)
    erro = abs(estimativa - real)
    print("=" * 65)
    print("VALIDAÇÃO: PREVISÃO X REALIDADE | JANEIRO DE 2026")
    print(f"Trecho: {TRECHO}")
    print(f"Previsão: {estimativa:.2f} | Acidentes reais: {real}")
    print(f"IDs únicos: {segmento['id'].nunique()} | IDs duplicados: {segmento['id'].duplicated().sum()}")
    print(f"Acidentes em 2025: {int(hist_2025['qtd_acidentes'].sum())}")
    print(f"Média mensal de 2025: {hist_2025['qtd_acidentes'].sum() / 12:.2f}")
    print(f"Erro absoluto: {erro:.2f}")
    if real:
        print(f"Erro percentual: {erro / real * 100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
