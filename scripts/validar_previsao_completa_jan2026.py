from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DADOS_PRF = ROOT.parent / "analise-acidentes-prf" / "Dados"
DADOS = ROOT / "data" / "processed"
ARQUIVO_REAL = DADOS_PRF / "datatran2026.csv"
ARQUIVO_PREVISAO = DADOS / "previsao_janeiro_2026.csv"
ARQUIVO_HISTORICO = DADOS / "acidentes_por_trecho_mes.csv"


def metricas(y_real, y_pred) -> dict[str, float]:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-9)
    k = max(1, int(np.ceil(len(y_real) * 0.10)))
    indices = np.argsort(y_pred, kind="stable")[-k:]
    captura = y_real[indices].sum() / y_real.sum() if y_real.sum() else 0.0
    return {
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "Poisson_Deviance": float(mean_poisson_deviance(y_real, y_pred)),
        "Captura_Top10": float(captura),
        "Lift_Top10": float(captura / 0.10),
        "Total_estimado": float(y_pred.sum()),
    }


def main() -> None:
    for caminho in (ARQUIVO_REAL, ARQUIVO_PREVISAO, ARQUIVO_HISTORICO):
        if not caminho.exists():
            raise FileNotFoundError(caminho)

    print("=" * 70)
    print("VALIDAÇÃO EXTERNA — JANEIRO DE 2026")
    print("=" * 70)

    previsoes = pd.read_csv(ARQUIVO_PREVISAO)
    if not previsoes["trecho_id"].is_unique:
        raise ValueError("Há mais de uma previsão por trecho.")

    real = pd.read_csv(
        ARQUIVO_REAL, sep=";", encoding="latin1",
        usecols=["id", "data_inversa", "uf", "br", "km"],
        dtype="string", low_memory=False,
    )
    real["data"] = pd.to_datetime(real["data_inversa"], errors="coerce")
    real["br_num"] = pd.to_numeric(real["br"].str.extract(r"(\d{1,3})", expand=False), errors="coerce")
    real["km_num"] = pd.to_numeric(real["km"].str.replace(",", ".", regex=False), errors="coerce")
    real["uf"] = real["uf"].str.strip().str.upper()

    janeiro = real[
        (real["data"] >= "2026-01-01") & (real["data"] < "2026-02-01")
        & real["uf"].str.fullmatch(r"[A-Z]{2}", na=False)
        & real["br_num"].between(1, 999)
        & real["km_num"].notna() & (real["km_num"] >= 0)
    ].copy()
    janeiro["br"] = janeiro["br_num"].astype(int).astype(str).str.zfill(3)
    janeiro["km_inicio"] = ((janeiro["km_num"] // 10) * 10).astype(int)
    janeiro["trecho_id"] = janeiro["uf"] + "_BR" + janeiro["br"] + "_KM" + janeiro["km_inicio"].astype(str)

    acidentes_reais = janeiro.groupby("trecho_id").size().rename("acidentes_reais").reset_index()
    avaliacao = previsoes[["trecho_id", "acidentes_estimados"]].merge(
        acidentes_reais, on="trecho_id", how="left", validate="one_to_one"
    )
    avaliacao["acidentes_reais"] = avaliacao["acidentes_reais"].fillna(0)

    historico = pd.read_csv(ARQUIVO_HISTORICO)
    historico_recente = historico[historico["ano_mes"].isin(["2025-10", "2025-11", "2025-12"])]
    baseline = historico_recente.groupby("trecho_id")["qtd_acidentes"].mean().rename("media_movel_3m").reset_index()
    avaliacao = avaliacao.merge(baseline, on="trecho_id", how="left", validate="one_to_one")
    if avaliacao["media_movel_3m"].isna().any():
        faltantes = int(avaliacao["media_movel_3m"].isna().sum())
        raise ValueError(f"{faltantes} trechos previstos não possuem baseline calculável de três meses.")

    modelo = metricas(avaliacao["acidentes_reais"], avaliacao["acidentes_estimados"])
    media_movel = metricas(avaliacao["acidentes_reais"], avaliacao["media_movel_3m"])
    tabela = pd.DataFrame([
        {"modelo": "HistGradientBoosting", **modelo},
        {"modelo": "Média móvel 3 meses", **media_movel},
    ])

    total_real = len(janeiro)
    cobertos = int(avaliacao["acidentes_reais"].sum())
    sem_previsao = total_real - cobertos
    cobertura = cobertos / total_real if total_real else 0.0

    avaliacao["erro_absoluto"] = (avaliacao["acidentes_estimados"] - avaliacao["acidentes_reais"]).abs()
    avaliacao = avaliacao.sort_values("acidentes_reais", ascending=False)
    avaliacao.to_csv(DADOS / "validacao_completa_jan2026.csv", index=False, encoding="utf-8-sig")
    tabela.to_csv(DADOS / "resultado_validacao_jan2026.csv", index=False, encoding="utf-8-sig")

    print("\nCOBERTURA")
    print(f"Acidentes PRF com identificação rodoviária válida: {total_real:,}")
    print(f"Acidentes em trechos previstos: {cobertos:,}")
    print(f"Acidentes em trechos sem previsão: {sem_previsao:,}")
    print(f"Cobertura: {cobertura:.2%}")
    print("\nCOMPARAÇÃO")
    print(tabela.to_string(index=False))
    print("\nTRECHOS COM MAIS ACIDENTES REAIS")
    print(avaliacao[["trecho_id", "acidentes_estimados", "acidentes_reais", "erro_absoluto"]].head(15).to_string(index=False))
    print("\nArquivos gerados: validacao_completa_jan2026.csv e resultado_validacao_jan2026.csv")


if __name__ == "__main__":
    main()
