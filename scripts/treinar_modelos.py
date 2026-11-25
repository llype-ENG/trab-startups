from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DADOS = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

ARQUIVO_TREINO = DADOS / "features_treino.csv"
ARQUIVO_PREVISAO = DADOS / "features_previsao_jan2026.csv"
TARGET = "target_acidentes"
CATEGORICAS = ["uf", "br"]
NUMERICAS = [
    "km_inicio", "historico_meses", "acidentes_mes_atual", "graves_mes_atual",
    "fatais_mes_atual", "mortos_mes_atual", "feridos_mes_atual", "acidentes_lag_1",
    "acidentes_lag_2", "acidentes_lag_3", "acidentes_lag_6", "acidentes_lag_12",
    "acidentes_soma_3m", "acidentes_soma_6m", "acidentes_soma_12m",
    "acidentes_media_3m", "acidentes_media_6m", "acidentes_media_12m",
    "graves_soma_3m", "graves_soma_12m", "fatais_soma_3m", "fatais_soma_12m",
    "mortos_soma_3m", "mortos_soma_12m", "mes_previsao", "ano_previsao", "mes_sin", "mes_cos",
]
FEATURES = CATEGORICAS + NUMERICAS
RANDOM_STATE = 42


def carregar_dados():
    if not ARQUIVO_TREINO.exists() or not ARQUIVO_PREVISAO.exists():
        raise FileNotFoundError("Execute gerar_features_temporais.py antes do treinamento.")
    treino = pd.read_csv(ARQUIVO_TREINO)
    previsao = pd.read_csv(ARQUIVO_PREVISAO)
    if treino.empty or previsao.empty:
        raise ValueError("Um dos datasets está vazio.")
    for coluna in NUMERICAS:
        treino[coluna] = pd.to_numeric(treino[coluna], errors="coerce")
        previsao[coluna] = pd.to_numeric(previsao[coluna], errors="coerce")
    treino[TARGET] = pd.to_numeric(treino[TARGET], errors="raise")
    treino["target_ano_mes"] = treino["target_ano_mes"].astype(str)
    previsao["target_ano_mes"] = previsao["target_ano_mes"].astype(str)
    categorias = {col: sorted(treino[col].fillna("NA").astype(str).unique().tolist()) for col in CATEGORICAS}
    for coluna in CATEGORICAS:
        treino[coluna] = pd.Categorical(treino[coluna].fillna("NA").astype(str), categories=categorias[coluna])
        previsao[coluna] = pd.Categorical(previsao[coluna].fillna("NA").astype(str), categories=categorias[coluna])
    return treino, previsao


def criar_modelos():
    preprocessador = ColumnTransformer(
        transformers=[
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
            ("numericas", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERICAS),
        ],
        sparse_threshold=1.0,
    )
    poisson = Pipeline([
        ("preprocessador", preprocessador),
        ("modelo", PoissonRegressor(alpha=0.5, max_iter=300, tol=1e-5)),
    ])
    gradient = HistGradientBoostingRegressor(
        loss="poisson", categorical_features="from_dtype",
        learning_rate=0.08, max_iter=120, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=False, random_state=RANDOM_STATE,
    )
    return {"regressao_poisson": poisson, "gradient_boosting_poisson": gradient}


def predizer_baseline(df: pd.DataFrame, nome: str) -> np.ndarray:
    if nome == "media_movel_3m":
        return df["acidentes_media_3m"].to_numpy(dtype=float)
    if nome == "ultimo_mes":
        return df["acidentes_lag_1"].to_numpy(dtype=float)
    raise ValueError(f"Baseline desconhecida: {nome}")


def calcular_metricas(y_real, y_pred) -> dict[str, float]:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-9)
    k = max(1, int(np.ceil(len(y_real) * 0.10)))
    indices = np.argsort(y_pred, kind="stable")[-k:]
    captura = y_real[indices].sum() / y_real.sum() if y_real.sum() > 0 else 0.0
    return {
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "Poisson_Deviance": float(mean_poisson_deviance(y_real, y_pred)),
        "Media_real": float(y_real.mean()),
        "Media_prevista": float(y_pred.mean()),
        "Captura_acidentes_top10": float(captura),
        "Lift_top10": float(captura / 0.10),
    }


def avaliar_baselines(df, conjunto, resultados):
    y = df[TARGET].to_numpy(dtype=float)
    for nome in ("media_movel_3m", "ultimo_mes"):
        previsao = predizer_baseline(df, nome)
        resultados.append({"conjunto": conjunto, "modelo": nome, **calcular_metricas(y, previsao)})


def avaliar_modelos(modelos, X_treino, y_treino, X_avaliacao, y_avaliacao, conjunto, resultados):
    for nome, estimador in modelos.items():
        print(f"\nTreinando {nome}...")
        modelo = clone(estimador)
        try:
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always")
                modelo.fit(X_treino, y_treino)
                previsao = modelo.predict(X_avaliacao)
                for aviso in avisos:
                    print(f"Aviso de {nome}: {aviso.message}")
            metricas = calcular_metricas(y_avaliacao, previsao)
            resultados.append({"conjunto": conjunto, "modelo": nome, **metricas})
            print(f"{conjunto} | {nome} | MAE: {metricas['MAE']:.4f} | RMSE: {metricas['RMSE']:.4f}")
        except Exception as erro:
            print(f"ERRO no modelo {nome}: {erro}")


def main():
    print("=" * 70)
    print("PLATAFORMA DE INTELIGÊNCIA VIÁRIA | TREINAMENTO")
    print("=" * 70)
    df, previsao_jan = carregar_dados()
    X = df[FEATURES]
    y = df[TARGET].to_numpy(dtype=float)
    datas = df["target_ano_mes"]
    mascara_treino = datas <= "2022-12"
    mascara_validacao = datas.str.startswith("2023")
    mascara_teste = (datas >= "2024-01") & (datas <= "2025-12")
    if any(mask.sum() == 0 for mask in (mascara_treino, mascara_validacao, mascara_teste)):
        raise ValueError("Uma das partições temporais ficou vazia.")

    X_treino, y_treino = X.loc[mascara_treino], y[mascara_treino]
    X_validacao, y_validacao = X.loc[mascara_validacao], y[mascara_validacao]
    X_teste, y_teste = X.loc[mascara_teste], y[mascara_teste]
    print(f"Amostras: treino={len(X_treino):,} | validação={len(X_validacao):,} | teste={len(X_teste):,}")
    resultados = []
    modelos = criar_modelos()

    print("\nETAPA 1 — VALIDAÇÃO 2023")
    avaliar_baselines(df.loc[mascara_validacao], "validacao_2023", resultados)
    avaliar_modelos(modelos, X_treino, y_treino, X_validacao, y_validacao, "validacao_2023", resultados)
    validacao = [r for r in resultados if r["conjunto"] == "validacao_2023"]
    if not validacao:
        raise RuntimeError("Nenhum resultado disponível na validação.")
    vencedor = min(validacao, key=lambda r: r["MAE"])
    melhor_nome = vencedor["modelo"]
    print(f"\nModelo selecionado pela validação: {melhor_nome} | MAE: {vencedor['MAE']:.4f}")

    print("\nETAPA 2 — TESTE 2024–2025")
    mascara_treino_validacao = mascara_treino | mascara_validacao
    X_tv, y_tv = X.loc[mascara_treino_validacao], y[mascara_treino_validacao]
    avaliar_baselines(df.loc[mascara_teste], "teste_2024_2025", resultados)
    avaliar_modelos(modelos, X_tv, y_tv, X_teste, y_teste, "teste_2024_2025", resultados)

    metricas = pd.DataFrame(resultados)
    metricas.to_csv(DADOS / "metricas_modelos.csv", index=False, encoding="utf-8-sig")
    print("\nCOMPARAÇÃO DOS RESULTADOS")
    print(metricas[["conjunto", "modelo", "MAE", "RMSE", "Poisson_Deviance", "Captura_acidentes_top10", "Lift_top10"]].to_string(index=False))

    print("\nETAPA 3 — PREVISÃO EXPERIMENTAL PARA JANEIRO DE 2026")
    if melhor_nome in ("media_movel_3m", "ultimo_mes"):
        y_previsto = predizer_baseline(previsao_jan, melhor_nome)
        artefato = {"tipo": "baseline", "nome": melhor_nome}
    else:
        modelo_final = clone(modelos[melhor_nome])
        modelo_final.fit(X, y)
        y_previsto = modelo_final.predict(previsao_jan[FEATURES])
        joblib.dump(modelo_final, ARTIFACTS / "modelo_previsao.joblib")
        artefato = {"tipo": "machine_learning", "nome": melhor_nome, "arquivo": "modelo_previsao.joblib"}

    y_previsto = np.maximum(y_previsto, 0)
    resultado = previsao_jan[["trecho_id", "uf", "br", "km_inicio", "km_fim", "ano_mes", "target_ano_mes"]].copy()
    resultado["acidentes_estimados"] = y_previsto
    resultado["ranking_risco"] = resultado["acidentes_estimados"].rank(method="first", ascending=False).astype(int)
    percentual = resultado["acidentes_estimados"].rank(pct=True, method="first")
    resultado["nivel_risco"] = np.select([percentual >= 2 / 3, percentual >= 1 / 3], ["Alto", "Medio"], default="Baixo")
    resultado = resultado.sort_values("ranking_risco")
    resultado.to_csv(DADOS / "previsao_janeiro_2026.csv", index=False, encoding="utf-8-sig")

    # Exporta previsões no período de teste para auditoria adicional.
    if melhor_nome in modelos:
        modelo_teste = clone(modelos[melhor_nome]).fit(X_tv, y_tv)
        y_teste_previsto = modelo_teste.predict(X_teste)
    else:
        y_teste_previsto = predizer_baseline(df.loc[mascara_teste], melhor_nome)
    resultado_teste = df.loc[mascara_teste, ["trecho_id", "ano_mes", "target_ano_mes"]].copy()
    resultado_teste["acidentes_reais"] = y_teste
    resultado_teste["acidentes_estimados"] = np.maximum(y_teste_previsto, 0)
    resultado_teste.to_csv(DADOS / "previsoes_teste_2024_2025.csv", index=False, encoding="utf-8-sig")

    with (ARTIFACTS / "config_modelo.json").open("w", encoding="utf-8") as f:
        json.dump(artefato, f, ensure_ascii=False, indent=4)

    print(f"\nModelo selecionado: {melhor_nome}")
    print(f"Trechos previstos: {len(resultado):,}")
    print(f"Total estimado: {resultado['acidentes_estimados'].sum():.2f}")
    print("\nTOP 10 TRECHOS")
    print(resultado[["trecho_id", "acidentes_estimados", "nivel_risco", "ranking_risco"]].head(10).to_string(index=False))
    print("\nArquivos gerados em data/processed e artifacts.")


if __name__ == "__main__":
    main()
