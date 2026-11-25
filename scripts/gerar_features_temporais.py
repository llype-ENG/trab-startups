from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DADOS = ROOT / "data" / "processed"
ENTRADA = DADOS / "acidentes_por_trecho_mes.csv"
JANELAS = [3, 6, 12]
LAGS = [1, 2, 3, 6, 12]
HISTORICO_MINIMO = 12


def carregar_base() -> pd.DataFrame:
    if not ENTRADA.exists():
        raise FileNotFoundError(f"Execute primeiro gerar_base_trecho_mes.py: {ENTRADA}")
    df = pd.read_csv(ENTRADA)
    if df.empty:
        raise ValueError("A base de acidentes está vazia.")
    df["periodo"] = pd.PeriodIndex(df["ano_mes"], freq="M")
    df = df.sort_values(["trecho_id", "periodo"]).reset_index(drop=True)
    print(f"Linhas carregadas: {len(df):,} | Trechos: {df['trecho_id'].nunique():,}")
    return df


def validar_sequencia_mensal(df: pd.DataFrame) -> None:
    numero_mes = df["periodo"].dt.year * 12 + df["periodo"].dt.month
    diferenca = numero_mes.groupby(df["trecho_id"]).diff().dropna()
    if not diferenca.eq(1).all():
        raise ValueError("Existem meses ausentes na sequência de algum trecho.")
    print("Todas as sequências mensais estão completas.")


def criar_atributos(df: pd.DataFrame) -> pd.DataFrame:
    print("Criando atributos temporais...")
    grupo = df.groupby("trecho_id", sort=False)
    df["historico_meses"] = grupo.cumcount() + 1
    df["acidentes_mes_atual"] = df["qtd_acidentes"]
    df["graves_mes_atual"] = df["qtd_acidentes_graves"]
    df["fatais_mes_atual"] = df["qtd_acidentes_fatais"]
    df["mortos_mes_atual"] = df["qtd_mortos"]
    df["feridos_mes_atual"] = df["qtd_feridos"]

    for lag in LAGS:
        df[f"acidentes_lag_{lag}"] = (
            df.groupby("trecho_id", sort=False)["qtd_acidentes"].shift(lag)
        )

    for janela in JANELAS:
        acidentes = df.groupby("trecho_id", sort=False)["qtd_acidentes"]
        df[f"acidentes_soma_{janela}m"] = acidentes.transform(
            lambda s: s.rolling(janela, min_periods=janela).sum()
        )
        df[f"acidentes_media_{janela}m"] = acidentes.transform(
            lambda s: s.rolling(janela, min_periods=janela).mean()
        )

    for coluna, nome in [
        ("qtd_acidentes_graves", "graves"),
        ("qtd_acidentes_fatais", "fatais"),
        ("qtd_mortos", "mortos"),
    ]:
        serie = df.groupby("trecho_id", sort=False)[coluna]
        for janela in (3, 12):
            df[f"{nome}_soma_{janela}m"] = serie.transform(
                lambda s: s.rolling(janela, min_periods=janela).sum()
            )

    df["target_acidentes"] = (
        df.groupby("trecho_id", sort=False)["qtd_acidentes"].shift(-1)
    )
    df["periodo_alvo"] = df["periodo"] + 1
    df["target_ano_mes"] = df["periodo_alvo"].astype(str)
    df["mes_previsao"] = df["periodo_alvo"].dt.month
    df["ano_previsao"] = df["periodo_alvo"].dt.year
    df["mes_sin"] = np.sin(2 * np.pi * df["mes_previsao"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes_previsao"] / 12)
    df["km_inicio"] = pd.to_numeric(df["km_inicio"], errors="coerce")
    df["historico_completo"] = df["historico_meses"] >= HISTORICO_MINIMO
    return df


def gerar_datasets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    historico = df[df["historico_completo"]].copy()
    treino = historico[historico["target_acidentes"].notna()].copy()
    treino["target_acidentes"] = treino["target_acidentes"].astype("int64")
    ultimas = df.groupby("trecho_id", sort=False).tail(1).copy()
    previsao = ultimas[ultimas["historico_completo"]].copy()
    sem_historico = ultimas[~ultimas["historico_completo"]].copy()

    if not previsao.empty and not previsao["target_ano_mes"].eq("2026-01").all():
        raise ValueError("A previsão atual deveria ter alvo em 2026-01; confira a última competência da base.")

    colunas = [
        "trecho_id", "uf", "br", "km_inicio", "km_fim", "ano_mes", "target_ano_mes",
        "historico_meses", "acidentes_mes_atual", "graves_mes_atual", "fatais_mes_atual",
        "mortos_mes_atual", "feridos_mes_atual", "acidentes_lag_1", "acidentes_lag_2",
        "acidentes_lag_3", "acidentes_lag_6", "acidentes_lag_12", "acidentes_soma_3m",
        "acidentes_soma_6m", "acidentes_soma_12m", "acidentes_media_3m", "acidentes_media_6m",
        "acidentes_media_12m", "graves_soma_3m", "graves_soma_12m", "fatais_soma_3m",
        "fatais_soma_12m", "mortos_soma_3m", "mortos_soma_12m", "mes_previsao",
        "ano_previsao", "mes_sin", "mes_cos", "target_acidentes",
    ]
    treino[colunas].to_csv(DADOS / "features_treino.csv", index=False, encoding="utf-8-sig")
    previsao[colunas[:-1]].to_csv(DADOS / "features_previsao_jan2026.csv", index=False, encoding="utf-8-sig")
    sem_historico[["trecho_id", "uf", "br", "km_inicio", "km_fim", "ano_mes", "historico_meses"]].to_csv(
        DADOS / "trechos_sem_historico_suficiente.csv", index=False, encoding="utf-8-sig"
    )
    return treino, previsao, sem_historico


def main() -> None:
    print("=" * 65)
    print("PLATAFORMA DE INTELIGÊNCIA VIÁRIA | FEATURES TEMPORAIS")
    print("=" * 65)
    df = carregar_base()
    validar_sequencia_mensal(df)
    df = criar_atributos(df)
    treino, previsao, sem_historico = gerar_datasets(df)
    assert treino["target_acidentes"].notna().all()
    assert (treino["target_acidentes"] >= 0).all()
    assert treino[["acidentes_media_3m", "acidentes_media_6m", "acidentes_media_12m"]].notna().all().all()
    print("\nENGENHARIA DE ATRIBUTOS CONCLUÍDA")
    print(f"Amostras de treinamento: {len(treino):,}")
    print(f"Trechos para previsão: {len(previsao):,}")
    print(f"Trechos sem 12 meses de histórico: {len(sem_historico):,}")
    print(f"Período dos alvos: {treino['target_ano_mes'].min()} até {treino['target_ano_mes'].max()}")
    print("Arquivos: features_treino.csv, features_previsao_jan2026.csv, trechos_sem_historico_suficiente.csv")


if __name__ == "__main__":
    main()
