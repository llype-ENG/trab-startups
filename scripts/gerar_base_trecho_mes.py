from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DADOS_PRF = ROOT / "data" / "raw"
SAIDA = ROOT / "data" / "processed"
SAIDA.mkdir(parents=True, exist_ok=True)

ANOS = range(2017, 2026)
COLUNAS = [
    "id", "data_inversa", "uf", "br", "km", "mortos", "feridos",
    "feridos_graves", "latitude", "longitude",
]


def converter_numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(
        serie.astype("string").str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def carregar_dados() -> pd.DataFrame:
    frames = []
    for ano in ANOS:
        arquivo = DADOS_PRF / f"datatran{ano}.csv"
        if not arquivo.exists():
            raise FileNotFoundError(
                f"Arquivo nÃ£o encontrado: {arquivo}. Coloque os nove CSVs 2017â€“2025 em {DADOS_PRF}."
            )
        print(f"Lendo {arquivo.name}...")
        df = pd.read_csv(
            arquivo, sep=";", encoding="latin1", usecols=COLUNAS,
            dtype="string", low_memory=False,
        )
        df["ano_arquivo"] = ano
        frames.append(df)
        print(f"  Registros: {len(df):,}")
    dados = pd.concat(frames, ignore_index=True)
    print(f"\nTotal carregado: {len(dados):,}")
    return dados


def preparar_dados(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\nPreparando e validando dados...")
    df = df.copy()
    df["data"] = pd.to_datetime(df["data_inversa"], format="%Y-%m-%d", errors="coerce")
    df["uf"] = df["uf"].str.strip().str.upper()
    br_extraida = df["br"].str.extract(r"(\d{1,3})", expand=False)
    df["br_num"] = pd.to_numeric(br_extraida, errors="coerce")
    df["km_num"] = converter_numero(df["km"])

    lat = converter_numero(df["latitude"])
    lon = converter_numero(df["longitude"])
    valida_original = lat.between(-34, 6) & lon.between(-74, -34)
    zero_zero = lat.eq(0) & lon.eq(0)
    lat_milhao = lat / 1_000_000
    lon_milhao = lon / 1_000_000
    corrigida_milhao = (
        ~valida_original & ~zero_zero
        & ((lat.abs() > 90) | (lon.abs() > 180))
        & lat_milhao.between(-34, 6)
        & lon_milhao.between(-74, -34)
    )
    lat_final = lat.copy()
    lon_final = lon.copy()
    lat_final.loc[corrigida_milhao] = lat_milhao.loc[corrigida_milhao]
    lon_final.loc[corrigida_milhao] = lon_milhao.loc[corrigida_milhao]
    coordenadas_validas = valida_original | corrigida_milhao

    df["latitude_mapa"] = lat_final.where(coordenadas_validas)
    df["longitude_mapa"] = lon_final.where(coordenadas_validas)
    df["status_coordenada"] = np.select(
        [valida_original, corrigida_milhao, zero_zero, lat.isna() | lon.isna()],
        ["valida_original", "corrigida_milhoes", "zero_zero", "ausente"],
        default="invalida",
    )

    for coluna in ["mortos", "feridos", "feridos_graves"]:
        df[coluna] = converter_numero(df[coluna]).fillna(0).clip(lower=0)

    br_valida = df["br_num"].between(1, 999)
    km_valido = df["km_num"].notna() & (df["km_num"] >= 0)
    uf_valida = df["uf"].str.fullmatch(r"[A-Z]{2}", na=False)
    data_valida = df["data"].notna()

    df["motivo_exclusao"] = ""
    df.loc[~br_valida, "motivo_exclusao"] = "br_ausente_ou_zero"
    df.loc[br_valida & ~km_valido, "motivo_exclusao"] = "quilometragem_invalida"
    df.loc[br_valida & km_valido & ~uf_valida, "motivo_exclusao"] = "uf_invalida"
    df.loc[br_valida & km_valido & uf_valida & ~data_valida, "motivo_exclusao"] = "data_invalida"

    mascara = br_valida & km_valido & uf_valida & data_valida
    excluidos = df.loc[~mascara].copy()
    excluidos.to_csv(SAIDA / "registros_sem_identificacao_rodoviaria.csv", index=False, encoding="utf-8-sig")
    dados = df.loc[mascara].copy()

    dados["br"] = dados["br_num"].astype(int).astype(str).str.zfill(3)
    dados["km_inicio"] = ((dados["km_num"] // 10) * 10).astype(int)
    dados["km_fim"] = dados["km_inicio"] + 10
    dados["trecho_id"] = (
        dados["uf"] + "_BR" + dados["br"] + "_KM" + dados["km_inicio"].astype(str)
    )
    dados["ano_mes"] = dados["data"].dt.to_period("M").astype(str)
    dados["acidente_fatal"] = (dados["mortos"] > 0).astype(int)
    dados["acidente_grave"] = (
        (dados["mortos"] > 0) | (dados["feridos_graves"] > 0)
    ).astype(int)

    coordenadas_ruins = df.loc[
        ~coordenadas_validas,
        ["id", "data_inversa", "uf", "br", "km", "latitude", "longitude", "status_coordenada"],
    ].copy()
    coordenadas_ruins.to_csv(
        SAIDA / "registros_coordenadas_nao_validas.csv",
        index=False, encoding="utf-8-sig",
    )

    print(f"Registros identificados: {len(dados):,}")
    print(f"Registros sem identificaÃ§Ã£o: {len(excluidos):,}")
    print(f"Coordenadas corrigidas: {int(corrigida_milhao.sum())}")
    print(f"Coordenadas nÃ£o mapeÃ¡veis: {len(coordenadas_ruins):,}")
    return dados, excluidos


def agregar_acidentes(df: pd.DataFrame) -> pd.DataFrame:
    print("\nAgregando por trecho e mÃªs...")
    mensal = (
        df.groupby(["trecho_id", "ano_mes"], observed=True)
        .agg(
            qtd_acidentes=("data", "size"),
            qtd_acidentes_fatais=("acidente_fatal", "sum"),
            qtd_acidentes_graves=("acidente_grave", "sum"),
            qtd_mortos=("mortos", "sum"),
            qtd_feridos=("feridos", "sum"),
        ).reset_index()
    )
    segmentos = (
        df.groupby("trecho_id", observed=True)
        .agg(
            uf=("uf", "first"), br=("br", "first"),
            km_inicio=("km_inicio", "first"), km_fim=("km_fim", "first"),
            latitude=("latitude_mapa", "median"),
            longitude=("longitude_mapa", "median"),
            primeiro_mes=("ano_mes", "min"),
        ).reset_index()
    )
    calendario = pd.DataFrame({
        "ano_mes": pd.period_range("2017-01", "2025-12", freq="M").astype(str)
    })
    segmentos["_chave"] = 1
    calendario["_chave"] = 1
    grade = segmentos[["trecho_id", "primeiro_mes", "_chave"]].merge(calendario, on="_chave")
    grade = grade.loc[grade["ano_mes"] >= grade["primeiro_mes"]].drop(columns=["primeiro_mes", "_chave"])
    base = grade.merge(mensal, on=["trecho_id", "ano_mes"], how="left", validate="one_to_one")
    metricas = [
        "qtd_acidentes", "qtd_acidentes_fatais", "qtd_acidentes_graves",
        "qtd_mortos", "qtd_feridos",
    ]
    base[metricas] = base[metricas].fillna(0).astype("int64")
    base = base.merge(
        segmentos.drop(columns="primeiro_mes"),
        on="trecho_id", how="left", validate="many_to_one",
    )
    colunas = [
        "trecho_id", "uf", "br", "km_inicio", "km_fim", "ano_mes",
        "qtd_acidentes", "qtd_acidentes_fatais", "qtd_acidentes_graves",
        "qtd_mortos", "qtd_feridos", "latitude", "longitude",
    ]
    return base[colunas].sort_values(["trecho_id", "ano_mes"]).reset_index(drop=True)


def main() -> None:
    print("=" * 65)
    print("PLATAFORMA DE INTELIGÃŠNCIA VIÃRIA | PIPELINE V2")
    print("=" * 65)
    brutos = carregar_dados()
    dados, excluidos = preparar_dados(brutos)
    base = agregar_acidentes(dados)

    duplicados = int(base.duplicated(["trecho_id", "ano_mes"]).sum())
    total_agregado = int(base["qtd_acidentes"].sum())
    assert not (base["br"] == "000").any()
    assert duplicados == 0
    assert total_agregado == len(dados)

    arquivo = SAIDA / "acidentes_por_trecho_mes.csv"
    base.to_csv(arquivo, index=False, encoding="utf-8-sig")
    resumo = pd.DataFrame([
        ["total_original", len(brutos)],
        ["registros_modelados", len(dados)],
        ["registros_sem_identificacao", len(excluidos)],
        ["trechos_identificados", base["trecho_id"].nunique()],
        ["linhas_trecho_mes", len(base)],
        ["acidentes_agregados", total_agregado],
        ["chaves_duplicadas", duplicados],
        ["meses_sem_acidentes", int((base["qtd_acidentes"] == 0).sum())],
        ["coordenadas_ausentes_trechos", int(base["latitude"].isna().sum())],
    ], columns=["indicador", "valor"])
    resumo.to_csv(SAIDA / "resumo_qualidade.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 65)
    print("PIPELINE CONCLUÃDO")
    print("=" * 65)
    print(resumo.to_string(index=False))
    print(f"\nArquivo principal: {arquivo}")


if __name__ == "__main__":
    main()

