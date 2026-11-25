from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DADOS = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(
    page_title="Inteligência Viária",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def carregar_dados():
    arquivos = {
        "historico": DADOS / "acidentes_por_trecho_mes.csv",
        "previsoes": DADOS / "previsao_janeiro_2026.csv",
        "metricas": DADOS / "metricas_modelos.csv",
    }
    ausentes = [str(p) for p in arquivos.values() if not p.exists()]
    if ausentes:
        raise FileNotFoundError("Execute os scripts de processamento e treinamento. Arquivos ausentes: " + ", ".join(ausentes))

    historico = pd.read_csv(arquivos["historico"])
    previsoes = pd.read_csv(arquivos["previsoes"])
    metricas = pd.read_csv(arquivos["metricas"])

    for df in (historico, previsoes):
        df["trecho_id"] = df["trecho_id"].astype(str)
        df["uf"] = df["uf"].astype(str)
        df["br"] = df["br"].astype(str).str.zfill(3)

    coordenadas = historico.groupby("trecho_id").agg(
        latitude=("latitude", "first"), longitude=("longitude", "first")
    ).reset_index()
    previsoes = previsoes.drop(columns=["latitude", "longitude"], errors="ignore")
    previsoes = previsoes.merge(coordenadas, on="trecho_id", how="left", validate="one_to_one")
    return historico, previsoes, metricas


try:
    historico, previsoes, metricas = carregar_dados()
except Exception as erro:
    st.error(f"Não foi possível carregar os arquivos do MVP: {erro}")
    st.stop()

config_path = ARTIFACTS / "config_modelo.json"
if config_path.exists():
    with config_path.open(encoding="utf-8") as f:
        modelo_nome = json.load(f).get("nome", "Não identificado")
else:
    modelo_nome = "Não identificado"

st.title("🛣️ Inteligência Viária")
st.markdown("### Plataforma de análise e previsão de acidentes em rodovias federais brasileiras")
st.caption("Dados históricos da Polícia Rodoviária Federal | Base de treinamento: 2017–2025")
st.warning(
    "Demonstração retrospectiva: a previsão exibida é para janeiro de 2026, usando informações até dezembro de 2025. "
    "Não representa uma previsão operacional atual."
)

st.sidebar.title("Filtros")
ufs = ["Todas"] + sorted(historico["uf"].dropna().unique().tolist())
uf_selecionada = st.sidebar.selectbox("Estado (UF)", ufs)
if uf_selecionada != "Todas":
    historico = historico[historico["uf"] == uf_selecionada]
    previsoes = previsoes[previsoes["uf"] == uf_selecionada]

brs = ["Todas"] + sorted(historico["br"].dropna().unique().tolist())
br_selecionada = st.sidebar.selectbox("Rodovia federal", brs)
if br_selecionada != "Todas":
    historico = historico[historico["br"] == br_selecionada]
    previsoes = previsoes[previsoes["br"] == br_selecionada]
st.sidebar.divider()
st.sidebar.caption(f"Modelo experimental: {modelo_nome}")
st.sidebar.caption("Fonte: Dados Abertos da PRF.")

aba1, aba2, aba3 = st.tabs(["📊 Visão geral", "📍 Mapa e previsões", "🧠 Desempenho do modelo"])

with aba1:
    st.subheader("Panorama dos acidentes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acidentes registrados", f"{int(historico['qtd_acidentes'].sum()):,}".replace(",", "."))
    c2.metric("Mortos", f"{int(historico['qtd_mortos'].sum()):,}".replace(",", "."))
    c3.metric("Acidentes graves", f"{int(historico['qtd_acidentes_graves'].sum()):,}".replace(",", "."))
    c4.metric("Trechos monitorados", f"{historico['trecho_id'].nunique():,}".replace(",", "."))

    st.divider()
    st.subheader("Evolução mensal")
    mensal = historico.groupby("ano_mes").agg(
        acidentes=("qtd_acidentes", "sum"), mortos=("qtd_mortos", "sum")
    ).reset_index().sort_values("ano_mes")
    fig = px.line(mensal, x="ano_mes", y="acidentes", markers=True,
                  title="Quantidade de acidentes por mês",
                  labels={"ano_mes": "Mês", "acidentes": "Acidentes"})
    fig.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rodovias com mais acidentes")
    ranking = historico.groupby(["uf", "br"])["qtd_acidentes"].sum().reset_index().sort_values("qtd_acidentes", ascending=False).head(15)
    ranking["rodovia"] = ranking["uf"] + " - BR-" + ranking["br"].str.lstrip("0")
    fig_rodovias = px.bar(ranking.sort_values("qtd_acidentes"), x="qtd_acidentes", y="rodovia", orientation="h",
                          title="15 rodovias com maior quantidade histórica de acidentes",
                          labels={"qtd_acidentes": "Acidentes", "rodovia": "Rodovia"})
    st.plotly_chart(fig_rodovias, use_container_width=True)
    st.caption("Os indicadores históricos consideram os acidentes com identificação rodoviária válida.")

with aba2:
    st.subheader("Estimativa de risco por trecho")
    st.caption("Horizonte experimental: janeiro de 2026.")
    if previsoes.empty:
        st.warning("Nenhuma previsão encontrada para os filtros selecionados.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Trechos avaliados", f"{len(previsoes):,}".replace(",", "."))
        total_estimado = float(previsoes["acidentes_estimados"].sum())
        c2.metric("Acidentes estimados", f"{total_estimado:,.1f}".replace(",", "X").replace(".", ",").replace("X", "."))
        c3.metric("Trechos de alto risco", f"{int((previsoes['nivel_risco'] == 'Alto').sum()):,}".replace(",", "."))

        st.divider()
        st.subheader("Mapa interativo de risco")
        mapa = previsoes.dropna(subset=["latitude", "longitude"]).copy()
        mapa = mapa[mapa["latitude"].between(-34, 6) & mapa["longitude"].between(-74, -34) & (mapa["acidentes_estimados"] > 0)]
        if not mapa.empty:
            mapa = mapa.nlargest(1500, "acidentes_estimados")
            fig_mapa = px.scatter_map(
                mapa, lat="latitude", lon="longitude", color="nivel_risco",
                size="acidentes_estimados", size_max=16, hover_name="trecho_id",
                hover_data={"uf": True, "br": True, "km_inicio": True, "km_fim": True,
                            "acidentes_estimados": ":.2f", "nivel_risco": True,
                            "latitude": False, "longitude": False},
                color_discrete_map={"Alto": "#E74C3C", "Medio": "#F39C12", "Baixo": "#3498DB"},
                zoom=3, center={"lat": -15.5, "lon": -52.5}, map_style="open-street-map",
            )
            fig_mapa.update_layout(height=600, margin={"l": 0, "r": 0, "t": 10, "b": 0}, legend_title_text="Nível de risco")
            st.plotly_chart(fig_mapa, use_container_width=True)
            st.caption("Os pontos representam coordenadas medianas dos acidentes no trecho, não a geometria exata da rodovia.")
        else:
            st.warning("Não existem coordenadas válidas para os trechos selecionados.")

        st.divider()
        st.subheader("Ranking de trechos prioritários")
        top = previsoes.sort_values("acidentes_estimados", ascending=False).head(20).copy()
        top["rodovia"] = top["uf"] + " - BR-" + top["br"].str.lstrip("0")
        top["trecho"] = "KM " + top["km_inicio"].astype(str) + "–" + top["km_fim"].astype(str)
        top["acidentes_estimados"] = top["acidentes_estimados"].round(2)
        st.dataframe(top[["ranking_risco", "rodovia", "trecho", "acidentes_estimados", "nivel_risco"]].rename(columns={
            "ranking_risco": "Posição", "rodovia": "Rodovia", "trecho": "Trecho",
            "acidentes_estimados": "Acidentes estimados", "nivel_risco": "Nível de risco",
        }), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Exportar previsões CSV", previsoes.to_csv(index=False).encode("utf-8-sig"),
                           file_name="previsoes_inteligencia_viaria.csv", mime="text/csv")

with aba3:
    st.subheader("Avaliação do modelo preditivo")
    st.markdown(f"**Modelo selecionado:** `{modelo_nome}`")
    teste = metricas[metricas["conjunto"] == "teste_2024_2025"].copy()
    if teste.empty:
        st.warning("Métricas do teste não encontradas. Execute treinar_modelos.py.")
    else:
        teste = teste.sort_values("MAE")
        exibicao = teste[["modelo", "MAE", "RMSE", "Poisson_Deviance", "Captura_acidentes_top10", "Lift_top10"]].rename(columns={
            "modelo": "Modelo", "Poisson_Deviance": "Poisson Deviance",
            "Captura_acidentes_top10": "Captura Top 10%", "Lift_top10": "Lift Top 10%",
        })
        st.dataframe(exibicao.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "Poisson Deviance": "{:.4f}", "Captura Top 10%": "{:.2%}", "Lift Top 10%": "{:.2f}"}), use_container_width=True, hide_index=True)
        fig_metricas = px.bar(teste, x="modelo", y="MAE", color="modelo", title="Comparação do erro absoluto médio (MAE)", labels={"modelo": "Modelo", "MAE": "Erro médio absoluto"})
        st.plotly_chart(fig_metricas, use_container_width=True)
        st.info("A captura Top 10% apresentada é calculada agrupando observações de 2024 e 2025. A avaliação operacional mensal deve ser analisada separadamente.")
    st.divider()
    st.subheader("Metodologia de validação")
    st.markdown("""
    | Etapa | Período |
    |---|---|
    | Treinamento inicial | 2018–2022 |
    | Validação | 2023 |
    | Teste independente | 2024–2025 |

    **Variável-alvo:** quantidade de acidentes no trecho durante o mês seguinte.

    **Métricas:** MAE, RMSE, Poisson Deviance e captura de acidentes entre as observações mais prioritárias.

    A classificação de risco é relativa ao conjunto de trechos avaliados e não representa probabilidade individual de acidente.
    """)

st.divider()
st.caption("INTELIGÊNCIA VIÁRIA | Protótipo experimental | Fonte: Polícia Rodoviária Federal — Dados Abertos.")
