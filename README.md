# Plataforma de Inteligência Viária — MVP reconstruído

Projeto reconstruído a partir dos scripts e resultados desenvolvidos nesta conversa. A aplicação usa dados abertos de acidentes da Polícia Rodoviária Federal (PRF), agrega ocorrências em segmentos de aproximadamente 10 km e competências mensais, treina modelos para estimar a contagem de acidentes do mês seguinte e disponibiliza um dashboard em Streamlit.

> **Importante:** este pacote contém código e documentação, não inclui os CSVs da PRF nem os artefatos de modelo. Os arquivos originais permanecem no repositório `analise-acidentes-prf`. É necessário manter os projetos lado a lado conforme a estrutura abaixo.

## 1. Estrutura esperada no Windows

Extraia esta pasta para `C:\Users\felly\Projetos\plataforma-inteligencia-viaria` e mantenha:

```text
C:\Users\felly\Projetos\
├── analise-acidentes-prf\
│   └── Dados\
│       ├── datatran2017.csv
│       ├── datatran2018.csv
│       ├── datatran2019.csv
│       ├── datatran2020.csv
│       ├── datatran2021.csv
│       ├── datatran2022.csv
│       ├── datatran2023.csv
│       ├── datatran2024.csv
│       ├── datatran2025.csv
│       └── datatran2026.csv   # opcional; usado nas validações externas
│
└── plataforma-inteligencia-viaria\
    ├── app.py
    ├── requirements.txt
    ├── scripts\
    ├── data\processed\
    └── artifacts\
```

O repositório `br-risk-ml` pode permanecer na mesma pasta, mas os scripts deste MVP reconstruído não dependem de uma instalação editável dele.

## 2. Ambiente virtual

No PowerShell:

```powershell
Set-Location "C:\Users\felly\Projetos\plataforma-inteligencia-viaria"
py -3.15 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

O ambiente Python 3.15.2 foi o utilizado na execução relatada pelo usuário. Se a versão 3.15 não estiver instalada em outro computador, use uma versão do Python compatível com as versões das dependências e ajuste o comando do launcher (`py -3.x`).

## 3. Ordem de execução

Execute os comandos abaixo a partir da raiz do projeto.

### Etapa A — auditar os CSVs anuais

```powershell
.\.venv\Scripts\python.exe .\scripts\auditar_dados_prf.py
```

O script procura os arquivos `datatran20xx.csv` em `..\analise-acidentes-prf\Dados` e confere colunas, período, datas e coordenadas.

### Etapa B — gerar a base trecho-mês

```powershell
.\.venv\Scripts\python.exe .\scripts\gerar_base_trecho_mes.py
```

Gera em `data\processed`:

- `acidentes_por_trecho_mes.csv`: base mensal agregada por UF, BR e segmento de 10 km.
- `registros_sem_identificacao_rodoviaria.csv`: ocorrências sem BR/quilometragem/UF/data utilizável para segmentação.
- `registros_coordenadas_nao_validas.csv`: ocorrências cujas coordenadas não devem ser plotadas.
- `resumo_qualidade.csv`: resumo dos controles de qualidade.

A versão V2 preserva ocorrências sem identificação rodoviária em arquivo separado, não cria trechos para BR 000, mantém quilometragens acima de 1.000 km e corrige apenas os dois casos identificados com coordenadas em escala de milionésimos.

### Etapa C — criar variáveis temporais

```powershell
.\.venv\Scripts\python.exe .\scripts\gerar_features_temporais.py
```

Gera:

- `features_treino.csv`
- `features_previsao_jan2026.csv`
- `trechos_sem_historico_suficiente.csv`

O alvo é a quantidade de acidentes na competência seguinte. A simulação atual usa a base encerrada em dezembro de 2025 para estimar janeiro de 2026. Trechos com menos de 12 meses observados ficam fora da previsão experimental.

### Etapa D — treinar e comparar modelos

```powershell
.\.venv\Scripts\python.exe .\scripts\treinar_modelos.py
```

Compara duas baselines (média móvel de três meses e último mês observado), Regressão de Poisson e HistGradientBoosting com perda Poisson. A divisão é temporal: alvos até 2022 para treinamento inicial, 2023 para validação e 2024–2025 para teste.

Gera:

- `data\processed\metricas_modelos.csv`
- `data\processed\previsao_janeiro_2026.csv`
- `data\processed\previsoes_teste_2024_2025.csv`
- `artifacts\modelo_previsao.joblib` (quando o modelo selecionado é de ML)
- `artifacts\config_modelo.json`

A seleção é feita com MAE na validação. MAE, RMSE, Poisson Deviance e captura de acidentes no Top 10% são apresentados como métricas complementares.

### Etapa E — validação externa com janeiro de 2026 (opcional)

Coloque o arquivo oficial `datatran2026.csv` em `analise-acidentes-prf\Dados` e execute:

```powershell
.\.venv\Scripts\python.exe .\scripts\validar_previsao_jan2026.py
.\.venv\Scripts\python.exe .\scripts\validar_previsao_completa_jan2026.py
```

O segundo script compara o resultado previsto com os acidentes reais de janeiro de 2026 para os trechos para os quais existe previsão e compara com uma baseline de média móvel. Gera `validacao_completa_jan2026.csv` e `resultado_validacao_jan2026.csv`.

### Etapa F — iniciar o dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

O Streamlit normalmente abre em `http://localhost:8501`.

## 4. Resultados registrados durante o desenvolvimento anterior

Os resultados abaixo foram relatados em uma execução anterior e servem como referência para conferir a reprodução. Uma nova execução pode produzir pequenas diferenças conforme versões, arquivos e ambiente.

- Registros PRF 2017–2025: 632.667.
- Ocorrências com identificação rodoviária utilizável: 631.263.
- Trechos identificados: 6.855.
- Linhas trecho-mês: 663.918.
- Amostras supervisionadas: 582.087.
- Trechos elegíveis para a previsão experimental de janeiro de 2026: 6.777.
- HistGradientBoosting no teste 2024–2025: MAE 0,6061; RMSE 1,0836; Poisson Deviance 0,8406.
- Validação externa janeiro de 2026: 5.816 ocorrências identificadas na base real; 5.803 em trechos com previsão (cobertura de 99,78%). MAE reportado: 0,5791 para HistGradientBoosting e 0,6088 para a média móvel de três meses.
- SC / BR-101 / km 200–210: previsão 47,65; realizado 49; erro absoluto 1,35 (2,75%).

Esses resultados não equivalem a validação comercial nem provam desempenho futuro em todos os meses e segmentos. A métrica Top 10% na execução original foi calculada de forma agrupada; recomenda-se backtest mês a mês para avaliar priorização operacional.

## 5. Limitações conhecidas

1. O universo de trechos é construído a partir de locais com pelo menos uma ocorrência histórica; não representa toda a malha federal.
2. A identificação do segmento usa UF + BR + faixa de quilometragem. Não diferencia pistas, sentidos ou segmentos descontínuos que compartilhem o mesmo identificador.
3. As coordenadas do mapa são medianas das ocorrências do trecho, não a geometria real da rodovia.
4. A previsão incluída no pacote é retrospectiva para janeiro de 2026. Para uma previsão operacional atual, atualize os dados, refaça o pipeline, gere as features e treine/avalie o modelo com cortes temporais apropriados.
5. A classificação baixo/médio/alto é relativa ao ranking do conjunto previsto; não representa probabilidade de acidente.
6. A contagem-alvo é uma estimativa esperada e pode ser fracionária; não é uma contagem exata de eventos individuais.

## 6. Fontes de dados

Dados Abertos da Polícia Rodoviária Federal: <https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf>.

Use os arquivos Datatran agrupados por ocorrência para contar acidentes. A base agrupada por pessoa tem outra unidade de observação e não deve ser usada para contar ocorrências diretamente.
