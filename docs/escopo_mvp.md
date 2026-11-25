# Escopo técnico do MVP

## Objetivo
Estimar a quantidade de acidentes por segmento rodoviário de aproximadamente 10 km para a competência mensal seguinte, utilizando dados históricos da PRF.

## Unidade de análise
Uma linha representa um trecho identificado por UF, BR e faixa de quilometragem em uma competência mensal. A identificação não substitui uma malha rodoviária georreferenciada oficial.

## Dados
Datatran agrupado por ocorrência, 2017–2025. As ocorrências sem identificação rodoviária válida são preservadas em arquivo separado e não entram na agregação trecho-mês.

## Modelo
Modelos comparados: média móvel de três meses, último mês observado, Regressão de Poisson e HistGradientBoosting com perda Poisson. A seleção usa validação temporal em 2023; teste final histórico em 2024–2025.

## Interface
Aplicação Streamlit com visão histórica, mapa aproximado, estimativas por trecho, ranking relativo e métricas do modelo.

## Limitações
A versão reconstruída reproduz a demonstração retrospectiva para janeiro de 2026. Não deve ser apresentada como previsão operacional atual sem atualização dos dados e novo processo de treinamento e validação.
