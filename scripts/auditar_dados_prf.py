from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRETORIO = ROOT.parent / "analise-acidentes-prf" / "Dados"


def converter_data(valor: str | None):
    valor = (valor or "").strip()
    for formato in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(valor[:10], formato).date()
        except ValueError:
            continue
    return None


def converter_float(valor: str | None):
    try:
        return float(str(valor).strip().replace(",", "."))
    except (ValueError, TypeError):
        return None


def main() -> None:
    arquivos = sorted(DIRETORIO.glob("datatran20*.csv"))
    if not arquivos:
        raise SystemExit(f"Nenhum datatran20xx.csv encontrado em {DIRETORIO}")

    referencia_colunas: set[str] | None = None
    total_geral = 0
    anos_encontrados: set[int] = set()

    print("=" * 75)
    print("AUDITORIA DOS DADOS ABERTOS DA PRF")
    print(f"Diretório: {DIRETORIO}")
    print("=" * 75)

    for arquivo in arquivos:
        match = re.search(r"20\d{2}", arquivo.stem)
        ano_arquivo = int(match.group()) if match else None
        if ano_arquivo:
            anos_encontrados.add(ano_arquivo)

        total = datas_validas = datas_invalidas = 0
        coordenadas_validas = coordenadas_ausentes = coordenadas_invalidas = 0
        anos_divergentes = 0
        data_minima = data_maxima = None

        print(f"\nArquivo: {arquivo.name} ({arquivo.stat().st_size / 1024**2:.2f} MB)")
        with arquivo.open("r", encoding="latin1", newline="") as f:
            leitor = csv.reader(f, delimiter=";")
            cabecalho = next(leitor, None)
            if not cabecalho:
                print("ERRO: arquivo sem cabeçalho.")
                continue

            colunas = [str(c).strip().strip('"').lstrip("\ufeff").lower() for c in cabecalho]
            indices = {nome: i for i, nome in enumerate(colunas)}
            essenciais = ["data_inversa", "uf", "br", "km", "latitude", "longitude"]
            faltantes = [c for c in essenciais if c not in indices]
            print(f"Colunas: {len(colunas)} | Ausentes: {faltantes or 'Nenhuma'}")

            if referencia_colunas is None:
                referencia_colunas = set(colunas)
            elif set(colunas) != referencia_colunas:
                print("ATENÇÃO: esquema diferente dos demais arquivos.")
                print("Ausentes:", sorted(referencia_colunas - set(colunas)))
                print("Adicionais:", sorted(set(colunas) - referencia_colunas))

            if faltantes:
                continue

            for linha in leitor:
                if not linha:
                    continue
                total += 1

                def obter(campo: str) -> str:
                    indice = indices[campo]
                    return linha[indice].strip() if indice < len(linha) else ""

                data = converter_data(obter("data_inversa"))
                if data:
                    datas_validas += 1
                    data_minima = min(data, data_minima) if data_minima else data
                    data_maxima = max(data, data_maxima) if data_maxima else data
                    if ano_arquivo and data.year != ano_arquivo:
                        anos_divergentes += 1
                else:
                    datas_invalidas += 1

                lat = converter_float(obter("latitude"))
                lon = converter_float(obter("longitude"))
                if lat is None or lon is None:
                    coordenadas_ausentes += 1
                elif -34 <= lat <= 6 and -74 <= lon <= -34:
                    coordenadas_validas += 1
                else:
                    coordenadas_invalidas += 1

        total_geral += total
        print(f"Registros: {total:,}")
        print(f"Datas válidas/inválidas: {datas_validas:,}/{datas_invalidas:,}")
        print(f"Período: {data_minima} até {data_maxima}")
        print(f"Coordenadas válidas/ausentes/fora: {coordenadas_validas:,}/{coordenadas_ausentes:,}/{coordenadas_invalidas:,}")
        print(f"Datas de outro ano: {anos_divergentes:,}")

    print("\n" + "=" * 75)
    print(f"TOTAL DE REGISTROS: {total_geral:,}")
    print("Anos encontrados:", sorted(anos_encontrados))
    print("Anos ausentes 2017–2025:", sorted(set(range(2017, 2026)) - anos_encontrados))
    print("=" * 75)


if __name__ == "__main__":
    main()
