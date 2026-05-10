"""
====================================================================================
  DASHBOARD DE CURTAILMENT MAURITI - PowerChina    (v3)
====================================================================================

Gera um dashboard HTML standalone do Complexo Fotovoltaico Mauriti com
benchmarking contra usinas do Ceara, atualizado a partir de dados publicos
do ONS e CCEE.

Novidades v3:
  - Layout editorial light (estilo annual report)
  - Tracker do mes corrente (acompanhamento semanal): expectativa vs realizada
    + CF% diario com referencia trimestral
  - Heatmap CF% hora x dia (revela padrao sistemico vs local)
  - Benchmark fixo por grupo nomeado (cada grupo = 1 linha agregada)
  - Cache inteligente: re-baixa os 3 meses mais recentes a cada execucao
  - Output em ./public/index.html (pronto para GitHub Pages)

Como usar
---------
    pip install -r requirements.txt
    python gerar_dashboard_curtailment.py
    # abra public/index.html no navegador

Para publicacao automatica via GitHub Actions, veja o README.md.
====================================================================================
"""

from __future__ import annotations

import json
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from jinja2 import Template
from tqdm import tqdm

import plotly.graph_objects as go
import plotly.io as pio


# =============================================================================
#  CONFIGURACAO
# =============================================================================

CONFIG: dict[str, Any] = {
    # Periodo de analise (AAAA-MM-DD). data_fim = None usa hoje.
    "data_inicio": "2025-07-01",
    "data_fim":    None,

    # Protagonista do estudo
    "mauriti_match": "MAURITI",

    # Benchmark - lista travada de grupos (cada grupo = 1 linha no chart)
    # Cada grupo agrega todas as usinas que casarem com 'match' (substring no
    # nom_usina, case/acento-insensitivel). 'fonte' so e usada pra colorir.
    "benchmark_groups": [
        {"label": "Abaiara 230 kV",  "match": "ABAIARA",       "fonte": "UFV"},
        {"label": "Lins",            "match": "LINS",          "fonte": "UFV"},
        {"label": "Calcario",        "match": "CALCARIO",      "fonte": "UFV"},
        {"label": "Alex",            "match": "ALEX",          "fonte": "UFV"},
        {"label": "Banabuiu",        "match": "BANABUIU",      "fonte": "EOL"},
        {"label": "Serra do Mato",   "match": "SERRA DO MATO", "fonte": "EOL"},
        {"label": "Sol do Futuro",   "match": "SOL DO FUTURO", "fonte": "UFV"},
    ],

    "submercado": "NE",
    "cache_dir":  "./cache",
    "output_html": "./public/index.html",

    # Cache: re-baixar sempre os N meses mais recentes (consistencia recorrente ONS)
    "refresh_recent_n": 3,

    "request_timeout": 60,
    "max_retries":     3,
}


# =============================================================================
#  CONSTANTES (URLs publicas)
# =============================================================================

ONS_BASE = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset"

URL_DETAIL_EOL = (f"{ONS_BASE}/restricao_coff_eolica_detail_tm/"
                  "RESTRICAO_COFF_EOLICA_DETAIL_{year}_{month:02d}.csv")
URL_DETAIL_UFV = (f"{ONS_BASE}/restricao_coff_fotovoltaica_detail_tm/"
                  "RESTRICAO_COFF_FOTOVOLTAICA_DETAIL_{year}_{month:02d}.csv")
URL_CONS_EOL = (f"{ONS_BASE}/restricao_coff_eolica_tm/"
                "RESTRICAO_COFF_EOLICA_{year}_{month:02d}.csv")
URL_CONS_UFV = (f"{ONS_BASE}/restricao_coff_fotovoltaica_tm/"
                "RESTRICAO_COFF_FOTOVOLTAICA_{year}_{month:02d}.csv")

CCEE_PLD_URLS: dict[int, str] = {
    2021: "https://pda-download.ccee.org.br/SMpDR_R7SCOOj6pMbk1BJg/content",
    2022: "https://pda-download.ccee.org.br/0YTnGY1jRb-tarnKnSNT9g/content",
    2023: "https://pda-download.ccee.org.br/HH4Xegm7R56M_H4qPNOvaw/content",
    2024: "https://pda-download.ccee.org.br/rMsBwN6TT-WUW2_LbGUvkw/content",
    2025: "https://pda-download.ccee.org.br/korJMXwpSLGyVlpRMQWduA/content",
    2026: "https://pda-download.ccee.org.br/6A5wq97KTCWv_bvs3CqsQQ/content",
}

RAZAO_LABEL = {
    "REL": "Indisponibilidade externa",
    "CNF": "Confiabilidade",
    "ENE": "Energetico",
    "PAR": "Parecer de acesso",
}
RAZAO_RESSARCIVEL = {"REL": True, "CNF": True, "ENE": False, "PAR": False}


# =============================================================================
#  UTILS
# =============================================================================

def _normalize(text: Any) -> str:
    if pd.isna(text): return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).upper().strip()


def _fmt_br(v: float, casas: int = 0) -> str:
    s = f"{v:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _ensure_dir(p: str | Path) -> Path:
    path = Path(p); path.mkdir(parents=True, exist_ok=True); return path


def _months_between(start: date, end: date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m)); m += 1
        if m == 13: m, y = 1, y + 1
    return out


def _is_recent_month(year: int, month: int, today: date, n: int) -> bool:
    """True se (year, month) esta nos N meses mais recentes incluindo o atual."""
    delta = (today.year - year) * 12 + (today.month - month)
    return 0 <= delta < n


_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}


def _ccee_session() -> requests.Session:
    """Cria uma sessao com cookies obtidos visitando a pagina do dataset CCEE,
    headers de navegador e Referer. Resolve bloqueio 403 que acontece quando
    se faz request direto na URL pda-download.ccee.org.br."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,*/*;q=0.8"),
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    })
    # Warm-up: visita a pagina do dataset pra coletar cookies
    try:
        s.get("https://dadosabertos.ccee.org.br/dataset/pld_horario",
                timeout=30, allow_redirects=True)
    except requests.RequestException as e:
        print(f"  [!] Warm-up CCEE falhou: {e} - tentando sem cookies")
    return s


def _download_ccee(session: requests.Session, url: str, dest: Path,
                    timeout: int, retries: int, force: bool = False) -> bool:
    """Download via sessao CCEE com Referer ajustado pro dataset."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return True
    if force and dest.exists():
        dest.unlink()
    headers = {"Referer": "https://dadosabertos.ccee.org.br/dataset/pld_horario"}
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=timeout, stream=True, headers=headers,
                              allow_redirects=True)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            total = int(r.headers.get("content-length", 0))
            with tmp.open("wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"  {dest.name}", leave=False, ncols=80,
            ) as bar:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk); bar.update(len(chunk))
            tmp.replace(dest); return True
        except requests.RequestException as e:
            if attempt == retries:
                print(f"  [!] Falha download CCEE {url}: {e}")
                return False
    return False


def _download(url: str, dest: Path, timeout: int, retries: int,
               force: bool = False) -> bool:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return True
    if force and dest.exists():
        dest.unlink()
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, stream=True,
                              headers=_BROWSER_HEADERS)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            total = int(r.headers.get("content-length", 0))
            with tmp.open("wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"  {dest.name}", leave=False, ncols=80,
            ) as bar:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk); bar.update(len(chunk))
            tmp.replace(dest); return True
        except requests.RequestException as e:
            if attempt == retries:
                print(f"  [!] Falha download {url}: {e}"); return False
    return False


def _read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep=";", decimal=",", encoding=enc,
                               na_values=["", "NULL", "null", "NaN", "-"],
                               dtype={"ceg": str, "id_ons": str},
                               low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Nao consegui ler {path}")


def _filter_relevant_usinas(df: pd.DataFrame,
                              patterns: list[str] | None) -> pd.DataFrame:
    """Mantem apenas linhas onde nom_usina OU nom_conjuntousina contem
    algum dos patterns. Reduz drasticamente uso de memoria descartando
    usinas irrelevantes logo na entrada."""
    if df.empty or not patterns:
        return df
    cols_check = [c for c in ("nom_usina", "nom_conjuntousina") if c in df.columns]
    if not cols_check:
        return df
    mask = pd.Series(False, index=df.index)
    for col in cols_check:
        norm = df[col].astype(str).map(_normalize)
        for pat in patterns:
            mask |= norm.str.contains(pat, na=False, regex=False)
    return df[mask].copy()


# =============================================================================
#  DOWNLOAD + LOAD
# =============================================================================

def _carrega_mes(url_tpl: str, source: str, kind: str,
                  year: int, month: int, cache_dir: Path,
                  cfg: dict, today: date) -> pd.DataFrame | None:
    sub = _ensure_dir(cache_dir / "ons" / kind / source)
    url = url_tpl.format(year=year, month=month)
    dest = sub / f"{kind}_{source}_{year}_{month:02d}.csv"
    force = _is_recent_month(year, month, today, cfg["refresh_recent_n"])
    if not _download(url, dest, cfg["request_timeout"], cfg["max_retries"],
                      force=force):
        return None
    try:
        df = _read_csv_robust(dest)
    except Exception as e:
        print(f"  [!] Erro lendo {dest}: {e}"); return None
    df["fonte"] = "EOL" if source == "eolica" else "UFV"
    df["din_instante"] = pd.to_datetime(df["din_instante"], errors="coerce")
    return df.dropna(subset=["din_instante"])


def carregar_detalhe(cfg: dict, dt_ini: date, dt_fim: date,
                       patterns: list[str] | None = None) -> pd.DataFrame:
    cache = _ensure_dir(cfg["cache_dir"])
    today = date.today()
    meses = _months_between(dt_ini, dt_fim)
    print(f"\n[1/4] Detalhe ONS por usina ({len(meses)} meses x 2 fontes)...")
    print(f"      Re-baixando os {cfg['refresh_recent_n']} meses mais recentes.")
    if patterns:
        print(f"      Filtrando precocemente {len(patterns)} padroes de usina "
              "(Mauriti + benchmark) para reduzir memoria.")
    frames = []
    for source, url in (("eolica", URL_DETAIL_EOL), ("solar", URL_DETAIL_UFV)):
        for y, m in tqdm(meses, desc=f"  detail {source:8s}", ncols=80):
            df = _carrega_mes(url, source, "detail", y, m, cache, cfg, today)
            if df is None or df.empty:
                continue
            if patterns:
                df = _filter_relevant_usinas(df, patterns)
            if not df.empty:
                frames.append(df)
    if not frames:
        raise RuntimeError("Nenhum dado de detalhe carregado.")
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["din_instante"] >= pd.Timestamp(dt_ini)) &
            (df["din_instante"] <= pd.Timestamp(dt_fim) + pd.Timedelta(days=1))]
    for col in ("val_geracaoestimada", "val_geracaoverificada",
                "val_ventoverificado"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
    df["nom_usina_norm"] = df["nom_usina"].apply(_normalize)
    print(f"  -> {len(df):,} linhas, {df['nom_usina'].nunique()} usinas distintas")
    return df


def carregar_solar_ne_agregado(cfg: dict, dt_ini: date, dt_fim: date,
                                  pld: pd.DataFrame) -> pd.DataFrame:
    """Carrega o universo de UFVs do submercado NE e agrega por hora.
    Retorna DF compacto com colunas: hora, mwh_total_ne, n_usinas.
    Diferente de carregar_detalhe: NAO filtra por patterns, mas agrega
    imediatamente cada mes (mantendo memoria baixa)."""
    cache = _ensure_dir(cfg["cache_dir"])
    today = date.today()
    meses = _months_between(dt_ini, dt_fim)
    print(f"\n[*] Universo solar NE - agregando por hora ({len(meses)} meses)...")
    rows = []
    for y, m in tqdm(meses, desc="  solar NE   ", ncols=80):
        df = _carrega_mes(URL_DETAIL_UFV, "solar", "detail", y, m,
                            cache, cfg, today)
        if df is None or df.empty:
            continue
        # Filtra submercado NE (id_subsistema)
        if "id_subsistema" in df.columns:
            df = df[df["id_subsistema"].astype(str).str.upper().str.strip() == "NE"]
        if df.empty:
            continue
        # Calcula geracao em MWh e agrega por hora cheia
        df["geracao_mwh"] = (pd.to_numeric(df["val_geracaoverificada"],
                                              errors="coerce")
                                .clip(lower=0) * 0.5)
        df["hora"] = df["din_instante"].dt.floor("h")
        agg = df.groupby("hora").agg(
            mwh_total_ne=("geracao_mwh", "sum"),
            n_usinas=("nom_usina", "nunique"),
        ).reset_index()
        rows.append(agg)
        del df  # libera memoria
    if not rows:
        return pd.DataFrame(columns=["hora", "mwh_total_ne", "n_usinas"])
    out = (pd.concat(rows, ignore_index=True)
              .groupby("hora")
              .agg(mwh_total_ne=("mwh_total_ne", "sum"),
                   n_usinas=("n_usinas", "max"))
              .reset_index()
              .sort_values("hora"))
    print(f"  -> {len(out):,} horas, ate {int(out['n_usinas'].max() or 0)} "
          "UFVs distintas no NE")
    return out


def carregar_consolidado(cfg: dict, dt_ini: date, dt_fim: date,
                           patterns: list[str] | None = None) -> pd.DataFrame:
    cache = _ensure_dir(cfg["cache_dir"])
    today = date.today()
    meses = _months_between(dt_ini, dt_fim)
    print(f"\n[2/4] Consolidado ONS com razoes do corte...")
    frames = []
    for source, url in (("eolica", URL_CONS_EOL), ("solar", URL_CONS_UFV)):
        for y, m in tqdm(meses, desc=f"  cons   {source:8s}", ncols=80):
            df = _carrega_mes(url, source, "cons", y, m, cache, cfg, today)
            if df is None or df.empty:
                continue
            if patterns:
                df = _filter_relevant_usinas(df, patterns)
            if not df.empty:
                frames.append(df)
    if not frames:
        print("  [!] Sem consolidado - razoes ficarao 'DESCONHECIDA'.")
        return pd.DataFrame(columns=["din_instante", "ceg",
                                       "cod_razaorestricao",
                                       "cod_origemrestricao"])
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["din_instante"] >= pd.Timestamp(dt_ini)) &
            (df["din_instante"] <= pd.Timestamp(dt_fim) + pd.Timedelta(days=1))]
    n_total = len(df)
    # IMPORTANTE: descartar linhas onde cod_razaorestricao eh nulo.
    # No dataset ONS, ha 1 linha por instante por usina mesmo SEM restricao -
    # nessas linhas cod_razaorestricao vem nulo. Manter essas linhas confunde
    # o merge depois. So queremos rows com restricao real.
    df = df.dropna(subset=["cod_razaorestricao"])
    n_real = len(df)
    df["cod_razaorestricao"] = df["cod_razaorestricao"].astype(str).str.strip().str.upper()
    df["cod_origemrestricao"] = (df["cod_origemrestricao"].fillna("DESCONHECIDA")
                                   .astype(str).str.strip().str.upper())
    if "ceg" in df.columns:
        df["ceg"] = df["ceg"].astype(str).str.strip().str.upper()
    print(f"  -> {n_real:,} restricoes reais (de {n_total:,} linhas totais)")
    return df


def carregar_pld(cfg: dict, dt_ini: date, dt_fim: date,
                  submercado: str) -> pd.DataFrame:
    cache = _ensure_dir(Path(cfg["cache_dir"]) / "ccee")
    today = date.today()
    anos = sorted({d.year for d in pd.date_range(dt_ini, dt_fim, freq="D")})
    print(f"\n[3/4] PLD horario CCEE ({anos})...")

    frames = []
    anos_pendentes = list(anos)

    # ========== ETAPA 1: lê PLD manual do repo (se disponivel) ==========
    manual_dir = Path("./pld_data")
    if manual_dir.exists():
        print(f"       Procurando arquivos manuais em {manual_dir}/...")
        for ano in list(anos_pendentes):
            manual_file = manual_dir / f"pld_horario_{ano}.csv"
            if manual_file.exists() and manual_file.stat().st_size > 0:
                try:
                    df_man = _read_csv_robust(manual_file)
                    frames.append(df_man)
                    anos_pendentes.remove(ano)
                    print(f"  [OK] PLD {ano} carregado MANUAL de {manual_file}"
                            f" ({manual_file.stat().st_size/1024:.0f} KB)")
                except Exception as e:
                    print(f"  [!] Erro lendo manual {manual_file}: {e}")

    # ========== ETAPA 2: download da CCEE (anos faltantes) ==========
    if anos_pendentes:
        print(f"       Tentando CCEE download para {anos_pendentes}...")
        print("       Iniciando sessao com warm-up no portal CCEE...")
        session = _ccee_session()
        for ano in anos_pendentes:
            url = CCEE_PLD_URLS.get(ano)
            if not url:
                continue
            dest = cache / f"pld_horario_{ano}.csv"
            force = (ano == today.year)
            if not _download_ccee(session, url, dest,
                                    cfg["request_timeout"], cfg["max_retries"],
                                    force=force):
                print(f"  [!] Nao baixou PLD {ano} da CCEE "
                      f"(provavel bloqueio IP)")
                continue
            try:
                frames.append(_read_csv_robust(dest))
                print(f"  [OK] PLD {ano} baixado da CCEE")
            except Exception as e:
                print(f"  [!] Erro lendo PLD {ano}: {e}")

    if not frames:
        print("  [!] CRITICO: nenhum dado de PLD disponivel.")
        print("  [!] >>> SOLUCAO: baixe manualmente os CSVs da CCEE em")
        print("  [!]     https://dadosabertos.ccee.org.br/dataset/pld_horario")
        print("  [!]     e coloque em pld_data/pld_horario_YYYY.csv no repo")
        print("  [!] Modulacao sera mostrada com aviso de indisponibilidade.")
        rng = pd.date_range(dt_ini, dt_fim + pd.Timedelta(days=1), freq="h")
        df_fb = pd.DataFrame({"hora": rng, "pld": 200.0})
        df_fb.attrs["fallback"] = True
        return df_fb

    pld = pd.concat(frames, ignore_index=True)
    pld.columns = [c.strip().lower() for c in pld.columns]

    # ===== DETECCAO DE FORMATO =====
    # Formato A (CCEE atual, manual download): MES_REFERENCIA;SUBMERCADO;
    #   PERIODO_COMERCIALIZACAO;DIA;HORA;PLD_HORA
    #   ex: 202512;NORDESTE;721;31;0;205.25
    # Formato B (CCEE antigo, dados abertos API): din_inicio_periodo,
    #   cd_submercado, val_pld
    formato_a = ("mes_referencia" in pld.columns and "dia" in pld.columns
                  and "hora" in pld.columns and "pld_hora" in pld.columns)

    if formato_a:
        print(f"  Formato A detectado: MES_REFERENCIA + DIA + HORA")
        pld["mes_referencia"] = pld["mes_referencia"].astype(str).str.strip()
        ano = pld["mes_referencia"].str[:4]
        mes = pld["mes_referencia"].str[4:6]
        dia = pd.to_numeric(pld["dia"], errors="coerce") \
                  .astype("Int64").astype(str).str.zfill(2)
        hr = pd.to_numeric(pld["hora"], errors="coerce") \
                .astype("Int64").astype(str).str.zfill(2)
        pld["hora_dt"] = pd.to_datetime(
            ano + "-" + mes + "-" + dia + " " + hr + ":00:00",
            errors="coerce")
        # Mapeia nomes longos para siglas: NORDESTE->NE, NORTE->N, SUDESTE->SE, SUL->S
        sub_map = {"NORDESTE": "NE", "NORTE": "N",
                    "SUDESTE": "SE", "SUL": "S"}
        pld["sub"] = (pld["submercado"].astype(str).str.upper().str.strip()
                         .map(sub_map)
                         .fillna(pld["submercado"]))
        pld["pld"] = pld["pld_hora"]
        pld["hora"] = pld["hora_dt"]
        pld = pld.drop(columns=["hora_dt"])
    else:
        print(f"  Formato B detectado: din_inicio + val_pld")
        col_data = next((c for c in pld.columns if "din_inicio" in c
                         or "din_referencia" in c or "din_instante" in c), None)
        col_sub = next((c for c in pld.columns if "submercado" in c), None)
        col_pld = next((c for c in pld.columns
                        if c == "val_pld" or ("pld" in c and "val" in c)), None)
        if col_pld is None:
            col_pld = next((c for c in pld.columns
                            if "preco" in c or "valor" in c), None)
        pld = pld.rename(columns={col_data: "hora", col_sub: "sub",
                                    col_pld: "pld"})
        pld["hora"] = pd.to_datetime(pld["hora"], errors="coerce")

    pld["pld"] = pd.to_numeric(pld["pld"].astype(str).str.replace(",", "."),
                                errors="coerce")
    pld = pld.dropna(subset=["hora", "pld"])
    pld = pld[pld["sub"].astype(str).str.upper().str.contains(submercado.upper())]
    pld = pld[(pld["hora"] >= pd.Timestamp(dt_ini)) &
              (pld["hora"] <= pd.Timestamp(dt_fim) + pd.Timedelta(days=1))]
    pld = pld[["hora", "pld"]].drop_duplicates("hora").sort_values("hora")
    pld.attrs["fallback"] = False
    print(f"  -> {len(pld):,} horas em {submercado}, "
          f"PLD min={pld['pld'].min():.0f} max={pld['pld'].max():.0f} "
          f"R$/MWh")
    return pld


# =============================================================================
#  ENRIQUECIMENTO + SELECAO + METRICAS
# =============================================================================

def enriquecer(detalhe: pd.DataFrame, cons: pd.DataFrame,
                pld: pd.DataFrame) -> pd.DataFrame:
    df = detalhe.copy()
    df["curtailment_mw"] = (df["val_geracaoestimada"] -
                            df["val_geracaoverificada"]).clip(lower=0)
    df["curtailment_mwh"] = df["curtailment_mw"] * 0.5
    df["geracao_mwh"] = df["val_geracaoverificada"].clip(lower=0) * 0.5
    df["estimada_mwh"] = df["val_geracaoestimada"].clip(lower=0) * 0.5

    # Garante nom_usina_norm presente
    if "nom_usina_norm" not in df.columns and "nom_usina" in df.columns:
        df["nom_usina_norm"] = df["nom_usina"].apply(_normalize)
    # Normaliza ceg / id_ons (defesa contra whitespace)
    if "ceg" in df.columns:
        df["ceg"] = df["ceg"].astype(str).str.strip().str.upper()
    if "id_ons" in df.columns:
        df["id_ons"] = df["id_ons"].astype(str).str.strip().str.upper()

    n_cortes = int((df["curtailment_mw"] > 0.01).sum())
    print(f"  Linhas em detail com curtailment > 0: {n_cortes:,}")

    # Garante nom_conjuntousina_norm no detail tambem
    if ("nom_conjuntousina" in df.columns
            and "nom_conjuntousina_norm" not in df.columns):
        df["nom_conjuntousina_norm"] = (df["nom_conjuntousina"]
                                          .astype(str).map(_normalize))

    if not cons.empty and "cod_razaorestricao" in cons.columns:
        # Prepara consolidado: normaliza identificadores e dedupe priorizando razao
        c2 = cons.copy()
        if "nom_usina_norm" not in c2.columns and "nom_usina" in c2.columns:
            c2["nom_usina_norm"] = c2["nom_usina"].apply(_normalize)
        # cons tambem tem nom_conjuntousina as vezes - normaliza tambem
        if ("nom_conjuntousina" in c2.columns
                and "nom_conjuntousina_norm" not in c2.columns):
            c2["nom_conjuntousina_norm"] = (c2["nom_conjuntousina"]
                                              .astype(str).map(_normalize))
        if "ceg" in c2.columns:
            c2["ceg"] = c2["ceg"].astype(str).str.strip().str.upper()
        if "id_ons" in c2.columns:
            c2["id_ons"] = c2["id_ons"].astype(str).str.strip().str.upper()

        # ===== DIAGNOSTICO: amostras pra ver por que match pode falhar =====
        print("\n  --- DIAGNOSTICO: amostras dos dois datasets ---")
        sd = df[df["curtailment_mw"] > 0.01].head(5)
        cols_d = [c for c in ["din_instante", "nom_usina", "nom_usina_norm",
                                "nom_conjuntousina", "nom_conjuntousina_norm",
                                "ceg", "id_ons"] if c in sd.columns]
        print(f"  DETAIL (5 primeiros cortes):")
        for _, r in sd[cols_d].iterrows():
            ln = f"    t={r['din_instante']} | nom='{r.get('nom_usina','?')}'"
            ln += f" | conj='{r.get('nom_conjuntousina','?')}'"
            ln += f" | conj_norm='{r.get('nom_conjuntousina_norm','?')}'"
            ln += f" | ceg='{r.get('ceg','?')}' | id_ons='{r.get('id_ons','?')}'"
            print(ln)
        # Tenta achar Mauriti no consolidado
        if "nom_usina_norm" in c2.columns:
            sc = c2[c2["nom_usina_norm"].str.contains("MAURITI", na=False)].head(3)
        elif "nom_usina" in c2.columns:
            sc = c2[c2["nom_usina"].astype(str).str.upper().str.contains(
                "MAURITI", na=False)].head(3)
        else:
            sc = c2.head(3)
        cols_c = [c for c in ["din_instante", "nom_usina", "nom_usina_norm",
                                "nom_conjuntousina", "nom_conjuntousina_norm",
                                "ceg", "id_ons", "cod_razaorestricao"]
                  if c in sc.columns]
        print(f"  CONS (3 primeiros Mauriti):")
        for _, r in sc[cols_c].iterrows():
            ln = f"    t={r['din_instante']} | nom='{r.get('nom_usina','?')}'"
            ln += f" | conj='{r.get('nom_conjuntousina','?')}'"
            ln += f" | ceg='{r.get('ceg','?')}' | id_ons='{r.get('id_ons','?')}'"
            ln += f" | razao='{r.get('cod_razaorestricao','?')}'"
            print(ln)
        print(f"  Tipo dtypes detail: din_instante={df['din_instante'].dtype}")
        print(f"  Tipo dtypes cons:   din_instante={c2['din_instante'].dtype}")
        print("  -----------------------------------------------\n")

        prio = {"CNF": 0, "REL": 1, "PAR": 2, "ENE": 3}

        def _try_merge(left, right, keys):
            """Tenta merge; retorna (df_merged, n_match_em_cortes)."""
            r = right.copy()
            r["prio"] = r["cod_razaorestricao"].map(prio).fillna(99)
            cols = keys + ["cod_razaorestricao", "cod_origemrestricao", "prio"]
            cols = [c for c in cols if c in r.columns]
            r = (r[cols].sort_values(keys + ["prio"])
                       .drop_duplicates(keys, keep="first")
                       .drop(columns=["prio"]))
            merged = left.merge(r, on=keys, how="left")
            n_m = int(merged.loc[merged["curtailment_mw"] > 0.01,
                                  "cod_razaorestricao"].notna().sum())
            return merged, n_m

        # Tentativa 1: nom_usina_norm
        if "nom_usina_norm" in df.columns and "nom_usina_norm" in c2.columns:
            merged, n_m = _try_merge(df, c2, ["din_instante", "nom_usina_norm"])
            rate = 100 * n_m / max(n_cortes, 1)
            print(f"  Match via nom_usina: {n_m:,}/{n_cortes:,} cortes "
                  f"({rate:.1f}%)")
            best_merged, best_rate = merged, rate
            if rate < 50:
                print("  [!] Match baixo via nom_usina - tentando ceg...")
                merged, n_m = _try_merge(df, c2, ["din_instante", "ceg"])
                rate = 100 * n_m / max(n_cortes, 1)
                print(f"  Match via ceg: {n_m:,}/{n_cortes:,} cortes "
                      f"({rate:.1f}%)")
                if rate > best_rate:
                    best_merged, best_rate = merged, rate
            if best_rate < 50 and "id_ons" in df.columns and "id_ons" in c2.columns:
                print("  [!] Match baixo via ceg - tentando id_ons...")
                merged, n_m = _try_merge(df, c2, ["din_instante", "id_ons"])
                rate = 100 * n_m / max(n_cortes, 1)
                print(f"  Match via id_ons: {n_m:,}/{n_cortes:,} cortes "
                      f"({rate:.1f}%)")
                if rate > best_rate:
                    best_merged, best_rate = merged, rate
            # Tentativa 4 (CRITICA): cons reporta razoes por CONJUNTO
            # (ex: "CONJ. MAURITI"), enquanto detail tem UFVs individuais
            # (Mauriti 1, Mauriti 2...). Casa nom_conjuntousina_norm
            # do detail com nom_usina_norm do cons.
            if (best_rate < 50 and "nom_conjuntousina_norm" in df.columns
                    and "nom_usina_norm" in c2.columns):
                print("  [!] Match baixo - tentando conjuntousina (detail) "
                        "x nom_usina (cons)...")
                # Renomeia coluna do detail temporariamente
                df_tmp = df.rename(
                    columns={"nom_conjuntousina_norm": "_join_key"})
                c2_tmp = c2.rename(columns={"nom_usina_norm": "_join_key"})
                merged, n_m = _try_merge(df_tmp, c2_tmp,
                                          ["din_instante", "_join_key"])
                rate = 100 * n_m / max(n_cortes, 1)
                print(f"  Match via conjuntousina: {n_m:,}/{n_cortes:,} "
                        f"cortes ({rate:.1f}%)")
                if rate > best_rate:
                    # Renomeia de volta
                    merged = merged.rename(
                        columns={"_join_key": "nom_conjuntousina_norm"})
                    best_merged, best_rate = merged, rate
            df = best_merged
            print(f"  >>> Melhor match: {best_rate:.1f}%")
        else:
            # Fallback para ceg se nom_usina_norm nao existir
            merged, n_m = _try_merge(df, c2, ["din_instante", "ceg"])
            print(f"  Match via ceg: {n_m:,}/{n_cortes:,} cortes")
            df = merged

    df["cod_razaorestricao"] = df.get("cod_razaorestricao",
                                        pd.Series(dtype=str)).fillna("DESCONHECIDA")
    df["cod_origemrestricao"] = df.get("cod_origemrestricao",
                                         pd.Series(dtype=str)).fillna("DESCONHECIDA")
    df["razao_label"] = df["cod_razaorestricao"].map(RAZAO_LABEL).fillna("Sem razao")
    df["ressarcivel"] = df["cod_razaorestricao"].map(RAZAO_RESSARCIVEL).fillna(False)

    df["hora"] = df["din_instante"].dt.floor("h")
    df = df.merge(pld, on="hora", how="left")
    df["pld"] = df["pld"].fillna(df["pld"].median() if not pld.empty else 200.0)
    df["receita_perdida"] = df["curtailment_mwh"] * df["pld"]
    return df


@dataclass
class Selecao:
    label: str
    df: pd.DataFrame
    nomes: list[str] = field(default_factory=list)


@dataclass
class Grupo:
    label: str
    fonte: str
    df: pd.DataFrame
    nomes: list[str]


def selecionar_mauriti(df: pd.DataFrame, match: str) -> Selecao:
    pat = _normalize(match)
    sub = df[df["nom_usina_norm"].str.contains(pat, na=False)].copy()
    nomes = sorted(sub["nom_usina"].dropna().unique().tolist())
    return Selecao(label=f"Complexo Mauriti ({len(nomes)} UFVs)",
                    df=sub, nomes=nomes)


def selecionar_grupos(df: pd.DataFrame,
                       groups_cfg: list[dict]) -> list[Grupo]:
    out: list[Grupo] = []
    # Pre-computa nom_conjuntousina_norm uma vez
    if ("nom_conjuntousina" in df.columns
            and "nom_conjuntousina_norm" not in df.columns):
        df = df.copy()
        df["nom_conjuntousina_norm"] = (df["nom_conjuntousina"]
                                            .astype(str).map(_normalize))
    for g in groups_cfg:
        pat = _normalize(g["match"])
        # Casa em nom_usina OU nom_conjuntousina (resolve clusters Tipo II-C)
        mask = df["nom_usina_norm"].str.contains(pat, na=False)
        if "nom_conjuntousina_norm" in df.columns:
            mask = mask | df["nom_conjuntousina_norm"].str.contains(pat,
                                                                       na=False)
        sub = df[mask].copy()
        nomes = sorted(sub["nom_usina"].dropna().unique().tolist())
        if not nomes:
            print(f"    ! {g['label']:20s}: NENHUMA usina encontrada "
                  f"(match='{pat}')")
            continue
        print(f"    + {g['label']:20s}: {len(nomes)} usina(s) "
              f"-> {', '.join(nomes[:3])}{' ...' if len(nomes)>3 else ''}")
        out.append(Grupo(g["label"], g["fonte"], sub, nomes))
    return out


def metricas(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"vazio": True, "n_usinas": 0}
    total_curt = float(df["curtailment_mwh"].sum())
    total_estim = float(df["estimada_mwh"].sum())
    receita = float(df["receita_perdida"].sum())
    eventos = df[df["curtailment_mw"] > 0.01]
    pld_corte = (float((eventos["pld"] * eventos["curtailment_mwh"]).sum() /
                        max(eventos["curtailment_mwh"].sum(), 1e-9))
                 if not eventos.empty else 0.0)
    rec_ress = float(df.loc[df["ressarcivel"], "receita_perdida"].sum())
    return dict(
        vazio=False,
        total_curt_mwh=total_curt,
        total_estimada_mwh=total_estim,
        curtailment_factor=(100*total_curt/total_estim) if total_estim>0 else 0,
        receita_perdida=receita,
        receita_ressarcivel=rec_ress,
        pct_ressarcivel=(100*rec_ress/receita) if receita>0 else 0,
        n_eventos_30min=int(len(eventos)),
        pld_durante_corte=pld_corte,
        pld_geral=float(df["pld"].mean()),
        n_usinas=int(df["nom_usina"].nunique()),
    )


# =============================================================================
#  ANALISE DE MODULACAO (efeito perfil / valor de perfil)
# =============================================================================

def agregar_horario_mauriti(df_mauriti: pd.DataFrame) -> pd.DataFrame:
    """Agrega Mauriti (todas as UFVs do complexo) em geracao horaria total."""
    if df_mauriti.empty:
        return pd.DataFrame(columns=["hora", "mwh"])
    df = df_mauriti.copy()
    df["hora"] = df["din_instante"].dt.floor("h")
    out = df.groupby("hora")["geracao_mwh"].sum().reset_index()
    out.columns = ["hora", "mwh"]
    return out


def calcular_modulacao(geracao_horaria: pd.DataFrame,
                        pld: pd.DataFrame) -> pd.DataFrame:
    """Cruza geracao horaria com PLD horario e calcula:
       receita_real (Sum hora h: MWh_h * PLD_h)
       receita_flat (Sum dia: MWh_dia * PLD_medio_dia)
       desconto_modulacao_rs = receita_real - receita_flat
       desconto_modulacao_pct = desconto_rs / receita_flat
       preco_efetivo = receita_real / MWh
       Retorna DF diario."""
    if geracao_horaria.empty or pld.empty:
        return pd.DataFrame()
    df = geracao_horaria.merge(pld, on="hora", how="left")
    df = df.dropna(subset=["pld"])
    if df.empty:
        return pd.DataFrame()
    df["receita_real"] = df["mwh"] * df["pld"]
    df["dia"] = df["hora"].dt.date
    diario = df.groupby("dia").agg(
        mwh_dia=("mwh", "sum"),
        receita_real=("receita_real", "sum"),
        pld_avg=("pld", "mean"),
    ).reset_index()
    diario["receita_flat"] = diario["mwh_dia"] * diario["pld_avg"]
    diario["desconto_rs"] = diario["receita_real"] - diario["receita_flat"]
    diario["desconto_pct"] = (100 * diario["desconto_rs"] /
                                diario["receita_flat"].replace(0, np.nan))
    diario["preco_efetivo"] = (diario["receita_real"] /
                                 diario["mwh_dia"].replace(0, np.nan))
    return diario


def metricas_modulacao(diario: pd.DataFrame) -> dict:
    """Resume um DF de modulacao diaria em KPIs de periodo."""
    if diario.empty:
        return {"vazio": True}
    rec_real = float(diario["receita_real"].sum())
    rec_flat = float(diario["receita_flat"].sum())
    desc_rs = rec_real - rec_flat
    desc_pct = (100 * desc_rs / rec_flat) if rec_flat > 0 else 0.0
    mwh_total = float(diario["mwh_dia"].sum())
    preco_ef = (rec_real / mwh_total) if mwh_total > 0 else 0.0
    pld_avg = float(diario["pld_avg"].mean())
    return dict(
        vazio=False,
        receita_real=rec_real,
        receita_flat=rec_flat,
        desconto_rs=desc_rs,
        desconto_pct=desc_pct,
        mwh_total=mwh_total,
        preco_efetivo=preco_ef,
        pld_medio=pld_avg,
        n_dias=int(len(diario)),
    )


def perfil_horario(geracao_horaria: pd.DataFrame,
                     pld: pd.DataFrame) -> pd.DataFrame:
    """Calcula o perfil horario tipico (media de cada hora 0-23):
       Mauriti gen avg by hour vs PLD avg by hour."""
    if geracao_horaria.empty or pld.empty:
        return pd.DataFrame()
    df = geracao_horaria.merge(pld, on="hora", how="left").dropna(subset=["pld"])
    df["hora_dia"] = df["hora"].dt.hour
    out = df.groupby("hora_dia").agg(
        gen_avg=("mwh", "mean"),
        pld_avg=("pld", "mean"),
    ).reset_index()
    return out


# =============================================================================
#  PALETA EDITORIAL + LAYOUT PLOTLY
# =============================================================================

EL = {
    "bg":"#fafaf6", "bg_alt":"#f3f1ea", "panel":"#ffffff",
    "border":"#e6e1d4", "border2":"#d8d2c2",
    "ink":"#1a1715", "ink_2":"#3d3833", "muted":"#857d72",
    "rule":"#c8c0ad",
    "accent":"#a8442f", "accent_2":"#7a4528",
    "accent_light":"#d6997a", "accent_today":"#d92e0f",
    "neutral":"#5a5147", "ok":"#2d5a3d",
}

LAY = dict(
    paper_bgcolor=EL["panel"], plot_bgcolor=EL["panel"],
    font=dict(family="'IBM Plex Sans',Georgia,serif", color=EL["ink"], size=12),
    margin=dict(l=60, r=30, t=60, b=50),
    legend=dict(orientation="h", y=-0.20,
                font=dict(family="'IBM Plex Mono',monospace", size=11)),
    xaxis=dict(gridcolor=EL["border"], zerolinecolor=EL["border2"],
                linecolor=EL["border2"], tickcolor=EL["border2"]),
    yaxis=dict(gridcolor=EL["border"], zerolinecolor=EL["border2"],
                linecolor=EL["border2"], tickcolor=EL["border2"]),
    hoverlabel=dict(font_family="'IBM Plex Mono',monospace",
                     bgcolor=EL["panel"], bordercolor=EL["accent"]),
)

def ed_title(text: str, size: int = 20):
    return dict(text=text, x=0, xanchor="left",
                font=dict(family="'Fraunces',Georgia,serif", size=size,
                          color=EL["ink"]))


# =============================================================================
#  CHARTS
# =============================================================================

def g_tracker(df_mauriti: pd.DataFrame, today: date | None = None):
    """Tracker do mes corrente: barras estim vs realizada empilhadas + linha CF%."""
    if today is None:
        today = date.today()
    cur_first = today.replace(day=1)
    if cur_first.month == 12:
        next_first = date(cur_first.year + 1, 1, 1)
    else:
        next_first = date(cur_first.year, cur_first.month + 1, 1)
    days_in_month = (next_first - cur_first).days
    all_days = [cur_first + timedelta(days=i) for i in range(days_in_month)]

    df = df_mauriti.copy()
    df["dia"] = df["din_instante"].dt.date
    cur = df[df["dia"] >= cur_first]

    daily = (cur.groupby("dia").agg(estim=("estimada_mwh","sum"),
                                       real=("geracao_mwh","sum"),
                                       curt=("curtailment_mwh","sum"))
                .reindex(all_days, fill_value=0).reset_index()
                .rename(columns={"index":"dia"}))
    daily["cf"] = (100 * daily["curt"] /
                    daily["estim"].replace(0, np.nan)).fillna(0)
    fut = daily["dia"] > today
    daily.loc[fut, ["estim","real","curt","cf"]] = None

    # CF% medio do trimestre fechado anterior
    ref_start = cur_first - timedelta(days=90)
    ref = df[(df["dia"] >= ref_start) & (df["dia"] < cur_first)]
    if not ref.empty:
        ref_d = ref.groupby("dia").agg(e=("estimada_mwh","sum"),
                                          c=("curtailment_mwh","sum"))
        ref_cf = float(100 * ref_d["c"].sum() / max(ref_d["e"].sum(), 1e-9))
    else:
        ref_cf = None

    x = [d.strftime("%d") for d in daily["dia"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=daily["real"], name="Geracao realizada (MWh)",
        marker=dict(color=EL["ink_2"], line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Realizada: %{y:,.1f} MWh<extra></extra>"))
    fig.add_trace(go.Bar(x=x, y=daily["curt"], name="Curtailment (MWh)",
        marker=dict(color=EL["accent"], line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Cortado: %{y:,.1f} MWh<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=daily["cf"], name="CF% do dia",
        mode="lines+markers", yaxis="y2",
        line=dict(color=EL["accent_today"], width=2.5),
        marker=dict(size=8, color=EL["accent_today"],
                     line=dict(color=EL["panel"], width=1.5)),
        hovertemplate="<b>Dia %{x}</b><br>CF: %{y:.2f}%<extra></extra>"))
    if ref_cf is not None:
        fig.add_hline(y=ref_cf, yref="y2",
            line=dict(color=EL["neutral"], width=1.2, dash="dot"),
            annotation_text=f"  CF medio trimestre: {ref_cf:.1f}%",
            annotation_position="top right",
            annotation=dict(font=dict(family="'IBM Plex Mono',monospace",
                                       size=10, color=EL["neutral"])))
    lay = dict(LAY); lay.update(
        title=ed_title(
            f"Mauriti — {cur_first.strftime('%B/%Y').lower()}  ·  "
            "realizada vs cortada por dia", 20),
        barmode="stack",
        xaxis=dict(title=f"dia (1 a {days_in_month})", gridcolor=EL["border"]),
        yaxis=dict(title="MWh / dia", gridcolor=EL["border"]),
        yaxis2=dict(title="CF% do dia", overlaying="y", side="right",
                     showgrid=False, color=EL["accent_today"],
                     ticksuffix="%", rangemode="tozero"),
        hovermode="x unified", height=460, bargap=0.30,
        legend=dict(orientation="h", y=-0.18,
                     font=dict(family="'IBM Plex Mono',monospace", size=11)),
    )
    fig.update_layout(**lay)
    return fig, daily, ref_cf, days_in_month


def g_serie(df, titulo):
    df = df.copy(); df["dia"] = df["din_instante"].dt.date
    d = df.groupby("dia").agg(estim=("estimada_mwh","sum"),
                                 real=("geracao_mwh","sum"),
                                 curt=("curtailment_mwh","sum")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["dia"], y=d["estim"], name="Estimada",
        line=dict(color=EL["ok"], width=1, dash="dot"), mode="lines",
        hovertemplate="<b>%{x}</b><br>Estim.: %{y:,.1f} MWh<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["dia"], y=d["real"], name="Realizada",
        line=dict(color=EL["ink"], width=2),
        fill="tonexty", fillcolor="rgba(168,68,47,0.10)",
        hovertemplate="<b>%{x}</b><br>Real: %{y:,.1f} MWh<extra></extra>"))
    fig.add_trace(go.Bar(x=d["dia"], y=d["curt"], name="Curtailment",
        marker=dict(color=EL["accent"]), opacity=0.85, yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Cortado: %{y:,.1f} MWh<extra></extra>"))
    lay = dict(LAY); lay.update(
        title=ed_title(titulo, 20),
        yaxis=dict(title="MWh", gridcolor=EL["border"]),
        yaxis2=dict(title="Curtailment (MWh)", overlaying="y", side="right",
                     showgrid=False, color=EL["accent"]),
        hovermode="x unified",
    )
    fig.update_layout(**lay); return fig


def g_donut_razao(df, titulo):
    cortes = df[df["curtailment_mw"] > 0.01]
    if cortes.empty:
        return go.Figure(layout=dict(LAY, title=ed_title(titulo)))
    soma = (cortes.groupby("cod_razaorestricao")["curtailment_mwh"]
                  .sum().sort_values(ascending=False))
    cores = {"REL":EL["accent"],"CNF":EL["accent_2"],
             "ENE":EL["neutral"],"PAR":EL["muted"],"DESCONHECIDA":EL["border2"]}
    fig = go.Figure(go.Pie(
        labels=[f"{c}  {RAZAO_LABEL.get(c,c)}" for c in soma.index],
        values=soma.values, hole=0.62,
        marker=dict(colors=[cores.get(c, EL["muted"]) for c in soma.index],
                    line=dict(color=EL["panel"], width=3)),
        textinfo="percent", textfont=dict(family="'IBM Plex Mono'",
                                            size=12, color=EL["panel"]),
        hovertemplate="<b>%{label}</b><br>%{value:,.1f} MWh (%{percent})<extra></extra>",
    ))
    total = float(soma.sum())
    lay = dict(LAY); lay.update(
        title=ed_title(titulo, 20),
        annotations=[dict(
            text=(f"<b style='font-family:Fraunces,Georgia,serif;"
                   f"font-size:32px'>{total/1000:,.1f}</b>"
                   f"<br><span style='color:{EL['muted']};font-size:11px;"
                   "letter-spacing:0.1em'>GWh CORTADOS</span>"),
            x=0.5, y=0.5, showarrow=False, font=dict(color=EL["ink"]))],
        showlegend=True,
    )
    fig.update_layout(**lay); return fig


def g_razoes_mes(df, titulo):
    df = df.copy(); df["mes"] = df["din_instante"].dt.to_period("M").astype(str)
    cortes = df[df["curtailment_mw"] > 0.01]
    if cortes.empty:
        return go.Figure(layout=dict(LAY, title=ed_title(titulo)))
    pivot = (cortes.groupby(["mes","cod_razaorestricao"])["curtailment_mwh"]
                .sum().unstack(fill_value=0))
    cores = {"REL":EL["accent"],"CNF":EL["accent_2"],
             "ENE":EL["neutral"],"PAR":EL["muted"]}
    fig = go.Figure()
    for col in ["REL","CNF","ENE","PAR"]:
        if col not in pivot.columns: continue
        fig.add_trace(go.Bar(
            x=pivot.index, y=pivot[col],
            name=f"{col}  {RAZAO_LABEL.get(col,col)}",
            marker=dict(color=cores.get(col, EL["muted"])),
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:,.1f}} MWh<extra></extra>",
        ))
    lay = dict(LAY); lay.update(
        title=ed_title(titulo, 20), barmode="stack",
        xaxis=dict(title="", gridcolor=EL["border"]),
        yaxis=dict(title="Curtailment (MWh)", gridcolor=EL["border"]),
        hovermode="x unified", height=380,
    )
    fig.update_layout(**lay); return fig


def g_comp_cf(mdf, grupos: list[Grupo]):
    """CF% mensal: Mauriti destacada + 1 linha por grupo do benchmark."""
    def mensal_cf(d):
        if d.empty: return None
        d = d.copy(); d["mes"] = d["din_instante"].dt.to_period("M").astype(str)
        m = d.groupby("mes").agg(c=("curtailment_mwh","sum"),
                                   e=("estimada_mwh","sum")).reset_index()
        m["cf"] = (100 * m["c"] / m["e"].replace(0, np.nan)).fillna(0)
        return m

    fig = go.Figure()
    eol_palette = ["#5b6b7d", "#7a8794", "#465261", "#8a9aa8"]
    ufv_palette = ["#9a8467", "#c4a987", "#7d6a52", "#b39477", "#5e4e3a", "#a48972"]
    eol_i, ufv_i = 0, 0

    for grp in grupos:
        m = mensal_cf(grp.df)
        if m is None or m.empty: continue
        if grp.fonte == "EOL":
            cor = eol_palette[eol_i % len(eol_palette)]; eol_i += 1
        else:
            cor = ufv_palette[ufv_i % len(ufv_palette)]; ufv_i += 1
        fig.add_trace(go.Scatter(
            x=m["mes"], y=m["cf"], name=f"{grp.fonte}  {grp.label}",
            mode="lines+markers",
            line=dict(color=cor, width=1.6),
            marker=dict(size=6, color=cor, line=dict(color=EL["panel"], width=1)),
            hovertemplate=f"<b>{grp.label}</b> ({grp.fonte})<br>"
                          "%{x}: %{y:.2f}%<extra></extra>",
            opacity=0.92,
        ))

    m = mensal_cf(mdf)
    if m is not None:
        fig.add_trace(go.Scatter(
            x=m["mes"], y=m["cf"], name="<b>MAURITI</b>",
            mode="lines+markers",
            line=dict(color=EL["accent"], width=3.5),
            marker=dict(size=11, color=EL["accent"],
                        line=dict(color=EL["panel"], width=2)),
            hovertemplate="<b>MAURITI</b><br>%{x}: %{y:.2f}%<extra></extra>",
        ))

    lay = dict(LAY); lay.update(
        title=ed_title("Curtailment factor mensal por ativo "
                        "— Mauriti vs benchmark fixo CE", 20),
        yaxis=dict(title="Curtailment factor (%)",
                    gridcolor=EL["border"], ticksuffix="%"),
        xaxis=dict(title="", gridcolor=EL["border"]),
        hovermode="x unified", height=460,
        legend=dict(orientation="v", x=1.02, y=1, yanchor="top", xanchor="left",
                     font=dict(family="'IBM Plex Mono',monospace", size=10),
                     bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=200, t=60, b=50),
    )
    fig.update_layout(**lay); return fig


# ----- Charts da aba MODULACAO -----

def g_mod_tracker(diario_m: pd.DataFrame, ref_pct_ne: float | None,
                    today: date | None = None):
    """Tracker do mes corrente: receita_real vs receita_flat por dia +
    linha de % desconto, com referencia do NE."""
    if today is None:
        today = date.today()
    cur_first = today.replace(day=1)
    if cur_first.month == 12:
        next_first = date(cur_first.year + 1, 1, 1)
    else:
        next_first = date(cur_first.year, cur_first.month + 1, 1)
    days_in_month = (next_first - cur_first).days
    all_days = [cur_first + timedelta(days=i) for i in range(days_in_month)]

    cur = diario_m[(diario_m["dia"] >= cur_first) & (diario_m["dia"] < next_first)].copy()
    cur = (cur.set_index("dia").reindex(all_days).reset_index()
              .rename(columns={"index": "dia"}))
    fut = cur["dia"] > today
    cur.loc[fut, ["receita_real", "receita_flat", "desconto_pct"]] = None

    x = [d.strftime("%d") for d in cur["dia"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=cur["receita_flat"], name="Receita flat (R$)",
        marker=dict(color=EL["muted"], opacity=0.65, line=dict(width=0)),
        hovertemplate="<b>Dia %{x}</b><br>Flat: R$ %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(x=x, y=cur["receita_real"], name="Receita real (R$)",
        marker=dict(color=EL["ink_2"], line=dict(width=0)),
        hovertemplate="<b>Dia %{x}</b><br>Real: R$ %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=cur["desconto_pct"], name="% desconto",
        mode="lines+markers", yaxis="y2",
        line=dict(color=EL["accent_today"], width=2.5),
        marker=dict(size=8, color=EL["accent_today"],
                     line=dict(color=EL["panel"], width=1.5)),
        hovertemplate="<b>Dia %{x}</b><br>Desc.: %{y:.2f}%<extra></extra>"))
    if ref_pct_ne is not None:
        fig.add_hline(y=ref_pct_ne, yref="y2",
            line=dict(color=EL["neutral"], width=1.2, dash="dot"),
            annotation_text=f"  benchmark NE: {ref_pct_ne:.1f}%",
            annotation_position="top right",
            annotation=dict(font=dict(family="'IBM Plex Mono',monospace",
                                       size=10, color=EL["neutral"])))
    lay = dict(LAY); lay.update(
        title=ed_title(f"Mauriti — {cur_first.strftime('%B/%Y').lower()}  ·  "
                        "receita real vs receita flat", 20),
        barmode="group",
        xaxis=dict(title=f"dia (1 a {days_in_month})", gridcolor=EL["border"]),
        yaxis=dict(title="R$ / dia", gridcolor=EL["border"]),
        yaxis2=dict(title="% desconto modulacao", overlaying="y", side="right",
                     showgrid=False, color=EL["accent_today"], ticksuffix="%"),
        hovermode="x unified", height=460, bargap=0.25, bargroupgap=0.05,
    )
    fig.update_layout(**lay); return fig


def g_mod_perfil_horario(perfil_m: pd.DataFrame):
    """Perfil horario tipico: PLD R$/MWh + Geracao Mauriti MW. Mostra
    o vale do PLD coincidindo com pico solar."""
    if perfil_m.empty:
        return go.Figure(layout=dict(LAY,
            title=ed_title("Perfil horario tipico", 20)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perfil_m["hora_dia"], y=perfil_m["pld_avg"],
        name="PLD medio (R$/MWh)", mode="lines+markers",
        line=dict(color="#5b6b7d", width=2.5),
        marker=dict(size=7, color="#5b6b7d"),
        hovertemplate="<b>%{x}h</b><br>PLD: R$ %{y:,.1f}/MWh<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=perfil_m["hora_dia"], y=perfil_m["gen_avg"],
        name="Geracao Mauriti (MWh/h)", mode="lines+markers",
        line=dict(color=EL["accent"], width=2.5), yaxis="y2",
        marker=dict(size=7, color=EL["accent"]),
        fill="tozeroy", fillcolor="rgba(168,68,47,0.10)",
        hovertemplate="<b>%{x}h</b><br>Gen: %{y:,.1f} MWh<extra></extra>",
    ))
    lay = dict(LAY); lay.update(
        title=ed_title("Por que perdemos receita: perfil horario tipico", 20),
        xaxis=dict(title="Hora do dia (0-23)", dtick=2,
                    gridcolor=EL["border"]),
        yaxis=dict(title="PLD medio (R$/MWh)", gridcolor=EL["border"],
                    color="#5b6b7d"),
        yaxis2=dict(title="Geracao media Mauriti (MWh/h)", overlaying="y",
                     side="right", showgrid=False, color=EL["accent"]),
        hovermode="x unified", height=420,
        legend=dict(orientation="h", y=-0.18,
                     font=dict(family="'IBM Plex Mono',monospace", size=11)),
    )
    fig.update_layout(**lay); return fig


def g_mod_historico_mensal(diario_m: pd.DataFrame, diario_ne: pd.DataFrame):
    """Historico mensal % desconto: Mauriti vs benchmark NE."""
    def por_mes(d):
        if d.empty: return pd.DataFrame()
        d = d.copy(); d["mes"] = pd.to_datetime(d["dia"]).dt.to_period("M").astype(str)
        m = d.groupby("mes").agg(
            rr=("receita_real", "sum"),
            rf=("receita_flat", "sum"),
        ).reset_index()
        m["pct"] = (100 * (m["rr"] - m["rf"]) / m["rf"].replace(0, np.nan))
        return m

    m_m = por_mes(diario_m)
    m_n = por_mes(diario_ne)
    fig = go.Figure()
    if not m_m.empty:
        fig.add_trace(go.Bar(
            x=m_m["mes"], y=m_m["pct"], name="Mauriti",
            marker=dict(color=EL["accent"]),
            hovertemplate="<b>Mauriti %{x}</b><br>Desc.: %{y:.2f}%<extra></extra>",
        ))
    if not m_n.empty:
        fig.add_trace(go.Bar(
            x=m_n["mes"], y=m_n["pct"], name="Frota NE solar",
            marker=dict(color=EL["neutral"]),
            hovertemplate="<b>NE %{x}</b><br>Desc.: %{y:.2f}%<extra></extra>",
        ))
    lay = dict(LAY); lay.update(
        title=ed_title("% desconto modulacao por mes — "
                        "Mauriti vs benchmark NE", 20),
        barmode="group",
        xaxis=dict(title="", gridcolor=EL["border"]),
        yaxis=dict(title="% desconto modulacao", gridcolor=EL["border"],
                    ticksuffix="%"),
        hovermode="x unified", height=400,
    )
    fig.update_layout(**lay); return fig


def g_mod_top_dias(diario_m: pd.DataFrame, n: int = 10):
    """Top N dias com maior desconto absoluto em R$."""
    if diario_m.empty:
        return go.Figure(layout=dict(LAY,
            title=ed_title("Top dias", 20)))
    top = (diario_m[diario_m["desconto_rs"] < 0]
              .nsmallest(n, "desconto_rs")
              .sort_values("desconto_rs"))  # mais negativo no topo
    if top.empty:
        return go.Figure(layout=dict(LAY,
            title=ed_title("Top dias - sem dias negativos", 20)))
    labels = [d.strftime("%d/%m/%Y") for d in top["dia"]]
    fig = go.Figure(go.Bar(
        y=labels, x=top["desconto_rs"].abs(), orientation="h",
        marker=dict(color=EL["accent"]),
        text=[f"R$ {v:,.0f}   ·   {p:.2f}%   ·   PLD medio R$ {pld:,.0f}"
              for v, p, pld in zip(top["desconto_rs"].abs(),
                                    top["desconto_pct"],
                                    top["pld_avg"])],
        textposition="outside",
        textfont=dict(color=EL["ink_2"], size=10,
                      family="'IBM Plex Mono',monospace"),
        hovertemplate="<b>%{y}</b><br>Perdido: R$ %{x:,.0f}<extra></extra>",
    ))
    lay = dict(LAY); lay.update(
        title=ed_title(f"Top {n} dias com maior desconto absoluto", 20),
        xaxis=dict(title="R$ perdidos no dia", gridcolor=EL["border"]),
        yaxis=dict(autorange="reversed", gridcolor=EL["border"]),
        height=max(280, 40 * len(top) + 100),
        margin=dict(l=110, r=280, t=60, b=50),
        showlegend=False,
    )
    fig.update_layout(**lay); return fig


def g_heatmap_horario(df_mauriti):
    df = df_mauriti.copy()
    df["dia"] = df["din_instante"].dt.date
    df["hora"] = df["din_instante"].dt.hour
    df = df[(df["hora"] >= 5) & (df["hora"] <= 19)]
    pivot_curt = (df.groupby(["hora","dia"])["curtailment_mwh"]
                     .sum().unstack(fill_value=0))
    pivot_estim = (df.groupby(["hora","dia"])["estimada_mwh"]
                      .sum().unstack(fill_value=0))
    z = (100 * pivot_curt / pivot_estim.replace(0, np.nan)).fillna(0)
    fig = go.Figure(data=go.Heatmap(
        z=z.values, x=[str(d) for d in z.columns], y=z.index,
        colorscale=[[0.0, EL["bg"]], [0.05, "#f1e8d8"], [0.20, "#e6c7a8"],
                     [0.45, EL["accent_light"]], [0.75, EL["accent"]],
                     [1.0, "#5e1d10"]],
        colorbar=dict(title=dict(text="CF%", font=dict(size=11)),
                       outlinewidth=0, ticksuffix="%",
                       tickfont=dict(family="'IBM Plex Mono'", size=10)),
        zmax=min(80, float(z.values.max()) if z.size > 0 else 80), zmin=0,
        hovertemplate="Dia %{x}  Hora %{y}h<br>CF: %{z:.1f}%<extra></extra>",
    ))
    lay = dict(LAY); lay.update(
        title=ed_title("Heatmap hora × dia — onde os cortes acontecem", 20),
        yaxis=dict(title="Hora do dia", dtick=2, autorange="reversed",
                    gridcolor=EL["border"]),
        xaxis=dict(title="Dia (apuracao do periodo)",
                    gridcolor=EL["border"], showticklabels=False),
        height=420, margin=dict(l=70, r=30, t=60, b=50),
    )
    fig.update_layout(**lay); return fig


# =============================================================================
#  INSIGHTS automaticos
# =============================================================================

def _gera_insights_mauriti(df: pd.DataFrame, met: dict) -> str:
    if met.get("vazio"):
        return ""
    cortes = df[df["curtailment_mw"] > 0.01]
    if cortes.empty:
        return "Nao houve eventos relevantes de corte no periodo."
    razao_grp = cortes.groupby("cod_razaorestricao")["curtailment_mwh"].sum()
    if razao_grp.empty or razao_grp.sum() <= 0:
        return "Nao foi possivel atribuir razao aos cortes do periodo."
    razao_top = razao_grp.idxmax()
    pct_top = (razao_grp[razao_top] / razao_grp.sum() * 100)
    origem_grp = cortes.groupby("cod_origemrestricao")["curtailment_mwh"].sum()
    origem_top = origem_grp.idxmax() if not origem_grp.empty else "—"
    horas = cortes.groupby(cortes["din_instante"].dt.hour)["curtailment_mwh"].sum()
    hora_pico = int(horas.idxmax()) if not horas.empty else 12
    return (
        f"A razao dominante do corte foi <strong>{razao_top} - "
        f"{RAZAO_LABEL.get(razao_top, razao_top)}</strong>, respondendo por "
        f"<strong>{pct_top:.1f}%</strong> do volume cortado. A origem predominante "
        f"foi <strong>{origem_top}</strong>. Os cortes se concentram em torno "
        f"das <strong>{hora_pico}h</strong> — padrao tipico de UFV em horario "
        f"de pico de geracao solar coincidente com baixa demanda do SIN. "
        f"R$ {met['receita_ressarcivel']/1e6:.2f} M ({met['pct_ressarcivel']:.1f}%) "
        f"sao potencialmente ressarciveis via REN 1.030/2022 (REL+CNF)."
    )


def _gera_insights_comp(mauriti_met: dict, bench_met: dict) -> str:
    if mauriti_met.get("vazio") or bench_met.get("vazio"):
        return ""
    delta = mauriti_met["curtailment_factor"] - bench_met["curtailment_factor"]
    if abs(delta) < 0.5:
        return (f"O CF de Mauriti ({mauriti_met['curtailment_factor']:.2f}%) "
                f"esta <strong>em linha</strong> com a media do benchmark "
                f"({bench_met['curtailment_factor']:.2f}%), sugerindo que os "
                f"cortes refletem dinamica sistemica do submercado NE mais do "
                f"que problema especifico do ativo.")
    if delta > 0:
        return (f"Mauriti apresenta CF <strong>{delta:.2f} pp acima</strong> da "
                f"media do benchmark ({mauriti_met['curtailment_factor']:.2f}% "
                f"vs {bench_met['curtailment_factor']:.2f}%), indicando "
                f"exposicao acima da media — vale aprofundar nos motivos "
                f"locais (parecer de acesso, capacidade da SE Mauriti / "
                f"LT Bom Nome - Milagres).")
    return (f"Mauriti apresenta CF <strong>{abs(delta):.2f} pp abaixo</strong> "
            f"da media do benchmark ({mauriti_met['curtailment_factor']:.2f}% "
            f"vs {bench_met['curtailment_factor']:.2f}%) — performance "
            f"superior em relacao aos pares no periodo.")


# =============================================================================
#  HTML TEMPLATE - editorial light
# =============================================================================

HTML_TEMPLATE = r"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Mauriti — Estudo de Curtailment & Modulação</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root{
  --bg:#fafaf6; --bg-alt:#f3f1ea; --panel:#ffffff;
  --border:#e6e1d4; --border2:#d8d2c2;
  --ink:#1a1715; --ink-2:#3d3833; --muted:#857d72;
  --rule:#c8c0ad; --accent:#a8442f; --accent-2:#7a4528;
  --accent-today:#d92e0f; --neutral:#5a5147; --ok:#2d5a3d;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
  font-family:'IBM Plex Sans',Georgia,serif;
  -webkit-font-smoothing:antialiased; line-height:1.5}
.wrap{max-width:1100px;margin:0 auto;padding:80px 32px 96px}
.masthead{display:flex;align-items:center;justify-content:space-between;
  border-bottom:1px solid var(--ink);padding-bottom:16px;margin-bottom:32px;
  font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--ink-2);letter-spacing:0.18em;text-transform:uppercase}
.masthead .vol{font-weight:600}

/* ====== TABS ====== */
.tabs{display:flex;gap:0;border-bottom:1px solid var(--rule);
  margin-bottom:48px;}
.tab{background:none;border:none;cursor:pointer;
  font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:500;
  color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;
  padding:14px 24px;margin-right:6px;
  border-bottom:2px solid transparent;
  transition:color 0.15s, border-color 0.15s}
.tab:hover{color:var(--ink-2)}
.tab.active{color:var(--accent);border-bottom-color:var(--accent);
  font-weight:600}

.hero{margin-bottom:60px}
.hero .kicker{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--accent);letter-spacing:0.25em;text-transform:uppercase;
  margin-bottom:18px}
.hero h1{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:clamp(40px,7vw,72px);line-height:0.98;
  letter-spacing:-0.02em;margin:0 0 24px;font-variation-settings:"opsz" 144}
.hero h1 em{font-style:italic;font-weight:400;color:var(--accent)}
.hero .lede{font-family:'Fraunces',Georgia,serif;font-weight:400;font-size:20px;
  line-height:1.45;color:var(--ink-2);max-width:720px;
  font-variation-settings:"opsz" 36}
.hero .lede strong{color:var(--ink);font-weight:500}
.hero .byline{margin-top:32px;font-family:'IBM Plex Mono',monospace;
  font-size:11px;color:var(--muted);letter-spacing:0.1em;
  text-transform:uppercase;line-height:1.8}
.hero .byline span{color:var(--ink)}

.tracker{margin:48px 0 64px;padding:32px 36px;background:var(--bg-alt);
  border:1px solid var(--rule);border-radius:2px;position:relative}
.tracker .liveflag{position:absolute;top:-12px;left:32px;
  background:var(--accent-today);color:#fff;
  font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;
  letter-spacing:0.2em;padding:5px 10px;text-transform:uppercase}
.tracker h2{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:28px;line-height:1.1;letter-spacing:-0.01em;margin:0 0 6px}
.tracker .when{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--muted);letter-spacing:0.12em;text-transform:uppercase;
  margin-bottom:24px}
.tracker-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:24px;margin:20px 0 28px;padding:20px 0;
  border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.t-stat .lbl{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);text-transform:uppercase;letter-spacing:0.15em;
  margin-bottom:8px}
.t-stat .val{font-family:'Fraunces',Georgia,serif;font-weight:400;
  font-size:30px;line-height:1;letter-spacing:-0.01em}
.t-stat .val .unit{font-family:'IBM Plex Sans',sans-serif;font-size:13px;
  color:var(--muted);font-weight:400;margin-left:3px}
.t-stat.alt .val{color:var(--accent)}
.t-stat .delta{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--muted);margin-top:6px;letter-spacing:0.04em}
.t-stat .delta.up{color:var(--accent-today);font-weight:500}
.t-stat .delta.down{color:var(--ok);font-weight:500}
.bignum{display:grid;grid-template-columns:1fr 1.4fr;gap:64px;
  align-items:end;margin:64px 0 48px;padding-top:40px;
  border-top:3px double var(--rule)}
.bignum .figure{font-family:'Fraunces',Georgia,serif;font-weight:300;
  font-size:clamp(80px,16vw,160px);line-height:0.9;letter-spacing:-0.04em;
  color:var(--accent);font-variation-settings:"opsz" 144}
.bignum .figure span{font-size:0.32em;color:var(--ink);font-weight:500;
  margin-left:14px;letter-spacing:0;display:inline-block}
.bignum .copy h2{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:28px;line-height:1.15;letter-spacing:-0.01em;margin:0 0 16px}
.bignum .copy p{font-size:15px;line-height:1.65;color:var(--ink-2);
  margin:0 0 8px;max-width:520px}
@media (max-width:780px){.bignum{grid-template-columns:1fr}}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:36px 40px;margin:48px 0;padding:32px 0;
  border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.stat .lbl{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);text-transform:uppercase;letter-spacing:0.18em;
  margin-bottom:10px}
.stat .val{font-family:'Fraunces',Georgia,serif;font-weight:400;
  font-size:34px;line-height:1;letter-spacing:-0.02em;color:var(--ink);
  font-variation-settings:"opsz" 72}
.stat .val .unit{font-family:'IBM Plex Sans',sans-serif;font-size:13px;
  color:var(--muted);font-weight:400;margin-left:4px}
.stat .delta{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--muted);margin-top:8px;letter-spacing:0.05em}
.stat.alt .val{color:var(--accent)}
.section-head{margin:64px 0 28px;padding-bottom:14px;
  border-bottom:1px solid var(--ink);
  display:flex;align-items:baseline;gap:18px}
.section-head .num{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--muted);letter-spacing:0.2em}
.section-head h3{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:26px;line-height:1.1;letter-spacing:-0.01em;margin:0;flex:1;
  font-variation-settings:"opsz" 36}
.section-head .tag{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);text-transform:uppercase;letter-spacing:0.15em}
.section-desc{font-family:'Fraunces',Georgia,serif;font-size:17px;
  line-height:1.55;color:var(--ink-2);max-width:680px;margin:0 0 28px;
  font-variation-settings:"opsz" 28}
.pullquote{margin:40px 0;padding:28px 36px;background:var(--bg-alt);
  border-left:3px solid var(--accent);
  font-family:'Fraunces',Georgia,serif;font-size:19px;line-height:1.5;
  color:var(--ink);font-variation-settings:"opsz" 36}
.pullquote strong{color:var(--accent);font-weight:500;font-style:italic}
.pullquote cite{display:block;margin-top:14px;font-family:'IBM Plex Mono',monospace;
  font-size:10px;color:var(--muted);font-style:normal;
  letter-spacing:0.15em;text-transform:uppercase}
.disclaimer{margin:32px 0;padding:20px 24px;background:transparent;
  border:1px dashed var(--rule);border-radius:2px;
  font-family:'IBM Plex Sans',sans-serif;font-size:13px;line-height:1.6;
  color:var(--muted)}
.disclaimer strong{color:var(--ink-2);font-weight:500}
.chart{background:var(--panel);border:1px solid var(--border);
  border-radius:2px;margin:24px 0;padding:8px 4px 4px}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:24px 0}
@media (max-width:880px){.chart-row{grid-template-columns:1fr}}
footer{margin-top:96px;padding-top:32px;border-top:1px solid var(--ink);
  font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);
  line-height:1.8;letter-spacing:0.04em}
footer p{margin:0 0 10px;max-width:680px}
footer .colofao{font-family:'Fraunces',Georgia,serif;font-style:italic;
  font-size:13px;color:var(--ink-2);margin-top:24px}

.tab-pane{display:none}
.tab-pane.active{display:block}
</style>
</head>
<body>

<div class="wrap">

  <div class="masthead">
    <div class="vol">Mauriti Report — N&deg; 01</div>
    <div>{{ periodo }} &nbsp;&middot;&nbsp; atualizado {{ gerado_em }}</div>
  </div>

  <!-- TABS -->
  <div class="tabs">
    <button class="tab active" data-tab="curt">Curtailment</button>
    <button class="tab" data-tab="mod">Efeito Modula&ccedil;&atilde;o</button>
  </div>

  <!-- ============================================================ -->
  <!-- TAB: CURTAILMENT                                              -->
  <!-- ============================================================ -->
  <div class="tab-pane active" data-tab="curt">

    <div class="hero">
      <div class="kicker">PowerChina &middot; Complexo fotovoltaico Mauriti, Cear&aacute;</div>
      <h1>Quanto custou ao Mauriti<br>
          cada megawatt-hora <em>cortado</em>.</h1>
      <p class="lede">
        Estudo de constrained-off do <strong>Complexo Fotovoltaico Mauriti</strong>
        no per&iacute;odo de {{ periodo }}, com benchmark contra usinas e&oacute;licas
        e fotovoltaicas do Cear&aacute;, quebra por raz&atilde;o do corte
        (REL/CNF/ENE/PAR) e estimativa de receita perdida a PLD hor&aacute;rio.
      </p>
      <div class="byline">
        Submercado <span>{{ submercado }}</span> &nbsp;&middot;&nbsp;
        UFVs Mauriti <span>{{ met_m.n_usinas }}</span> &nbsp;&middot;&nbsp;
        Benchmark <span>{{ n_grupos }} grupos</span> &nbsp;&middot;&nbsp;
        Apurado em <span>{{ gerado_em }}</span>
      </div>
    </div>

    <div class="tracker">
      <div class="liveflag">&bull; Atualizado semanalmente</div>
      <h2>Acompanhamento — m&ecirc;s corrente</h2>
      <div class="when">Per&iacute;odo: {{ tracker.cur_first }} &middot;
        {{ tracker.dias_decorridos }} de {{ tracker.days_in_month }} dias decorridos</div>

      <div class="tracker-stats">
        <div class="t-stat">
          <div class="lbl">Esperada (m&ecirc;s)</div>
          <div class="val">{{ "%.1f"|format(tracker.esperada_mes) }}<span class="unit">MWh</span></div>
          <div class="delta">{{ tracker.dias_decorridos }} dias decorridos</div>
        </div>
        <div class="t-stat">
          <div class="lbl">Cortada (m&ecirc;s)</div>
          <div class="val">{{ "%.1f"|format(tracker.cortada_mes) }}<span class="unit">MWh</span></div>
          <div class="delta">de {{ "%.1f"|format(tracker.esperada_mes) }} esperados</div>
        </div>
        <div class="t-stat alt">
          <div class="lbl">CF% do m&ecirc;s</div>
          <div class="val">{{ "%.2f"|format(tracker.cf_mes) }}<span class="unit">%</span></div>
          {% if tracker.delta_cf is not none %}
          <div class="delta {% if tracker.delta_cf > 0 %}up{% else %}down{% endif %}">
            {% if tracker.delta_cf > 0 %}+{% endif %}{{ "%.2f"|format(tracker.delta_cf) }} pp vs trim.
          </div>
          {% endif %}
        </div>
        <div class="t-stat">
          <div class="lbl">CF medio trim.</div>
          <div class="val">{% if tracker.ref_cf is not none %}{{ "%.2f"|format(tracker.ref_cf) }}{% else %}—{% endif %}<span class="unit">%</span></div>
          <div class="delta">refer&ecirc;ncia 90 dias</div>
        </div>
        <div class="t-stat">
          <div class="lbl">Pior dia (CF%)</div>
          <div class="val">{{ "%.1f"|format(tracker.pior_cf) }}<span class="unit">%</span></div>
          <div class="delta">dia {{ tracker.pior_dia }}</div>
        </div>
      </div>

      <div style="background:transparent;padding:0">
        <div id="tracker" style="height:460px"></div>
      </div>
    </div>

    <div class="bignum">
      <div class="figure">
        {{ "%.1f"|format(met_m.total_curt_mwh/1000) }}<span>GWh</span>
      </div>
      <div class="copy">
        <h2>Energia n&atilde;o entregue por restri&ccedil;&atilde;o de opera&ccedil;&atilde;o</h2>
        <p>Equivale a <strong>R$ {{ "%.1f"|format(met_m.receita_perdida/1e6) }}
          milh&otilde;es</strong> em receita perdida estimada a PLD hor&aacute;rio
          do submercado {{ submercado }}, com curtailment factor de
          <strong>{{ "%.2f"|format(met_m.curtailment_factor) }}%</strong>.</p>
        <p>Desse total, <strong>{{ "%.1f"|format(met_m.pct_ressarcivel) }}%
          potencialmente ressarciveis</strong> sob a REN ANEEL 1.030/2022
          (raz&otilde;es REL e CNF).</p>
      </div>
    </div>

    <div class="stats">
      <div class="stat alt">
        <div class="lbl">Curtailment Factor</div>
        <div class="val">{{ "%.2f"|format(met_m.curtailment_factor) }}<span class="unit">%</span></div>
        <div class="delta">cortado / refer&ecirc;ncia</div>
      </div>
      <div class="stat">
        <div class="lbl">Receita perdida</div>
        <div class="val">R$ {{ "%.1f"|format(met_m.receita_perdida/1e6) }}<span class="unit">M</span></div>
        <div class="delta">a PLD hor&aacute;rio {{ submercado }}</div>
      </div>
      <div class="stat">
        <div class="lbl">Pot. ressarciv&eacute;l</div>
        <div class="val">R$ {{ "%.1f"|format(met_m.receita_ressarcivel/1e6) }}<span class="unit">M</span></div>
        <div class="delta">REL + CNF</div>
      </div>
      <div class="stat">
        <div class="lbl">PLD durante corte</div>
        <div class="val">{{ "%.0f"|format(met_m.pld_durante_corte) }}<span class="unit">R$/MWh</span></div>
        <div class="delta">vs m&eacute;dio R$ {{ "%.0f"|format(met_m.pld_geral) }}/MWh</div>
      </div>
      <div class="stat">
        <div class="lbl">vs Benchmark CE</div>
        {% set delta = met_m.curtailment_factor - met_b.curtailment_factor %}
        <div class="val">{% if delta > 0 %}+{% endif %}{{ "%.2f"|format(delta) }}<span class="unit">pp</span></div>
        <div class="delta">delta de CF</div>
      </div>
    </div>

    <div class="section-head">
      <span class="num">I.</span><h3>O ritmo dos cortes</h3>
      <span class="tag">SERIE TEMPORAL</span>
    </div>
    <p class="section-desc">
      A &aacute;rea entre a refer&ecirc;ncia (linha pontilhada verde) e a gera&ccedil;&atilde;o
      realizada representa a energia que poderia ter sido produzida e n&atilde;o foi.
    </p>
    <div class="chart"><div id="serie" style="height:440px"></div></div>

    <div class="section-head">
      <span class="num">II.</span><h3>Por que se corta</h3>
      <span class="tag">RAZ&Otilde;ES DA RESTRI&Ccedil;&Atilde;O</span>
    </div>
    <p class="section-desc">
      Indisponibilidade externa (REL) e Confiabilidade (CNF) s&atilde;o tipicamente
      trat&aacute;veis sob a REN 1.030/2022; raz&atilde;o energ&eacute;tica (ENE) raramente o &eacute;.
    </p>
    <div class="chart-row">
      <div class="chart"><div id="donut_razao" style="height:380px"></div></div>
      <div class="chart"><div id="razoes_mes" style="height:380px"></div></div>
    </div>

    {% if insights_m %}
    <div class="pullquote">{{ insights_m|safe }}<cite>Leitura autom&aacute;tica</cite></div>
    {% endif %}

    <div class="section-head">
      <span class="num">III.</span><h3>Mauriti vs ativos do CE</h3>
      <span class="tag">BENCHMARK FIXO</span>
    </div>
    <p class="section-desc">
      Compara&ccedil;&atilde;o direta com {{ n_grupos }} grupos: <strong>{{ grupos_str }}</strong>.
    </p>
    <div class="chart"><div id="comp_cf" style="height:480px"></div></div>

    {% if insights_c %}
    <div class="pullquote">{{ insights_c|safe }}<cite>An&aacute;lise comparativa</cite></div>
    {% endif %}

    <div class="section-head">
      <span class="num">IV.</span><h3>Onde, no dia, se corta</h3>
      <span class="tag">PADR&Atilde;O HOR&Aacute;RIO</span>
    </div>
    <p class="section-desc">
      Heatmap do CF% por hora do dia. Cortes em 11h&ndash;14h indicam restri&ccedil;&atilde;o
      sist&ecirc;mica do NE; pulverizados sugerem limita&ccedil;&atilde;o local.
    </p>
    <div class="chart"><div id="heatmap" style="height:440px"></div></div>

  </div><!-- /tab curt -->


  <!-- ============================================================ -->
  <!-- TAB: MODULACAO                                                -->
  <!-- ============================================================ -->
  <div class="tab-pane" data-tab="mod">

    {% if pld_fallback %}
    <div style="background:#fff7e0;border:2px solid #d4a017;
                padding:24px 28px;margin:0 0 40px;border-radius:2px">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                  color:#a07a00;letter-spacing:0.18em;text-transform:uppercase;
                  font-weight:600;margin-bottom:10px">
        &#9888; PLD Indispon&iacute;vel — Modula&ccedil;&atilde;o n&atilde;o calculada
      </div>
      <div style="font-family:'Fraunces',Georgia,serif;font-size:17px;
                  line-height:1.5;color:#3d3833;margin-bottom:14px">
        N&atilde;o foi poss&iacute;vel obter o PLD hor&aacute;rio da CCEE nesta execu&ccedil;&atilde;o
        (a CCEE bloqueia requests de IPs de cloud providers como o GitHub Actions).
        Os gr&aacute;ficos abaixo est&atilde;o usando R$ 200/MWh como placeholder, o que
        <strong>zera artificialmente o desconto de modula&ccedil;&atilde;o</strong>.
      </div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;
                  line-height:1.6;color:#5a5147;
                  border-top:1px dashed #d4a017;padding-top:12px;margin-top:8px">
        <strong style="color:#3d3833">Como resolver:</strong> baixe os CSVs de
        <code style="background:#fdf3d0;padding:2px 6px;border-radius:2px">
        pld_horario_2025.csv</code> e
        <code style="background:#fdf3d0;padding:2px 6px;border-radius:2px">
        pld_horario_2026.csv</code> em
        <a href="https://dadosabertos.ccee.org.br/dataset/pld_horario"
           style="color:#a8442f">dadosabertos.ccee.org.br</a>,
        crie a pasta <code style="background:#fdf3d0;padding:2px 6px">pld_data/</code>
        no repo e commite os arquivos l&aacute;. O script vai us&aacute;-los na pr&oacute;xima execu&ccedil;&atilde;o.
      </div>
    </div>
    {% endif %}

    <div class="hero">
      <div class="kicker">PowerChina &middot; Mauriti &middot; Valor de perfil</div>
      <h1>Quanto custou <em>quando</em><br>geramos cada MWh.</h1>
      <p class="lede">
        Compara&ccedil;&atilde;o entre a receita real do Mauriti
        (Σ MWh<sub>hora</sub> × PLD<sub>hora</sub>) e a receita "flat"
        que se obteria se a gera&ccedil;&atilde;o fosse plana ao longo do dia.
        A diferen&ccedil;a &eacute; o <strong>desconto de modula&ccedil;&atilde;o</strong> &mdash;
        custo do timing, n&atilde;o do volume.
      </p>
      <div class="byline">
        Submercado <span>{{ submercado }}</span> &nbsp;&middot;&nbsp;
        Benchmark <span>frota UFV NE</span> &nbsp;&middot;&nbsp;
        Per&iacute;odo <span>{{ periodo }}</span>
      </div>
    </div>

    {% if not met_mod_m.vazio %}
    <div class="tracker">
      <div class="liveflag">&bull; Atualizado semanalmente</div>
      <h2>Acompanhamento — m&ecirc;s corrente</h2>
      <div class="when">Receita real vs receita flat por dia</div>

      <div class="tracker-stats">
        <div class="t-stat alt">
          <div class="lbl">Desconto m&ecirc;s</div>
          {% if mod_tracker.cur_pct is not none %}
          <div class="val">{{ "%.2f"|format(mod_tracker.cur_pct) }}<span class="unit">%</span></div>
          {% else %}<div class="val">—</div>{% endif %}
          <div class="delta">no m&ecirc;s corrente</div>
        </div>
        <div class="t-stat">
          <div class="lbl">Desconto m&ecirc;s (R$)</div>
          {% if mod_tracker.cur_rs is not none %}
          <div class="val">{{ "%.0f"|format(mod_tracker.cur_rs/1000) }}<span class="unit">k R$</span></div>
          {% else %}<div class="val">—</div>{% endif %}
          <div class="delta">receita potencial perdida</div>
        </div>
        <div class="t-stat">
          <div class="lbl">PLD m&eacute;dio m&ecirc;s</div>
          {% if mod_tracker.cur_pld is not none %}
          <div class="val">{{ "%.0f"|format(mod_tracker.cur_pld) }}<span class="unit">R$/MWh</span></div>
          {% else %}<div class="val">—</div>{% endif %}
          <div class="delta">submercado {{ submercado }}</div>
        </div>
        <div class="t-stat">
          <div class="lbl">Benchmark NE</div>
          {% if met_mod_ne.vazio %}<div class="val">—</div>
          {% else %}
          <div class="val">{{ "%.2f"|format(met_mod_ne.desconto_pct) }}<span class="unit">%</span></div>
          {% endif %}
          <div class="delta">frota UFV NE no per&iacute;odo</div>
        </div>
        <div class="t-stat">
          <div class="lbl">vs benchmark</div>
          {% if mod_tracker.delta_pp is not none %}
          <div class="delta {% if mod_tracker.delta_pp < 0 %}up{% else %}down{% endif %}"
                style="font-size:30px;font-family:Fraunces,Georgia,serif;
                       letter-spacing:-0.01em;margin-top:0">
            {% if mod_tracker.delta_pp > 0 %}+{% endif %}{{ "%.2f"|format(mod_tracker.delta_pp) }} pp
          </div>
          <div class="delta">{{ "Mauriti pior" if mod_tracker.delta_pp < 0 else "Mauriti melhor" }}</div>
          {% else %}<div class="val">—</div>{% endif %}
        </div>
      </div>

      <div style="background:transparent;padding:0">
        <div id="mod_tracker" style="height:460px"></div>
      </div>
    </div>

    <div class="bignum">
      <div class="figure">
        {{ "%.1f"|format((met_mod_m.desconto_rs|abs)/1e6) }}<span>R$ M</span>
      </div>
      <div class="copy">
        <h2>Receita potencial perdida pela modula&ccedil;&atilde;o</h2>
        <p>O Mauriti realizou
          <strong>R$ {{ "%.1f"|format(met_mod_m.receita_real/1e6) }} M</strong>
          quando, com gera&ccedil;&atilde;o &agrave; m&eacute;dia di&aacute;ria do PLD, teria realizado
          <strong>R$ {{ "%.1f"|format(met_mod_m.receita_flat/1e6) }} M</strong>.</p>
        <p>Isso &eacute; um desconto de
          <strong>{{ "%.2f"|format(met_mod_m.desconto_pct) }}%</strong> da
          receita flat &mdash; pre&ccedil;o efetivo de
          <strong>R$ {{ "%.0f"|format(met_mod_m.preco_efetivo) }}/MWh</strong>
          contra um PLD m&eacute;dio de
          <strong>R$ {{ "%.0f"|format(met_mod_m.pld_medio) }}/MWh</strong>.</p>
      </div>
    </div>

    <div class="stats">
      <div class="stat alt">
        <div class="lbl">% Desconto Mauriti</div>
        <div class="val">{{ "%.2f"|format(met_mod_m.desconto_pct) }}<span class="unit">%</span></div>
        <div class="delta">receita_real / receita_flat - 1</div>
      </div>
      <div class="stat">
        <div class="lbl">% Desconto NE (frota)</div>
        {% if met_mod_ne.vazio %}<div class="val">—</div>
        {% else %}<div class="val">{{ "%.2f"|format(met_mod_ne.desconto_pct) }}<span class="unit">%</span></div>{% endif %}
        <div class="delta">benchmark frota UFV NE</div>
      </div>
      <div class="stat">
        <div class="lbl">vs Benchmark NE</div>
        {% if met_mod_ne.vazio %}<div class="val">—</div>
        {% else %}
        {% set d = met_mod_m.desconto_pct - met_mod_ne.desconto_pct %}
        <div class="val">{% if d > 0 %}+{% endif %}{{ "%.2f"|format(d) }}<span class="unit">pp</span></div>
        {% endif %}
        <div class="delta">delta de % desconto</div>
      </div>
      <div class="stat">
        <div class="lbl">Pre&ccedil;o efetivo</div>
        <div class="val">R$ {{ "%.0f"|format(met_mod_m.preco_efetivo) }}<span class="unit">/MWh</span></div>
        <div class="delta">receita_real / MWh</div>
      </div>
      <div class="stat">
        <div class="lbl">PLD m&eacute;dio</div>
        <div class="val">R$ {{ "%.0f"|format(met_mod_m.pld_medio) }}<span class="unit">/MWh</span></div>
        <div class="delta">submercado {{ submercado }}</div>
      </div>
    </div>

    <div class="section-head">
      <span class="num">I.</span><h3>Por que perdemos receita</h3>
      <span class="tag">PERFIL HOR&Aacute;RIO T&Iacute;PICO</span>
    </div>
    <p class="section-desc">
      Dia tipico (m&eacute;dia de cada hora ao longo do per&iacute;odo). O pico de
      gera&ccedil;&atilde;o solar coincide com o vale do PLD do submercado &mdash;
      vendemos a maior parte do MWh quando ele vale menos.
    </p>
    <div class="chart"><div id="mod_perfil" style="height:420px"></div></div>

    <div class="section-head">
      <span class="num">II.</span><h3>Mauriti vs frota NE solar</h3>
      <span class="tag">HIST&Oacute;RICO MENSAL</span>
    </div>
    <p class="section-desc">
      Compara&ccedil;&atilde;o do % de desconto m&ecirc;s a m&ecirc;s. Per&iacute;odos secos
      (PLD geralmente alto o dia todo) tendem a ter desconto menor; per&iacute;odos
      de carga baixa (mais excedente solar no SIN) ampliam o efeito.
    </p>
    <div class="chart"><div id="mod_hist" style="height:400px"></div></div>

    <div class="section-head">
      <span class="num">III.</span><h3>Top dias mais doloridos</h3>
      <span class="tag">RANKING POR R$</span>
    </div>
    <p class="section-desc">
      Os dias em que o desconto absoluto (em R$) foi maior. Geralmente
      coincidem com PLD muito baixo no meio do dia
      (excesso de oferta solar no SIN).
    </p>
    <div class="chart"><div id="mod_top" style="height:520px"></div></div>

    <div class="disclaimer">
      <p><strong>Importante:</strong> esta an&aacute;lise mostra o desconto
      <em>te&oacute;rico</em> assumindo que toda a gera&ccedil;&atilde;o &eacute; liquidada no
      PLD hor&aacute;rio. O impacto real na receita do PowerChina depende do
      regime de comercializa&ccedil;&atilde;o de cada UFV: PPA fixo (R$/MWh) absorve
      o efeito mas perde valor no spot; liquida&ccedil;&atilde;o no MCP captura o
      impacto integral; produtos shape com a CCEE neutralizam o desconto.
      Os n&uacute;meros aqui s&atilde;o <strong>indicador direcional</strong> do
      pr&ecirc;mio que valeria a pena pagar por um hedge perfeito de modula&ccedil;&atilde;o.</p>
    </div>

    {% endif %}

  </div><!-- /tab mod -->


  <footer>
    <p><strong>Fontes</strong> Restri&ccedil;&atilde;o de Opera&ccedil;&atilde;o por Constrained-off
      (ONS, base semi-hor&aacute;ria, detalhamento por usina + consolidado com raz&otilde;es).
      PLD hor&aacute;rio (CCEE).</p>
    <p><strong>Defini&ccedil;&otilde;es</strong> Curtailment = max(0,
      val_geracaoestimada &minus; val_geracaoverificada). Receita perdida =
      curtailment_MWh &times; PLD_horario. CF = curtailment / esperada (%).
      Desconto modula&ccedil;&atilde;o = (Σ MWh<sub>h</sub> × PLD<sub>h</sub>) − (Σ MWh<sub>dia</sub> × PLD<sub>medio_dia</sub>).</p>
    <p><strong>Benchmark NE</strong> O benchmark de modula&ccedil;&atilde;o agrega
      todas as UFVs do submercado NE como uma frota &uacute;nica, ponderando pela
      gera&ccedil;&atilde;o. Mostra o desconto que a frota NE coletivamente sofreu.</p>
    <p class="colofao">Mauriti — Curtailment & Modula&ccedil;&atilde;o, gerado em
      {{ gerado_em }}.</p>
  </footer>

</div>

<script>
const FIGS = {{ figs_json|safe }};

// Renderiza todos os charts no load (mesmo os que estao em tab oculto)
for (const [k, fig] of Object.entries(FIGS)) {
  if (document.getElementById(k)) {
    Plotly.newPlot(k, fig.data, fig.layout, {
      responsive:true, displaylogo:false,
      modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d','zoomIn2d','zoomOut2d']
    });
  }
}

// Tab switching
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab').forEach(b =>
      b.classList.toggle('active', b === btn));
    document.querySelectorAll('.tab-pane').forEach(p =>
      p.classList.toggle('active', p.dataset.tab === target));
    // Resize Plotly charts na aba que ficou visivel
    setTimeout(() => {
      document.querySelectorAll(`.tab-pane.active [id]`).forEach(div => {
        if (window.Plotly && div._fullData) Plotly.Plots.resize(div);
      });
    }, 60);
  });
});
</script>
</body>
</html>
"""


# =============================================================================
#  RENDER
# =============================================================================

def gerar_html(mauriti: Selecao, grupos: list[Grupo], pld: pd.DataFrame,
                ne_horario: pd.DataFrame, pld_sub: str,
                periodo: str, output: Path, today: date) -> None:
    met_m = metricas(mauriti.df)
    bench_df = (pd.concat([g.df for g in grupos], ignore_index=True)
                if grupos else pd.DataFrame())
    met_b = metricas(bench_df) if not bench_df.empty else {"vazio": True,
                                                             "n_usinas": 0,
                                                             "curtailment_factor": 0}

    fig_tracker, daily_t, ref_cf, days_in_month = g_tracker(mauriti.df, today)
    done = daily_t.dropna(subset=["estim"])
    esperada_mes = float(done["estim"].sum())
    cortada_mes = float(done["curt"].sum())
    cf_mes = (100 * cortada_mes / esperada_mes) if esperada_mes > 0 else 0.0
    dias_decorridos = len(done)
    delta_cf = (cf_mes - ref_cf) if ref_cf is not None else None
    if not done.empty and done["cf"].notna().any():
        pior = done.loc[done["cf"].idxmax()]
        pior_dia = pior["dia"].strftime("%d/%m")
        pior_cf = float(pior["cf"])
    else:
        pior_dia, pior_cf = "—", 0.0

    # ========== MODULACAO ==========
    print("\n[*] Calculando modulacao Mauriti + benchmark NE...")
    hor_m = agregar_horario_mauriti(mauriti.df)
    print(f"  Mauriti: {len(hor_m)} horas de geracao agregada, "
          f"total {hor_m['mwh'].sum():.0f} MWh")
    diario_m = calcular_modulacao(hor_m, pld)
    perfil_m = perfil_horario(hor_m, pld)
    met_mod_m = metricas_modulacao(diario_m)
    if not met_mod_m.get("vazio"):
        print(f"  Mauriti receita real: R$ {met_mod_m['receita_real']/1e6:.2f}M")
        print(f"  Mauriti receita flat: R$ {met_mod_m['receita_flat']/1e6:.2f}M")
        print(f"  Mauriti desconto:     R$ {met_mod_m['desconto_rs']/1e6:.2f}M "
              f"({met_mod_m['desconto_pct']:.2f}%)")
        print(f"  Mauriti preco efetivo: R$ {met_mod_m['preco_efetivo']:.0f}/MWh "
              f"vs PLD medio R$ {met_mod_m['pld_medio']:.0f}/MWh")

    if not ne_horario.empty:
        ne_for_calc = ne_horario.rename(columns={"mwh_total_ne": "mwh"})
        diario_ne = calcular_modulacao(ne_for_calc[["hora", "mwh"]], pld)
        met_mod_ne = metricas_modulacao(diario_ne)
        if not met_mod_ne.get("vazio"):
            print(f"  Frota NE desconto:    "
                  f"{met_mod_ne['desconto_pct']:.2f}%")
    else:
        diario_ne = pd.DataFrame()
        met_mod_ne = {"vazio": True}

    # Tracker stats da modulacao (mes corrente)
    cur_first = today.replace(day=1)
    if not diario_m.empty:
        cur_m = diario_m[diario_m["dia"] >= cur_first]
        if not cur_m.empty:
            cur_pct = float((100 * (cur_m["receita_real"].sum()
                                     - cur_m["receita_flat"].sum()) /
                              max(cur_m["receita_flat"].sum(), 1e-9)))
            cur_rs = float(cur_m["receita_real"].sum() -
                            cur_m["receita_flat"].sum())
            cur_pld = float(cur_m["pld_avg"].mean())
        else:
            cur_pct, cur_rs, cur_pld = None, None, None
    else:
        cur_pct, cur_rs, cur_pld = None, None, None
    delta_pp = ((cur_pct - met_mod_ne["desconto_pct"])
                if cur_pct is not None and not met_mod_ne.get("vazio") else None)

    # ========== FIGURAS ==========
    figs: dict[str, Any] = {"tracker": json.loads(pio.to_json(fig_tracker))}
    if not met_m.get("vazio"):
        figs["serie"]       = json.loads(pio.to_json(g_serie(mauriti.df,
            "Geracao diaria — estimada vs realizada")))
        figs["donut_razao"] = json.loads(pio.to_json(g_donut_razao(mauriti.df,
            "Razoes do corte")))
        figs["razoes_mes"]  = json.loads(pio.to_json(g_razoes_mes(mauriti.df,
            "Razoes por mes")))
        figs["heatmap"]     = json.loads(pio.to_json(g_heatmap_horario(mauriti.df)))
    figs["comp_cf"] = json.loads(pio.to_json(g_comp_cf(mauriti.df, grupos)))

    # Charts modulacao
    if not met_mod_m.get("vazio"):
        ref_pct_ne = met_mod_ne.get("desconto_pct") if not met_mod_ne.get("vazio") else None
        figs["mod_tracker"] = json.loads(pio.to_json(
            g_mod_tracker(diario_m, ref_pct_ne, today)))
        figs["mod_perfil"]  = json.loads(pio.to_json(
            g_mod_perfil_horario(perfil_m)))
        figs["mod_hist"]    = json.loads(pio.to_json(
            g_mod_historico_mensal(diario_m, diario_ne)))
        figs["mod_top"]     = json.loads(pio.to_json(
            g_mod_top_dias(diario_m, n=10)))

    grupos_str = ", ".join([g.label for g in grupos]) if grupos else "—"
    pld_fallback = bool(pld.attrs.get("fallback", False))
    html = Template(HTML_TEMPLATE).render(
        met_m=met_m, met_b=met_b,
        n_grupos=len(grupos), grupos_str=grupos_str,
        submercado=pld_sub, periodo=periodo,
        gerado_em=today.strftime("%d/%m/%Y"),
        figs_json=json.dumps(figs),
        insights_m=_gera_insights_mauriti(mauriti.df, met_m),
        insights_c=_gera_insights_comp(met_m, met_b),
        pld_fallback=pld_fallback,
        tracker=dict(
            cur_first=today.replace(day=1).strftime("%B/%Y"),
            days_in_month=days_in_month,
            dias_decorridos=dias_decorridos,
            esperada_mes=esperada_mes,
            cortada_mes=cortada_mes,
            cf_mes=cf_mes,
            ref_cf=ref_cf,
            delta_cf=delta_cf,
            pior_dia=pior_dia,
            pior_cf=pior_cf,
        ),
        met_mod_m=met_mod_m,
        met_mod_ne=met_mod_ne,
        mod_tracker=dict(
            cur_pct=cur_pct, cur_rs=cur_rs, cur_pld=cur_pld,
            delta_pp=delta_pp,
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    print("=" * 78)
    print(" CURTAILMENT MAURITI - dashboard ONS+CCEE  (v3) ")
    print("=" * 78)

    cfg = CONFIG
    dt_ini = datetime.strptime(cfg["data_inicio"], "%Y-%m-%d").date()
    dt_fim = (datetime.strptime(cfg["data_fim"], "%Y-%m-%d").date()
              if cfg["data_fim"] else date.today())

    # Constroi a lista de padroes (Mauriti + benchmark) pra filtragem precoce
    # Isso reduz uso de memoria de 5-10 GB para algumas centenas de MB.
    patterns: list[str] = [_normalize(cfg["mauriti_match"])]
    patterns += [_normalize(g["match"]) for g in cfg["benchmark_groups"]]
    patterns = list(dict.fromkeys(patterns))  # dedupe preservando ordem

    detalhe = carregar_detalhe(cfg, dt_ini, dt_fim, patterns=patterns)
    cons    = carregar_consolidado(cfg, dt_ini, dt_fim, patterns=patterns)
    pld     = carregar_pld(cfg, dt_ini, dt_fim, cfg["submercado"])
    ne_horario = carregar_solar_ne_agregado(cfg, dt_ini, dt_fim, pld)

    print("\n[4/4] Enriquecendo + selecionando grupos...")
    df = enriquecer(detalhe, cons, pld)

    mauriti = selecionar_mauriti(df, cfg["mauriti_match"])
    print(f"\n  Mauriti: {len(mauriti.nomes)} UFVs encontradas")
    for n in mauriti.nomes:
        print(f"    . {n}")

    print(f"\n  Grupos benchmark configurados:")
    grupos = selecionar_grupos(df, cfg["benchmark_groups"])

    out = Path(cfg["output_html"]).resolve()
    periodo = f"{dt_ini.strftime('%d/%m/%Y')} a {dt_fim.strftime('%d/%m/%Y')}"
    gerar_html(mauriti, grupos, pld, ne_horario, cfg["submercado"],
                periodo, out, date.today())
    print(f"\n[OK] Dashboard salvo em: {out}")
    print(f"     Para preview local: abra '{out}' no navegador.")
    print(f"     Para publicar: ./public/ esta pronto pra deploy (GitHub Pages).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido."); sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e.__class__.__name__}: {e}"); raise
