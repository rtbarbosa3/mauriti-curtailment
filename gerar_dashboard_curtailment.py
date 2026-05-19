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

# Tenta importar curl_cffi para bypass de TLS fingerprinting (Akamai).
# Se nao disponivel (ambiente local sem instalar), cai pro requests padrao.
try:
    from curl_cffi import requests as cffi_requests
    _CFFI_OK = True
except ImportError:
    cffi_requests = None
    _CFFI_OK = False

import plotly.graph_objects as go
import plotly.io as pio


# =============================================================================
#  CONFIGURACAO
# =============================================================================

# Versao do dashboard. Atualizar a cada onda de mudancas.
DASH_VERSION = "5.8"
DASH_VERSION_DATE = "2026-05-19"
DASH_CHANGES = [
    "v5.8 (2026-05-19): Onda 1 - PLD freshness badge, modo apresentacao, "
    "tooltips, modo dark/light, atalhos teclado, print/screenshot, footer",
    "v5.7 (2026-05-19): TLS fingerprint Chrome 131 via curl_cffi pra bypass Akamai/CCEE",
    "v5.6 (2026-05-17): Monthly forecast (CCEE View + Commercial + 3 cards)",
    "v5.5 (2026-05-15): Tracker upgrades + Benchmark v2 multi-select",
    "v5.4 (2026-05-13): Cron diario + Solar fix + 2 PPAs",
]

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
    "output_csv_ren1030": "./public/eventos_elegiveis_ren1030.csv",

    # Cache: re-baixar sempre os N meses mais recentes (consistencia recorrente ONS)
    "refresh_recent_n": 3,

    # Coordenadas da UFV Mauriti pra puxar irradiancia NASA POWER
    # https://power.larc.nasa.gov/
    "mauriti_lat": -7.40,
    "mauriti_lon": -38.78,

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

# Origem da restricao (cod_origemrestricao). Dataset ONS usa 2 valores principais:
# LOC = local (proximo da usina); SIS = sistemico (rede regional).
ORIGEM_LABEL = {
    "LOC": "LOC — Local (near plant)",
    "SIS": "SIS — Systemic (regional grid)",
    "DESCONHECIDA": "Unknown",
}


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


def _ccee_session():
    """Cria uma sessao com cookies obtidos visitando a pagina do dataset CCEE.

    ESTRATEGIA DE BYPASS DE AKAMAI/WAF:
    -----------------------------------
    A CCEE usa proteçao Akamai que detecta IPs de cloud (Azure/AWS/GCP) e
    bloqueia requests com TLS fingerprint de Python `requests` (chave JA3).

    Solucao: se curl_cffi estiver disponivel, usa ela com impersonate=chrome131
    que faz o TLS handshake EXATAMENTE como um Chrome real (cipher suites,
    ordem, extensions, HTTP/2 frames). Isso ja eh suficiente pra bypassar
    muitas protecoes Akamai mesmo de IPs de cloud.

    Se curl_cffi nao estiver instalado (ex: rodando local sem `pip install`),
    cai pro requests padrao com headers de browser. Funciona se IP for
    residencial.
    """
    if _CFFI_OK and cffi_requests is not None:
        # curl_cffi com TLS fingerprint do Chrome 131 real
        s = cffi_requests.Session(impersonate="chrome131")
        print("  [INFO] CCEE session: usando curl_cffi (Chrome 131 TLS fingerprint)")
    else:
        # Fallback: requests padrao com headers de browser
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
        print("  [INFO] CCEE session: usando requests padrao "
                "(curl_cffi nao disponivel)")

    # Warm-up: visita a pagina do dataset pra coletar cookies
    try:
        r = s.get("https://dadosabertos.ccee.org.br/dataset/pld_horario",
                    timeout=30, allow_redirects=True)
        print(f"  [INFO] Warm-up CCEE: HTTP {r.status_code} "
              f"({len(r.content)} bytes)")
    except Exception as e:
        print(f"  [!] Warm-up CCEE falhou: {e.__class__.__name__}: {e}")
    return s


def _download_ccee(session, url: str, dest: Path,
                    timeout: int, retries: int, force: bool = False) -> bool:
    """Download via sessao CCEE com Referer ajustado pro dataset.
    Funciona com requests.Session ou curl_cffi.requests.Session."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return True
    if force and dest.exists():
        dest.unlink()
    headers = {"Referer": "https://dadosabertos.ccee.org.br/dataset/pld_horario"}
    for attempt in range(1, retries + 1):
        try:
            # curl_cffi nao suporta stream=True da mesma forma; baixa tudo em
            # memoria. Arquivos PLD sao pequenos (~500KB), entao OK.
            r = session.get(url, timeout=timeout, headers=headers,
                              allow_redirects=True)
            if r.status_code == 404:
                return False
            if r.status_code != 200:
                preview = r.text[:200].replace('\n', ' ')
                print(f"  [!] CCEE HTTP {r.status_code}: {preview[:150]}")
                if attempt == retries:
                    return False
                continue
            content = r.content
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as f:
                f.write(content)
            tmp.replace(dest); return True
        except Exception as e:
            if attempt == retries:
                print(f"  [!] Falha download CCEE {url}: "
                      f"{e.__class__.__name__}: {e}")
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
    Retorna DF compacto com colunas: hora, mwh_total_ne, mwh_estim_ne,
    mwh_curt_ne, n_usinas.
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
        # Calcula geracao em MWh, estimada e curtailment (cada base 30min -> 0.5h)
        df["geracao_mwh"] = (pd.to_numeric(df["val_geracaoverificada"],
                                              errors="coerce")
                                .clip(lower=0) * 0.5)
        df["estimada_mwh"] = (pd.to_numeric(df["val_geracaoestimada"],
                                              errors="coerce")
                                .clip(lower=0) * 0.5)
        df["curtailment_mwh"] = (df["estimada_mwh"]
                                    - df["geracao_mwh"]).clip(lower=0)
        df["hora"] = df["din_instante"].dt.floor("h")
        agg = df.groupby("hora").agg(
            mwh_total_ne=("geracao_mwh", "sum"),
            mwh_estim_ne=("estimada_mwh", "sum"),
            mwh_curt_ne=("curtailment_mwh", "sum"),
            n_usinas=("nom_usina", "nunique"),
        ).reset_index()
        rows.append(agg)
        del df  # libera memoria
    if not rows:
        return pd.DataFrame(columns=["hora", "mwh_total_ne", "mwh_estim_ne",
                                       "mwh_curt_ne", "n_usinas"])
    out = (pd.concat(rows, ignore_index=True)
              .groupby("hora")
              .agg(mwh_total_ne=("mwh_total_ne", "sum"),
                   mwh_estim_ne=("mwh_estim_ne", "sum"),
                   mwh_curt_ne=("mwh_curt_ne", "sum"),
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


def carregar_irradiancia_nasa(cfg: dict, dt_ini: date, dt_fim: date) -> pd.DataFrame:
    """Puxa GHI (Global Horizontal Irradiance, W/m^2) horario da NASA POWER
    para a coordenada do Mauriti. API publica, sem auth.
    https://power.larc.nasa.gov/api/temporal/hourly/point

    NASA POWER tem latencia de ~5-7 dias para o dado mais recente.
    Cache local em ./cache/nasa_power_<lat>_<lon>.csv (atualiza tudo se ja existe).

    Retorna DF com colunas: hora (datetime), ghi (W/m^2), temp (C).
    Se a API falhar, retorna DF vazio (gracioso).
    """
    print(f"\n[*] NASA POWER: GHI horario para Mauriti "
          f"({cfg['mauriti_lat']}, {cfg['mauriti_lon']})...")
    cache = _ensure_dir(Path(cfg["cache_dir"]) / "nasa")
    cache_file = cache / f"power_{cfg['mauriti_lat']}_{cfg['mauriti_lon']}.csv"

    # Estima cobertura esperada para validar cache (24h * dias do periodo)
    n_dias_periodo = max(1, (dt_fim - dt_ini).days + 1)
    horas_esperadas = n_dias_periodo * 24
    cobertura_min = 0.70  # exigir ao menos 70% de cobertura

    # Se cache existe e foi atualizado hoje E tem cobertura suficiente, le dele
    if cache_file.exists():
        try:
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime).date()
            if mtime == date.today():
                df = pd.read_csv(cache_file)
                df["hora"] = pd.to_datetime(df["hora"])
                df = df[(df["hora"] >= pd.Timestamp(dt_ini)) &
                        (df["hora"] <= pd.Timestamp(dt_fim) + pd.Timedelta(days=1))]
                cobertura = len(df) / horas_esperadas
                if cobertura >= cobertura_min:
                    print(f"  [cache hoje] {len(df):,} horas "
                          f"({100*cobertura:.0f}% do periodo)")
                    return df
                else:
                    print(f"  [!] Cache hoje tem cobertura BAIXA "
                          f"({len(df):,}h / {horas_esperadas:,}h = "
                          f"{100*cobertura:.0f}%) — re-baixando do zero")
                    cache_file.unlink()  # remove cache ruim
        except Exception as e:
            print(f"  [!] Erro lendo cache ({e}) — re-baixando")

    # NASA POWER API: max 366 dias por chamada -> dividir em pedacos anuais
    today = date.today()
    # Latencia NASA: pega ate 5 dias atras pra evitar dados ruins
    dt_fim_real = min(dt_fim, today - timedelta(days=5))
    if dt_fim_real <= dt_ini:
        print(f"  [!] Janela invalida (dt_fim={dt_fim_real} <= dt_ini={dt_ini})")
        return pd.DataFrame(columns=["hora", "ghi", "temp"])

    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    all_rows = []
    cur = dt_ini
    while cur <= dt_fim_real:
        chunk_end = min(cur + timedelta(days=300), dt_fim_real)
        params = {
            "parameters": "ALLSKY_SFC_SW_DWN,T2M",
            "community": "RE",
            "longitude": cfg["mauriti_lon"],
            "latitude": cfg["mauriti_lat"],
            "start": cur.strftime("%Y%m%d"),
            "end": chunk_end.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": "UTC",
        }
        try:
            r = requests.get(base_url, params=params,
                              timeout=cfg["request_timeout"])
            if r.status_code != 200:
                print(f"  [!] NASA POWER {cur} a {chunk_end}: HTTP {r.status_code}")
                cur = chunk_end + timedelta(days=1)
                continue
            data = r.json()
            ghi = data.get("properties", {}).get("parameter", {}).get(
                "ALLSKY_SFC_SW_DWN", {})
            t2m = data.get("properties", {}).get("parameter", {}).get(
                "T2M", {})
            for ts_str, ghi_val in ghi.items():
                # ts_str formato YYYYMMDDHH em UTC
                try:
                    ts = datetime.strptime(ts_str, "%Y%m%d%H")
                except ValueError:
                    continue
                # Converte de UTC para BRT (UTC-3)
                ts_brt = ts - timedelta(hours=3)
                # NASA usa -999 para "sem medida" — tratar como 0 (ao inves
                # de descartar) preserva a linha temporal e o merge inner.
                # Para horas noturnas isso eh correto (GHI=0). Para horas
                # diurnas com nuvem espessa, GHI=0 eh uma aproximacao OK.
                if ghi_val in (None, -999, "-999"):
                    ghi_v = 0.0
                else:
                    ghi_v = float(ghi_val)
                temp_v = t2m.get(ts_str)
                temp_v = (float(temp_v) if temp_v not in (None, -999, "-999")
                          else None)
                all_rows.append({"hora": ts_brt, "ghi": ghi_v, "temp": temp_v})
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"  [!] NASA POWER chunk {cur}-{chunk_end}: {e}")
        cur = chunk_end + timedelta(days=1)

    if not all_rows:
        print("  [!] NASA POWER nao retornou dados")
        return pd.DataFrame(columns=["hora", "ghi", "temp"])

    df = pd.DataFrame(all_rows)
    # Note: nao usa dropna(subset=['ghi']) — agora -999 vira 0 acima.
    df = df.drop_duplicates("hora").sort_values("hora")
    # Salva cache
    try:
        df.to_csv(cache_file, index=False)
    except Exception:
        pass
    print(f"  -> {len(df):,} horas, GHI medio diurno "
          f"{df[df['ghi'] > 50]['ghi'].mean():.0f} W/m2")
    return df


def _baixar_pld_dados_abertos(year: int, cfg: dict,
                                 session=None) -> Path | None:
    """Tenta baixar PLD do portal Dados Abertos CCEE via API CKAN.
    Retorna o caminho do arquivo baixado se sucesso, None se falha.

    Estratégia:
    1. Chama API CKAN: GET /api/3/action/package_show?id=pld_horario
    2. Acha o resource com nome 'pld_horario_<year>'
    3. Baixa o CSV do url do resource
    4. Salva em pld_data/pld_horario_<year>.csv (sobrescrevendo se ja existe)

    IMPORTANTE: usa sessao CCEE com TLS fingerprint de Chrome via curl_cffi
    (se disponivel). Sem isso, o Akamai detecta IP de cloud + python-requests
    TLS fingerprint e retorna 403 "Bloqueio Manutençao".

    Vantagem da API CKAN: as URLs dos resources sao UUIDs estaveis. Se
    a CCEE rotacionar o UUID, a API atualiza e o script continua funcionando.
    """
    # Se nao recebeu sessao, cria uma nova (com curl_cffi se disponivel)
    if session is None:
        session = _ccee_session()

    api_url = ("https://dadosabertos.ccee.org.br/api/3/action/"
               "package_show?id=pld_horario")
    try:
        # API CKAN espera Accept JSON, complementa a sessao
        api_headers = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://dadosabertos.ccee.org.br/dataset/pld_horario",
            "X-Requested-With": "XMLHttpRequest",
        }
        r = session.get(api_url, timeout=cfg["request_timeout"],
                          headers=api_headers)
        if r.status_code != 200:
            print(f"  [!] API CKAN retornou {r.status_code} para {year} "
                  f"(URL: {api_url})")
            # Debug: mostra primeiros 200 chars da resposta pra detectar
            # se eh erro de Cloudflare/WAF (HTML), rate limit, etc.
            preview = r.text[:200].replace('\n', ' ').strip()
            if preview:
                print(f"  [!] Resposta: {preview}")
            return None
        data = r.json()
        resources = data.get("result", {}).get("resources", [])
        target = f"pld_horario_{year}"
        for res in resources:
            nome = (res.get("name") or "").lower().strip()
            if nome.startswith(target.lower()):
                csv_url = res.get("url")
                if not csv_url:
                    continue
                # Baixa o CSV usando a MESMA sessao (cookies validos)
                csv_headers = {
                    "Accept": "text/csv, application/csv, text/plain, */*",
                    "Referer": "https://dadosabertos.ccee.org.br/dataset/pld_horario",
                }
                csv_resp = session.get(csv_url,
                                          timeout=cfg["request_timeout"],
                                          headers=csv_headers,
                                          allow_redirects=True)
                if csv_resp.status_code != 200:
                    print(f"  [!] Download {target} HTTP {csv_resp.status_code} "
                          f"(URL: {csv_url})")
                    continue
                content = csv_resp.content
                if len(content) < 1000:
                    print(f"  [!] CSV {target} muito pequeno "
                          f"({len(content)} bytes), descartando")
                    continue
                # Salva em pld_data/ (sobrescreve)
                dest_dir = Path("./pld_data")
                dest_dir.mkdir(exist_ok=True)
                dest = dest_dir / f"pld_horario_{year}.csv"
                dest.write_bytes(content)
                print(f"  [OK] PLD {year} baixado de Dados Abertos CCEE "
                      f"({len(content)/1024:.0f} KB) -> {dest}")
                return dest
        print(f"  [!] Resource 'pld_horario_{year}' nao encontrado na API "
              f"(resources disponiveis: "
              f"{[r.get('name') for r in resources[:5]]}...)")
        return None
    except Exception as e:
        # Captura tudo: requests.RequestException, curl_cffi.requests.errors.*,
        # JSONDecodeError, KeyError, etc.
        print(f"  [!] Erro baixando PLD {year} via Dados Abertos: "
              f"{e.__class__.__name__}: {e}")
        return None


def carregar_pld(cfg: dict, dt_ini: date, dt_fim: date,
                  submercado: str) -> pd.DataFrame:
    cache = _ensure_dir(Path(cfg["cache_dir"]) / "ccee")
    today = date.today()
    anos = sorted({d.year for d in pd.date_range(dt_ini, dt_fim, freq="D")})
    print(f"\n[3/4] PLD horario CCEE ({anos})...")

    frames = []
    anos_pendentes = list(anos)

    # ========== ETAPA 1: tenta baixar do portal Dados Abertos CCEE ==========
    # Esse portal foi lancado em julho/2025 e publica diariamente. Substitui
    # o site antigo da CCEE (que era bloqueado por Akamai em IPs de cloud).
    print(f"       Tentando baixar de Dados Abertos CCEE para {anos_pendentes}...")
    # Cria UMA sessao CCEE warmed-up (cookies + headers Mozilla) e reaproveita
    # entre as chamadas dos varios anos. Sem isso o IP do GitHub Actions
    # (Azure) recebe 403 do CKAN.
    ccee_sess = _ccee_session()
    for ano in list(anos_pendentes):
        # Para ano corrente, sempre re-baixa (dados sao atualizados diariamente)
        # Para anos passados, so baixa se nao existe arquivo manual
        manual_file = Path("./pld_data") / f"pld_horario_{ano}.csv"
        ja_existe = manual_file.exists() and manual_file.stat().st_size > 1000
        eh_ano_atual = (ano == today.year)
        if ja_existe and not eh_ano_atual:
            continue  # ano passado ja tem arquivo, nao precisa rebaixar
        baixado = _baixar_pld_dados_abertos(ano, cfg, session=ccee_sess)
        if baixado:
            try:
                frames.append(_read_csv_robust(baixado))
                anos_pendentes.remove(ano)
            except Exception as e:
                print(f"  [!] Erro lendo PLD {ano} baixado: {e}")

    # ========== ETAPA 2: le PLD manual do repo (fallback ou ano passado) ==========
    manual_dir = Path("./pld_data")
    if manual_dir.exists() and anos_pendentes:
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

    # ========== ETAPA 3: ultimo recurso - portal CCEE legado ==========
    if anos_pendentes:
        print(f"       Ultimo recurso: portal CCEE legado para {anos_pendentes}...")
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
                print(f"  [!] Nao baixou PLD {ano} da CCEE legacy "
                      f"(provavel bloqueio IP)")
                continue
            try:
                frames.append(_read_csv_robust(dest))
                print(f"  [OK] PLD {ano} baixado da CCEE legacy")
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

    # ===== NEW (v5.4): agregado mensal por submercado pro simulador PPA =====
    # Antes de filtrar pelo submercado configurado, calcula agregado mensal
    # de TODOS os submercados pra usar no simulador PPA (atribuido como attrs)
    pld_filtrado_periodo = pld[
        (pld["hora"] >= pd.Timestamp(dt_ini)) &
        (pld["hora"] <= pd.Timestamp(dt_fim) + pd.Timedelta(days=1))
    ].copy()
    if not pld_filtrado_periodo.empty:
        pld_filtrado_periodo["sub_norm"] = (
            pld_filtrado_periodo["sub"].astype(str).str.upper().str.strip()
        )
        # Normalizacao de nomes de submercado
        sub_map = {
            "NORDESTE": "NE", "NE": "NE",
            "SUDESTE/CENTROOESTE": "SECO", "SUDESTE": "SECO",
            "SECO": "SECO", "SE/CO": "SECO", "SE": "SECO",
            "SUL": "S", "S": "S",
            "NORTE": "N", "N": "N",
        }
        pld_filtrado_periodo["sub_code"] = (
            pld_filtrado_periodo["sub_norm"].map(sub_map)
            .fillna(pld_filtrado_periodo["sub_norm"])
        )
        pld_filtrado_periodo["mes"] = (
            pld_filtrado_periodo["hora"].dt.to_period("M").dt.to_timestamp()
        )
        mensal_por_sub = (
            pld_filtrado_periodo.groupby(["mes", "sub_code"])
            .agg(mean_pld=("pld", "mean"),
                  sum_pld=("pld", "sum"),
                  n_horas=("pld", "count"))
            .reset_index()
        )
        print(f"  PLD mensal por submercado: "
              f"{mensal_por_sub['sub_code'].nunique()} submercados, "
              f"{mensal_por_sub['mes'].nunique()} meses")
    else:
        mensal_por_sub = pd.DataFrame(columns=["mes", "sub_code", "mean_pld",
                                                "sum_pld", "n_horas"])

    pld = pld[pld["sub"].astype(str).str.upper().str.contains(submercado.upper())]
    pld = pld[(pld["hora"] >= pd.Timestamp(dt_ini)) &
              (pld["hora"] <= pd.Timestamp(dt_fim) + pd.Timedelta(days=1))]
    pld = pld[["hora", "pld"]].drop_duplicates("hora").sort_values("hora")
    pld.attrs["fallback"] = False
    pld.attrs["mensal_por_sub"] = mensal_por_sub  # pra simulador PPA
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
#  ONDA 2A: Analises adicionais (YoY, ITM/OTM, breakdown REL/CNF/ENE/PAR)
# =============================================================================

def calcular_yoy_modulacao(diario_m: pd.DataFrame, today: date) -> dict:
    """Compara desconto de modulacao do mes corrente com o mesmo mes do ano
    anterior. Retorna dict com: cur_pct, prior_pct, delta_pp, label_cur,
    label_prior, status (better/worse/equal), vazio.

    'Vazio' se nao tem dado nem do mes corrente nem do ano anterior.
    """
    if diario_m.empty:
        return {"vazio": True}
    df = diario_m.copy()
    df["dia"] = pd.to_datetime(df["dia"])
    df["ano"] = df["dia"].dt.year
    df["mes"] = df["dia"].dt.month

    # Mes corrente (ainda parcial - so dias decorridos)
    cur_mes = today.month
    cur_ano = today.year
    prior_ano = cur_ano - 1

    # Filtra mes corrente parcial
    cur_df = df[(df["ano"] == cur_ano) & (df["mes"] == cur_mes)]
    # Filtra mesmo mes do ano passado (so ate o mesmo dia, pra comparar
    # mesmas N dias decorridos -- mais justo)
    cur_dia_max = today.day
    prior_df = df[(df["ano"] == prior_ano) & (df["mes"] == cur_mes) &
                    (df["dia"].dt.day <= cur_dia_max)]

    def _desc_pct(d):
        if d.empty:
            return None
        rr = d["receita_real"].sum()
        rf = d["receita_flat"].sum()
        return float(100 * (rr - rf) / rf) if rf > 1e-9 else None

    cur_pct = _desc_pct(cur_df)
    prior_pct = _desc_pct(prior_df)

    if cur_pct is None and prior_pct is None:
        return {"vazio": True}

    # Label de mes em portugues (consistente com resto do dashboard)
    meses = ["jan", "fev", "mar", "abr", "mai", "jun",
              "jul", "ago", "set", "out", "nov", "dez"]
    label_cur = f"{meses[cur_mes-1]}/{str(cur_ano)[2:]}"
    label_prior = f"{meses[cur_mes-1]}/{str(prior_ano)[2:]}"

    delta_pp = None
    status = "no_data"
    if cur_pct is not None and prior_pct is not None:
        delta_pp = cur_pct - prior_pct
        # Desconto eh negativo (perda). Delta NEGATIVO = pior (mais desconto)
        if abs(delta_pp) < 0.5:
            status = "equal"
        elif delta_pp < 0:
            status = "worse"
        else:
            status = "better"

    return dict(
        vazio=False,
        cur_pct=cur_pct, prior_pct=prior_pct,
        delta_pp=delta_pp, status=status,
        label_cur=label_cur, label_prior=label_prior,
        cur_n_dias=int(len(cur_df)), prior_n_dias=int(len(prior_df)),
    )


def calcular_in_out_money(geracao_horaria: pd.DataFrame,
                              pld: pd.DataFrame, cur_first: date) -> dict:
    """Analisa quantas horas Mauriti gerou em PLD acima da media mensal
    (in-the-money / hora valiosa) vs abaixo (out-the-money / hora barata).

    Util pra entender se solar pega ou nao as horas caras do mes. Ajuda
    a contextualizar o desconto de modulacao em termos hora-a-hora.
    """
    if geracao_horaria.empty or pld.empty:
        return {"vazio": True}

    # Filtra mes corrente
    cur_first_ts = pd.Timestamp(cur_first)
    df = geracao_horaria.merge(pld, on="hora", how="left").dropna(subset=["pld"])
    df = df[df["hora"] >= cur_first_ts]
    if df.empty:
        return {"vazio": True}

    pld_medio_mes = float(df["pld"].mean())
    if pld_medio_mes <= 0:
        return {"vazio": True}

    df["receita"] = df["mwh"] * df["pld"]
    # Filtra so horas com geracao (Mauriti so gera de dia)
    df_gen = df[df["mwh"] > 0]
    if df_gen.empty:
        return {"vazio": True}

    itm = df_gen[df_gen["pld"] > pld_medio_mes]
    otm = df_gen[df_gen["pld"] <= pld_medio_mes]

    total_horas_gen = int(len(df_gen))
    total_mwh = float(df_gen["mwh"].sum())
    total_receita = float(df_gen["receita"].sum())

    def _stats(group, label):
        n = int(len(group))
        mwh = float(group["mwh"].sum())
        receita = float(group["receita"].sum())
        return dict(
            label=label, n_horas=n, mwh=mwh, receita=receita,
            pct_horas=100*n/total_horas_gen if total_horas_gen > 0 else 0,
            pct_mwh=100*mwh/total_mwh if total_mwh > 0 else 0,
            pct_receita=100*receita/total_receita if total_receita > 0 else 0,
            pld_medio_no_grupo=float(group["pld"].mean()) if n > 0 else 0,
        )

    return dict(
        vazio=False,
        pld_referencia=pld_medio_mes,
        total_horas_gen=total_horas_gen,
        total_mwh=total_mwh,
        total_receita=total_receita,
        itm=_stats(itm, "In-the-money (PLD > média)"),
        otm=_stats(otm, "Out-of-the-money (PLD ≤ média)"),
    )


def calcular_curtailment_por_razao(df_mauriti: pd.DataFrame,
                                       pld: pd.DataFrame) -> pd.DataFrame:
    """Agrega curtailment por codigo de razao do ONS (REL/CNF/ENE/PAR),
    calcula MWh e perda financeira (curt × PLD) de cada categoria.

    Retorna DataFrame com colunas: razao, label, ressarcivel,
    mwh_total, n_horas, perd_rs, pct_mwh, pct_rs.
    """
    if df_mauriti.empty or "cod_razaorestricao" not in df_mauriti.columns:
        return pd.DataFrame()

    df = df_mauriti.copy()
    # Mantem so cortes reais (curtailment > 0)
    df = df[df["curtailment_mwh"] > 0]
    if df.empty:
        return pd.DataFrame()

    # Cruza com PLD para perda financeira
    if not pld.empty:
        df = df.merge(pld[["hora", "pld"]], left_on="din_instante",
                          right_on="hora", how="left")
        df["perd_rs"] = df["curtailment_mwh"] * df["pld"].fillna(0)
    else:
        df["perd_rs"] = 0.0

    # Agrega
    agg = df.groupby("cod_razaorestricao").agg(
        mwh_total=("curtailment_mwh", "sum"),
        n_eventos=("curtailment_mwh", "count"),
        perd_rs=("perd_rs", "sum"),
    ).reset_index()
    agg = agg.rename(columns={"cod_razaorestricao": "razao"})

    # Adiciona labels e flag de ressarcivel
    agg["label"] = agg["razao"].map(RAZAO_LABEL).fillna("Outros")
    agg["ressarcivel"] = agg["razao"].map(RAZAO_RESSARCIVEL).fillna(False)

    # Percentuais
    total_mwh = agg["mwh_total"].sum()
    total_rs = agg["perd_rs"].sum()
    agg["pct_mwh"] = 100 * agg["mwh_total"] / total_mwh if total_mwh > 0 else 0
    agg["pct_rs"] = 100 * agg["perd_rs"] / total_rs if total_rs > 0 else 0

    # Ordena por MWh decrescente
    agg = agg.sort_values("mwh_total", ascending=False).reset_index(drop=True)
    return agg


def g_donut_curtailment_razao(df_razao: pd.DataFrame) -> go.Figure:
    """Gera donut chart com breakdown REL/CNF/ENE/PAR (% MWh).
    Cores: REL/CNF = ressarciveis (laranja escuro), ENE/PAR = nao (cinza).
    """
    if df_razao.empty:
        return go.Figure()

    # Cores por categoria
    cores = {
        "REL": "#a8442f",  # ressarcivel - laranja escuro
        "CNF": "#d57255",  # ressarcivel - laranja claro
        "ENE": "#857d72",  # nao-ressarcivel - cinza
        "PAR": "#c8c0ad",  # nao-ressarcivel - bege
    }
    colors = [cores.get(r, "#999999") for r in df_razao["razao"]]

    # Labels descritivas
    labels = [f"{r} ({l})" for r, l in
                  zip(df_razao["razao"], df_razao["label"])]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=df_razao["mwh_total"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#fafaf6", width=2)),
        textposition="outside",
        textinfo="label+percent",
        textfont=dict(family="IBM Plex Sans", size=11),
        hovertemplate="<b>%{label}</b><br>" +
                       "%{value:,.0f} MWh (%{percent})<br>" +
                       "<extra></extra>",
        sort=False,
    )])

    total_mwh = df_razao["mwh_total"].sum()
    fig.add_annotation(
        text=f"<b>{total_mwh:,.0f}</b><br>"
              f"<span style='font-size:10px;color:#857d72'>MWh total</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(family="IBM Plex Sans", size=18, color="#1a1715"),
    )
    fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        height=340,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig



def eventos_elegiveis_ren1030(df_mauriti: pd.DataFrame) -> pd.DataFrame:
    """Filtra cortes do Mauriti que sao elegiveis a ressarcimento sob a
    REN 1.030/2022 -- razoes REL (indisponibilidade externa) e CNF
    (confiabilidade). Retorna DF agregado por evento (consecutivo no
    tempo, mesma usina, mesma razao).

    Colunas: data_inicio, data_fim, hora_inicio, hora_fim, usina, razao,
             origem, mwh_cortado, duracao_h, pld_medio.
    """
    if df_mauriti.empty:
        return pd.DataFrame()

    # Filtra apenas cortes reais com razao ressarcivel (REL/CNF)
    cortes = df_mauriti[
        (df_mauriti["curtailment_mw"] > 0.01)
        & (df_mauriti["cod_razaorestricao"].isin(["REL", "CNF"]))
    ].copy()

    if cortes.empty:
        return pd.DataFrame()

    # Ordena por usina + razao + tempo (importante: ordem garante consecutividade)
    cortes = cortes.sort_values(
        ["nom_usina", "cod_razaorestricao", "din_instante"])

    # Detecta "eventos": agrupa linhas consecutivas da mesma usina+razao
    # cuja diferenca de tempo entre instantes <= 1h (intervalo dataset = 30min).
    # delta dentro de cada (usina, razao) -- primeiro elemento sempre NaN.
    cortes["delta"] = cortes.groupby(
        ["nom_usina", "cod_razaorestricao"])["din_instante"].diff()
    # Cada True marca inicio de um novo evento (gap > 1h ou primeira linha do grupo)
    novo_evento = (
        (cortes["delta"] > pd.Timedelta(hours=1))
        | cortes["delta"].isna()
    )
    # cumsum global da serie boolean -> IDs unicos crescentes por evento
    cortes["evento_id"] = novo_evento.cumsum()

    eventos = cortes.groupby("evento_id").agg(
        usina=("nom_usina", "first"),
        razao=("cod_razaorestricao", "first"),
        origem=("cod_origemrestricao", "first"),
        data_inicio=("din_instante", "min"),
        data_fim=("din_instante", "max"),
        mwh_cortado=("curtailment_mwh", "sum"),
        pld_medio=("pld", "mean"),
    ).reset_index(drop=True)

    eventos["duracao_h"] = (
        (eventos["data_fim"] - eventos["data_inicio"]).dt.total_seconds() / 3600
        + 0.5  # cada linha cobre 30min, ultima linha contribui 30min
    )
    eventos["hora_inicio"] = eventos["data_inicio"].dt.strftime("%H:%M")
    eventos["hora_fim"] = eventos["data_fim"].dt.strftime("%H:%M")
    eventos["data_inicio_str"] = eventos["data_inicio"].dt.strftime("%Y-%m-%d")
    eventos["data_fim_str"] = eventos["data_fim"].dt.strftime("%Y-%m-%d")
    # Origem legivel: "LOC" -> "LOC — Local (near plant)"
    eventos["origem_label"] = eventos["origem"].map(
        lambda x: ORIGEM_LABEL.get(x, x))
    eventos = eventos.sort_values("data_inicio", ascending=False)

    # Reordena colunas (mantem 'origem' = codigo ONS oficial + 'origem_label' legivel)
    cols_ord = ["data_inicio_str", "hora_inicio", "data_fim_str", "hora_fim",
                "duracao_h", "usina", "razao", "origem", "origem_label",
                "mwh_cortado", "pld_medio"]
    return eventos[cols_ord]


def metricas_ren1030(eventos: pd.DataFrame, df_mauriti: pd.DataFrame) -> dict:
    """Retorna KPIs do tracker REN 1.030."""
    if eventos.empty:
        return {"vazio": True, "n_eventos": 0, "mwh_total": 0.0,
                "n_dias": 0, "n_usinas": 0, "pct_total": 0.0,
                "rel_mwh": 0.0, "cnf_mwh": 0.0}

    mwh_total_eligible = float(eventos["mwh_cortado"].sum())
    mwh_total_all = float(df_mauriti.loc[
        df_mauriti["curtailment_mw"] > 0.01, "curtailment_mwh"].sum())

    return dict(
        vazio=False,
        n_eventos=int(len(eventos)),
        mwh_total=mwh_total_eligible,
        n_dias=int(eventos["data_inicio_str"].nunique()),
        n_usinas=int(eventos["usina"].nunique()),
        pct_total=(100 * mwh_total_eligible / mwh_total_all
                    if mwh_total_all > 0 else 0.0),
        rel_mwh=float(eventos[eventos["razao"] == "REL"]["mwh_cortado"].sum()),
        cnf_mwh=float(eventos[eventos["razao"] == "CNF"]["mwh_cortado"].sum()),
        n_eventos_rel=int((eventos["razao"] == "REL").sum()),
        n_eventos_cnf=int((eventos["razao"] == "CNF").sum()),
    )


# =============================================================================
#  IRRADIANCIA NASA POWER - cruzamento com curtailment
# =============================================================================

def cruzar_irradiancia(df_mauriti: pd.DataFrame,
                         irradiancia: pd.DataFrame) -> pd.DataFrame:
    """Cruza geracao+curtailment Mauriti com GHI da NASA POWER.
    Retorna DF horario com colunas: hora, mwh_gen, mwh_curt, ghi, temp.

    IMPORTANTE: o `mwh_curt` aqui considera APENAS linhas com razao ONS
    oficialmente classificada (REL/CNF/ENE/PAR). Linhas com razao
    'DESCONHECIDA' (subgeracao nao-classificada, ramp-up matinal, etc)
    sao EXCLUIDAS porque nao representam curtailment 'verdadeiro' --
    sao apenas artefatos de calibração da estimativa do ONS.

    Ja `mwh_gen` e `mwh_estim` continuam refletindo o total real da usina."""
    if df_mauriti.empty or irradiancia.empty:
        return pd.DataFrame()
    df = df_mauriti.copy()
    df["hora"] = df["din_instante"].dt.floor("h")
    # Curtailment "classificado" = so o que tem razao ONS oficial.
    # IMPLEMENTACAO ROBUSTA (pandas 3.0+ / CoW friendly): usa mascara
    # multiplicativa em vez de .where(), garantindo que o resultado seja
    # sempre uma Series de float64 sem ambiguidade de tipos.
    razoes_oficiais = ["REL", "CNF", "ENE", "PAR"]
    mask = df["cod_razaorestricao"].isin(razoes_oficiais).astype(float)
    df["curt_classificado_mwh"] = (df["curtailment_mwh"].astype(float)
                                     * mask).fillna(0.0)
    agg = df.groupby("hora", as_index=False).agg(
        mwh_gen=("geracao_mwh", "sum"),
        mwh_curt=("curt_classificado_mwh", "sum"),
        mwh_estim=("estimada_mwh", "sum"),
    )

    # ===== DIAGNOSTICO GRANULAR =====
    # Mostra o cruz_irr resumido por hora_dia (mean) ANTES de retornar.
    # Compare com o [DEBUG] anterior pra identificar onde o dado some.
    print("\n  [DEBUG cruzar_irradiancia] resumo POR HORA DO DIA:")
    print("       hora_dia | n_buckets | sum_curt(MWh) | mean_curt(MWh/h) | "
          "max_curt(MWh)")
    diag = agg.copy()
    diag["hora_dia"] = pd.to_datetime(diag["hora"]).dt.hour
    for h, sub in diag.groupby("hora_dia"):
        if h in (0, 1, 2, 3, 22, 23):
            continue
        print(f"         {h:>2}h    | {len(sub):>9,} | "
              f"{sub['mwh_curt'].sum():>13,.0f} | "
              f"{sub['mwh_curt'].mean():>16,.2f} | "
              f"{sub['mwh_curt'].max():>13,.1f}")

    out = agg.merge(irradiancia[["hora", "ghi", "temp"]],
                     on="hora", how="inner")
    print(f"  [DEBUG] cruz_irr final: {len(out):,} linhas "
          f"(de {len(agg):,} agg + {len(irradiancia):,} NASA, inner join)")

    # ===== DIAGNOSTICO DO out APOS MERGE NASA =====
    print("\n  [DEBUG cruzar APOS merge NASA] resumo POR HORA DO DIA:")
    print("       hora_dia | n_apos_merge | sum_curt(MWh) | mean_curt(MWh/h)")
    out_d = out.copy()
    out_d["hora_dia"] = pd.to_datetime(out_d["hora"]).dt.hour
    for h, sub in out_d.groupby("hora_dia"):
        if h < 4 or h > 19:
            continue
        n_buckets = len(sub)
        sum_c = sub["mwh_curt"].sum()
        mean_c = sub["mwh_curt"].mean() if n_buckets > 0 else 0
        # Quantos buckets têm curt > 0?
        n_com_curt = int((sub["mwh_curt"] > 0.01).sum())
        print(f"         {h:>2}h    | {n_buckets:>12,} | "
              f"{sum_c:>13,.0f} | {mean_c:>16,.2f} "
              f"(n com curt>0: {n_com_curt})")
    return out


def metricas_irradiancia(cruz: pd.DataFrame) -> dict:
    """KPIs do cruzamento irradiancia x curtailment."""
    if cruz.empty:
        return {"vazio": True}
    sol_pleno = cruz[cruz["ghi"] > 600]  # horas de "ceu limpo"
    if sol_pleno.empty:
        return {"vazio": True}
    horas_corte = sol_pleno[sol_pleno["mwh_curt"] > 0.01]
    pct_horas_corte = 100 * len(horas_corte) / len(sol_pleno)
    cf_sol = (100 * sol_pleno["mwh_curt"].sum()
              / max(sol_pleno["mwh_estim"].sum(), 1e-9))
    return dict(
        vazio=False,
        ghi_medio_diurno=float(cruz[cruz["ghi"] > 50]["ghi"].mean()),
        ghi_pico=float(cruz["ghi"].max()),
        n_horas_pleno=int(len(sol_pleno)),
        pct_horas_corte_em_sol=pct_horas_corte,
        cf_em_sol_pleno=cf_sol,
        temp_media_corte=float(horas_corte["temp"].mean())
            if not horas_corte.empty else 0.0,
    )


# =============================================================================
#  TENDENCIA - sparkline 30/90/365 dias
# =============================================================================

def calcular_tendencia(df_mauriti: pd.DataFrame, today: date) -> dict:
    """Calcula CF% nas 3 janelas (30, 90, 365 dias) terminadas em today,
    e o sentido da tendencia (piorando, estavel, melhorando)."""
    if df_mauriti.empty:
        return {"vazio": True}
    df = df_mauriti.copy()
    df["dia"] = df["din_instante"].dt.date
    today_ts = pd.Timestamp(today)
    out = {"vazio": False}
    valores = []
    for nome, ndays in (("d30", 30), ("d90", 90), ("d365", 365)):
        ini = (today_ts - pd.Timedelta(days=ndays)).date()
        sub = df[df["dia"] >= ini]
        e = float(sub["estimada_mwh"].sum())
        c = float(sub["curtailment_mwh"].sum())
        cf = (100 * c / e) if e > 0 else 0.0
        out[f"cf_{nome}"] = cf
        out[f"mwh_{nome}"] = c
        valores.append(cf)
    # Tendencia: 30d > 90d > 365d -> piorando
    cf30, cf90, cf365 = valores
    if cf30 > cf90 * 1.10 and cf90 > cf365 * 1.05:
        out["tendencia"] = "piorando"
    elif cf30 < cf90 * 0.90 and cf90 < cf365 * 0.95:
        out["tendencia"] = "melhorando"
    else:
        out["tendencia"] = "estavel"
    out["delta_30_vs_365"] = cf30 - cf365
    return out


# =============================================================================
#  HEATMAP DOW x HORA (dia-da-semana x hora)
# =============================================================================

def heatmap_dow_hora(df_mauriti: pd.DataFrame) -> pd.DataFrame:
    """Calcula CF% por (dia_da_semana, hora). Pivot retornado pronto pra plot.
    DOW: 0=Mon, 6=Sun."""
    if df_mauriti.empty:
        return pd.DataFrame()
    df = df_mauriti.copy()
    df["hora_dia"] = df["din_instante"].dt.hour
    df["dow"] = df["din_instante"].dt.dayofweek
    df = df[(df["hora_dia"] >= 5) & (df["hora_dia"] <= 19)]
    pivot_curt = (df.groupby(["hora_dia", "dow"])["curtailment_mwh"]
                    .sum().unstack(fill_value=0))
    pivot_estim = (df.groupby(["hora_dia", "dow"])["estimada_mwh"]
                     .sum().unstack(fill_value=0))
    cf = (100 * pivot_curt / pivot_estim.replace(0, np.nan)).fillna(0)
    return cf


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
    # MELHORIA: cortar eixo X ate today + 1 buffer, igual ao mod tracker
    days_to_show = min(today.day + 1, days_in_month)
    all_days = [cur_first + timedelta(days=i) for i in range(days_to_show)]

    df = df_mauriti.copy()
    df["dia"] = df["din_instante"].dt.date
    cur = df[df["dia"] >= cur_first]

    # Conserva calculo full do mes pra retornar 'daily' completo (usado por callers)
    daily_full = (cur.groupby("dia").agg(estim=("estimada_mwh","sum"),
                                            real=("geracao_mwh","sum"),
                                            curt=("curtailment_mwh","sum"))
                     .reindex([cur_first + timedelta(days=i)
                                for i in range(days_in_month)], fill_value=0)
                     .reset_index().rename(columns={"index":"dia"}))
    daily_full["cf"] = (100 * daily_full["curt"] /
                         daily_full["estim"].replace(0, np.nan)).fillna(0)
    fut_full = daily_full["dia"] > today
    daily_full.loc[fut_full, ["estim","real","curt","cf"]] = None

    # Subset para o grafico (so dias decorridos + buffer)
    daily = daily_full.iloc[:days_to_show].copy()

    # CF% medio do trimestre fechado anterior
    ref_start = cur_first - timedelta(days=90)
    ref = df[(df["dia"] >= ref_start) & (df["dia"] < cur_first)]
    if not ref.empty:
        ref_d = ref.groupby("dia").agg(e=("estimada_mwh","sum"),
                                          c=("curtailment_mwh","sum"))
        ref_cf = float(100 * ref_d["c"].sum() / max(ref_d["e"].sum(), 1e-9))
    else:
        ref_cf = None

    # Labels do eixo X: numero do dia + dia da semana abreviado
    dow_short = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    x_labels = []
    for d in daily["dia"]:
        if pd.isna(d):
            x_labels.append("")
        else:
            dow_idx = d.weekday()
            # Weekend em bold sutil
            if dow_idx >= 5:
                x_labels.append(f"<b>{d.strftime('%d')}</b><br>"
                                f"<span style='font-size:9px;color:#857d72'>"
                                f"<b>{dow_short[dow_idx]}</b></span>")
            else:
                x_labels.append(f"{d.strftime('%d')}<br>"
                                f"<span style='font-size:9px;color:#857d72'>"
                                f"{dow_short[dow_idx]}</span>")

    # Pre-computa percentuais pra tooltip rico (Realizada/Cortada/Esperada)
    pct_real = [(100 * r / e) if (e is not None and e > 0)
                else None for r, e in zip(daily["real"], daily["estim"])]
    pct_curt = [(100 * c / e) if (e is not None and e > 0)
                else None for c, e in zip(daily["curt"], daily["estim"])]
    delta_ref = ([cf - ref_cf if (cf is not None and ref_cf is not None) else None
                  for cf in daily["cf"]] if ref_cf is not None
                 else [None] * len(daily))

    fig = go.Figure()

    # ---------- ZONAS DE SEVERIDADE NO Y2 (CF%) ----------
    if len(x_labels) > 0:
        x_left, x_right = -0.5, len(x_labels) - 0.5
        # Zona OK: 0% a 15% (verde claro)
        fig.add_shape(type="rect", xref="x", yref="y2",
                       x0=x_left, x1=x_right, y0=0, y1=15,
                       fillcolor="rgba(45,90,61,0.04)",
                       line=dict(width=0), layer="below")
        # Zona ATENCAO: 15% a 30% (amarelo claro)
        fig.add_shape(type="rect", xref="x", yref="y2",
                       x0=x_left, x1=x_right, y0=15, y1=30,
                       fillcolor="rgba(217,140,30,0.05)",
                       line=dict(width=0), layer="below")
        # Zona CRITICA: > 30% (vermelho claro)
        fig.add_shape(type="rect", xref="x", yref="y2",
                       x0=x_left, x1=x_right, y0=30, y1=100,
                       fillcolor="rgba(217,46,15,0.05)",
                       line=dict(width=0), layer="below")

    # ---------- BARRAS (realizada + curtailment, stacked) ----------
    customdata_real = list(zip(daily["estim"], pct_real, delta_ref))
    fig.add_trace(go.Bar(x=x_labels, y=daily["real"], name="Geracao realizada (MWh)",
        marker=dict(color=EL["ink_2"], line=dict(width=0)),
        customdata=customdata_real,
        hovertemplate="Realizada: <b>%{y:,.0f}</b> MWh "
                      "(%{customdata[1]:.1f}% do esperado)<extra></extra>"))
    customdata_curt = list(zip(daily["estim"], pct_curt, daily["cf"]))
    fig.add_trace(go.Bar(x=x_labels, y=daily["curt"], name="Curtailment (MWh)",
        marker=dict(color=EL["accent"], line=dict(width=0)),
        customdata=customdata_curt,
        hovertemplate="Cortada: <b>%{y:,.0f}</b> MWh "
                      "(%{customdata[1]:.1f}% do esperado)<extra></extra>"))

    # ---------- LINHA CF% ----------
    fig.add_trace(go.Scatter(x=x_labels, y=daily["cf"], name="CF% do dia",
        mode="lines+markers", yaxis="y2",
        line=dict(color=EL["accent_today"], width=2.5),
        marker=dict(size=8, color=EL["accent_today"],
                     line=dict(color=EL["panel"], width=1.5)),
        customdata=delta_ref,
        hovertemplate="CF: <b>%{y:.2f}%</b>"
                      "<extra></extra>"))

    # ---------- MARKER DIA ATUAL ----------
    today_str = today.strftime("%d")
    # Encontra index do today_str no x_labels (precisa procurar nos labels que comecam com today_str)
    idx_today = None
    for i, lab in enumerate(x_labels):
        if lab.startswith(today_str) or lab.startswith(f"<b>{today_str}"):
            idx_today = i
            break
    if idx_today is not None:
        cf_today = daily["cf"].iloc[idx_today]
        if pd.notna(cf_today):
            fig.add_trace(go.Scatter(
                x=[x_labels[idx_today]], y=[cf_today],
                mode="markers+text", yaxis="y2",
                marker=dict(size=18, color=EL["accent_today"],
                             line=dict(color="#1a1a1a", width=2),
                             symbol="circle"),
                text=["today"],
                textposition="top center",
                textfont=dict(family="'IBM Plex Mono'", size=10,
                              color="#1a1a1a"),
                name="today",
                hoverinfo="skip",
                showlegend=False,
            ))

    # ---------- ANNOTATION PIOR DIA ----------
    cf_validos = daily["cf"].dropna()
    if len(cf_validos) > 0:
        idx_pior = cf_validos.idxmax()
        pior_cf = daily.loc[idx_pior, "cf"]
        pior_dia = daily.loc[idx_pior, "dia"]
        if pior_cf >= 25 and idx_pior != idx_today:  # so destaca se ruim e nao for hoje
            fig.add_annotation(
                x=x_labels[idx_pior], y=pior_cf,
                yref="y2",
                text=f"<b>worst: day {pior_dia.day} · {pior_cf:.1f}%</b>",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                arrowcolor=EL["accent_today"],
                ax=-50, ay=-40,
                bgcolor="#fafaf6", borderpad=6, borderwidth=1,
                bordercolor=EL["accent_today"],
                font=dict(family="'IBM Plex Mono'", size=10,
                          color=EL["accent_today"]),
            )

    # ---------- LINHA CF MEDIO TRIMESTRE (mais visivel) ----------
    if ref_cf is not None:
        fig.add_hline(y=ref_cf, yref="y2",
            line=dict(color=EL["neutral"], width=1.5, dash="dash"),
            annotation_text=f"  prior-quarter avg: {ref_cf:.1f}%",
            annotation_position="top right",
            annotation=dict(font=dict(family="'IBM Plex Mono',monospace",
                                       size=10, color=EL["neutral"]),
                              bgcolor="rgba(250,250,246,0.85)",
                              borderpad=3))

    # Subtitle dinamico
    mes_capitalized = cur_first.strftime("%B/%Y")  # ex: "May/2026"
    subtitle = (f"<span style='font-size:11px;color:#857d72'>"
                f"{today.day} of {days_in_month} days elapsed "
                f"· stacked = expected gen = realized + curtailed</span>")

    lay = dict(LAY); lay.update(
        title=dict(
            text=(f"<span style='font-size:20px;font-family:Fraunces,serif;"
                  f"color:#1a1715'>Mauriti — {mes_capitalized}  ·  "
                  f"realized vs curtailed per day</span>"
                  f"<br>{subtitle}"),
            x=0.02, xanchor="left", y=0.97
        ),
        barmode="stack",
        xaxis=dict(title="", gridcolor=EL["border"],
                    tickfont=dict(family="'IBM Plex Mono',monospace", size=10)),
        yaxis=dict(title="MWh / day", gridcolor=EL["border"]),
        yaxis2=dict(title="CF% of the day", overlaying="y", side="right",
                     showgrid=False, color=EL["accent_today"],
                     ticksuffix="%", rangemode="tozero",
                     range=[0, max(60, float(cf_validos.max())+10
                                    if len(cf_validos) > 0 else 60)]),
        hovermode="x unified", height=480, bargap=0.30,
        margin=dict(l=70, r=110, t=90, b=60),
        legend=dict(orientation="h", y=-0.20,
                     font=dict(family="'IBM Plex Mono',monospace", size=11)),
    )
    fig.update_layout(**lay)
    return fig, daily_full, ref_cf, days_in_month


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
    days_to_show = min(today.day + 1, days_in_month)
    all_days = [cur_first + timedelta(days=i) for i in range(days_to_show)]

    cur = diario_m[(diario_m["dia"] >= cur_first) & (diario_m["dia"] < next_first)].copy()
    cur = (cur.set_index("dia").reindex(all_days).reset_index()
              .rename(columns={"index": "dia"}))
    fut = cur["dia"] > today
    cur.loc[fut, ["receita_real", "receita_flat", "desconto_pct"]] = None

    x = [d.strftime("%d") for d in cur["dia"]]
    fig = go.Figure()

    # ---------- ZONAS DE SEVERIDADE (background colorido em y2) ----------
    # Faixas horizontais indicam zona de risco do modulation discount.
    # NOTA: usamos shapes em yref='y2' com layer='below' pra ficar atras
    # das barras e da linha.
    if len(x) > 0:
        x_left, x_right = -0.5, len(x) - 0.5
        # Zona OK: 0% a -25% (verde claro)
        fig.add_shape(type="rect", xref="x", yref="y2",
                       x0=x_left, x1=x_right, y0=-25, y1=5,
                       fillcolor="rgba(45,90,61,0.04)",
                       line=dict(width=0), layer="below")
        # Zona ATENCAO: -25% a -40% (amarelo claro)
        fig.add_shape(type="rect", xref="x", yref="y2",
                       x0=x_left, x1=x_right, y0=-40, y1=-25,
                       fillcolor="rgba(217,140,30,0.05)",
                       line=dict(width=0), layer="below")
        # Zona CRITICA: -40% pra baixo (vermelho claro)
        fig.add_shape(type="rect", xref="x", yref="y2",
                       x0=x_left, x1=x_right, y0=-100, y1=-40,
                       fillcolor="rgba(217,46,15,0.05)",
                       line=dict(width=0), layer="below")
        # Linha zero (referencia "sem desconto") sutil
        fig.add_shape(type="line", xref="x", yref="y2",
                       x0=x_left, x1=x_right, y0=0, y1=0,
                       line=dict(color=EL["neutral"], width=0.8, dash="dash"),
                       layer="below")

    # ---------- BARRAS (receita flat / real) ----------
    fig.add_trace(go.Bar(x=x, y=cur["receita_flat"], name="Receita flat (R$)",
        marker=dict(color=EL["muted"], opacity=0.65, line=dict(width=0)),
        hovertemplate="<b>Dia %{x}</b><br>Flat: R$ %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Bar(x=x, y=cur["receita_real"], name="Receita real (R$)",
        marker=dict(color=EL["ink_2"], line=dict(width=0)),
        hovertemplate="<b>Dia %{x}</b><br>Real: R$ %{y:,.0f}<extra></extra>"))

    # ---------- LINHA % desconto ----------
    fig.add_trace(go.Scatter(x=x, y=cur["desconto_pct"], name="% desconto",
        mode="lines+markers", yaxis="y2",
        line=dict(color=EL["accent_today"], width=2.5),
        marker=dict(size=8, color=EL["accent_today"],
                     line=dict(color=EL["panel"], width=1.5)),
        hovertemplate="<b>Dia %{x}</b><br>Desc.: %{y:.2f}%<extra></extra>"))

    # ---------- MARKER DIA ATUAL ----------
    # Destaca o dia corrente com um circulo maior, borda preta, label "today"
    today_str = today.strftime("%d")
    if today_str in x:
        idx_today = x.index(today_str)
        val_today = cur["desconto_pct"].iloc[idx_today]
        if pd.notna(val_today):
            fig.add_trace(go.Scatter(
                x=[today_str], y=[val_today],
                mode="markers+text", yaxis="y2",
                marker=dict(size=16, color=EL["accent_today"],
                             line=dict(color="#1a1a1a", width=2),
                             symbol="circle"),
                text=["today"],
                textposition="top center",
                textfont=dict(family="'IBM Plex Mono'", size=10,
                              color="#1a1a1a", weight="bold"),
                name="today",
                hoverinfo="skip",
                showlegend=False,
            ))

    # ---------- ANNOTATION PIOR DIA ----------
    # So adiciona se tem pelo menos um dia critico (desconto < -40%)
    desconto_validos = cur["desconto_pct"].dropna()
    if len(desconto_validos) > 0:
        idx_pior = desconto_validos.idxmin()
        pior_val = cur.loc[idx_pior, "desconto_pct"]
        pior_dia = cur.loc[idx_pior, "dia"]
        if pior_val < -25:  # so destaca se realmente foi ruim
            fig.add_annotation(
                x=pior_dia.strftime("%d"), y=pior_val,
                yref="y2",
                text=f"<b>worst: day {pior_dia.day} · {pior_val:.1f}%</b>",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                arrowcolor=EL["accent_today"],
                ax=40, ay=30,
                bgcolor="#fafaf6", borderpad=6, borderwidth=1,
                bordercolor=EL["accent_today"],
                font=dict(family="'IBM Plex Mono'", size=10,
                          color=EL["accent_today"]),
            )

    # ---------- BENCHMARK NE ----------
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
        xaxis=dict(title=f"dia ({today.day} de {days_in_month} dias)",
                    gridcolor=EL["border"]),
        yaxis=dict(title="R$ / dia", gridcolor=EL["border"]),
        yaxis2=dict(title="% desconto modulacao", overlaying="y", side="right",
                     showgrid=False, color=EL["accent_today"], ticksuffix="%",
                     range=[-80, 10]),  # range fixo pras zonas ficarem visiveis
        hovermode="x unified", height=480, bargap=0.25, bargroupgap=0.05,
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


# ---- NOVOS CHARTS: REN 1.030, IRRADIANCIA, DOW x HORA, TENDENCIA ----

def g_ren_mensal(eventos: pd.DataFrame):
    """Stacked bar: MWh elegivel por mes (REL/CNF separados)."""
    if eventos.empty:
        return go.Figure(layout=dict(LAY,
                                       title=ed_title("REN 1.030 - eligible MWh per month", 20)))
    df = eventos.copy()
    df["mes"] = pd.to_datetime(df["data_inicio_str"]).dt.to_period("M").astype(str)
    pivot = (df.groupby(["mes", "razao"])["mwh_cortado"]
              .sum().unstack(fill_value=0))
    fig = go.Figure()
    cores = {"REL": EL["accent"], "CNF": EL["accent_2"]}
    for col in ["REL", "CNF"]:
        if col not in pivot.columns:
            continue
        fig.add_trace(go.Bar(
            x=pivot.index, y=pivot[col],
            name=f"{col} ({RAZAO_LABEL.get(col, col)})",
            marker=dict(color=cores.get(col, EL["muted"])),
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:,.1f}} MWh<extra></extra>",
        ))
    lay = dict(LAY); lay.update(
        title=ed_title("Eligible curtailment per month — REL + CNF only", 20),
        barmode="stack",
        xaxis=dict(title="", gridcolor=EL["border"]),
        yaxis=dict(title="MWh eligible", gridcolor=EL["border"]),
        hovermode="x unified", height=380,
    )
    fig.update_layout(**lay); return fig


def g_ren_origem(eventos: pd.DataFrame):
    """Bar horizontal: top origens (causas) dos eventos elegiveis."""
    if eventos.empty:
        return go.Figure(layout=dict(LAY,
                                       title=ed_title("Top causes", 20)))
    # Usa origem_label (legivel) se existir; senao fallback pra origem (codigo)
    col_label = "origem_label" if "origem_label" in eventos.columns else "origem"
    by_origem = (eventos.groupby(col_label)
                  .agg(mwh=("mwh_cortado", "sum"),
                        n=("usina", "count"))
                  .sort_values("mwh", ascending=True).tail(10))
    total_mwh = float(by_origem["mwh"].sum())
    total_evt = int(by_origem["n"].sum())

    # Cores semanticas por ORIGEM: LOC (local, controlavel) cinza-azulado,
    # SIS (sistemico, sem controle) vermelho. Detecta pelo prefixo do label.
    def _cor_origem(label: str) -> str:
        s = str(label).upper()
        if s.startswith("LOC"):
            return "#5b6b7d"  # cinza-azulado (causa local)
        if s.startswith("SIS"):
            return EL["accent_today"]  # vermelho (causa sistemica)
        return EL["accent_2"]  # fallback

    cores_bar = [_cor_origem(lab) for lab in by_origem.index]

    # Texto na ponta: percentual + MWh + events
    texto = []
    for lab, v, n in zip(by_origem.index, by_origem["mwh"], by_origem["n"]):
        pct = (100.0 * v / total_mwh) if total_mwh > 0 else 0.0
        texto.append(f"  {pct:.0f}% · {v:,.0f} MWh · {n} ev.")

    fig = go.Figure(go.Bar(
        y=by_origem.index, x=by_origem["mwh"], orientation="h",
        marker=dict(color=cores_bar, line=dict(width=0)),
        text=texto, textposition="outside",
        textfont=dict(color=EL["ink_2"], size=10,
                       family="'IBM Plex Mono',monospace"),
        hovertemplate="<b>%{y}</b><br>%{x:,.1f} MWh<extra></extra>",
        cliponaxis=False,
    ))

    # Total annotation no canto superior direito
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=1.10,
        xanchor="right", yanchor="top", showarrow=False,
        text=(f"<b>Total: {total_mwh:,.0f} MWh</b> &middot; "
              f"{total_evt} events"),
        bgcolor=EL["bg_alt"], bordercolor=EL["border"], borderwidth=1,
        borderpad=6,
        font=dict(family="'IBM Plex Mono',monospace", size=11,
                   color=EL["ink"]),
    )

    lay = dict(LAY); lay.update(
        title=ed_title("Top sources of eligible curtailment", 20),
        xaxis=dict(title="MWh", gridcolor=EL["border"],
                    range=[0, total_mwh * 0.85]),  # espaco pro texto
        yaxis=dict(gridcolor=EL["border"],
                    tickfont=dict(family="'IBM Plex Mono',monospace",
                                   size=11, color=EL["ink"])),
        height=max(300, 42 * len(by_origem) + 100),
        margin=dict(l=260, r=200, t=90, b=50),  # mais top space pro total
        showlegend=False,
    )
    fig.update_layout(**lay); return fig


def g_irrad_scatter(cruz: pd.DataFrame):
    """Scatter GHI vs curtailment (size = mwh_estim, color = mwh_curt)."""
    if cruz.empty:
        return go.Figure(layout=dict(LAY,
                                       title=ed_title("Solar resource vs curtailment", 20)))
    sub = cruz[cruz["ghi"] > 50].copy()  # so horas diurnas
    if sub.empty:
        return go.Figure(layout=dict(LAY,
                                       title=ed_title("Solar resource vs curtailment", 20)))
    sub["bin"] = pd.cut(sub["ghi"], bins=[0, 200, 400, 600, 800, 1100],
                          labels=["0-200", "200-400", "400-600",
                                   "600-800", "800+"])
    by_bin = sub.groupby("bin", observed=True).agg(
        mwh_curt=("mwh_curt", "sum"),
        mwh_estim=("mwh_estim", "sum"),
        n_horas=("ghi", "count"),
    ).reset_index()
    by_bin["cf"] = (100 * by_bin["mwh_curt"]
                    / by_bin["mwh_estim"].replace(0, np.nan)).fillna(0)
    fig = go.Figure(go.Bar(
        x=by_bin["bin"].astype(str), y=by_bin["cf"],
        marker=dict(color=EL["accent"]),
        text=[f"CF {cf:.1f}%<br>{n}h" for cf, n in
              zip(by_bin["cf"], by_bin["n_horas"])],
        textposition="outside",
        textfont=dict(family="'IBM Plex Mono',monospace", size=10,
                       color=EL["ink_2"]),
        hovertemplate=("<b>GHI %{x} W/m²</b><br>"
                        "CF: %{y:.2f}%<extra></extra>"),
    ))
    lay = dict(LAY); lay.update(
        title=ed_title("Curtailment factor by solar irradiance bin", 20),
        xaxis=dict(title="GHI bin (W/m²)", gridcolor=EL["border"]),
        yaxis=dict(title="Curtailment factor (%)", gridcolor=EL["border"],
                    ticksuffix="%"),
        height=380, showlegend=False,
    )
    fig.update_layout(**lay); return fig


def g_irrad_perfil(cruz: pd.DataFrame):
    """Perfil horario tipico: GHI medio + curtailment medio."""
    if cruz.empty:
        return go.Figure(layout=dict(LAY,
                                       title=ed_title("Hourly profile", 20)))
    df = cruz.copy()
    df["hora_dia"] = df["hora"].dt.hour
    perfil = df.groupby("hora_dia").agg(
        ghi=("ghi", "mean"),
        gen=("mwh_gen", "mean"),
        curt=("mwh_curt", "mean"),
    ).reset_index()

    # ===== DIAGNOSTICO: perfil que vai pro grafico =====
    print("\n  [DEBUG g_irrad_perfil] perfil que vai pro grafico:")
    print("       hora_dia | mean_ghi | mean_gen | mean_curt")
    for _, r in perfil.iterrows():
        h = int(r["hora_dia"])
        if h < 4 or h > 19:
            continue
        print(f"         {h:>2}h    | {r['ghi']:>8.0f} | "
              f"{r['gen']:>8.2f} | {r['curt']:>9.2f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perfil["hora_dia"], y=perfil["ghi"],
        name="GHI (W/m²)", mode="lines+markers",
        line=dict(color="#d4a017", width=2.5),
        marker=dict(size=7, color="#d4a017"),
        fill="tozeroy", fillcolor="rgba(212,160,23,0.15)",
        hovertemplate="<b>%{x}h</b><br>GHI: %{y:,.0f} W/m²<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=perfil["hora_dia"], y=perfil["curt"],
        name="Curtailment (MWh/h avg)", mode="lines+markers",
        line=dict(color=EL["accent"], width=2.5), yaxis="y2",
        marker=dict(size=7, color=EL["accent"]),
        hovertemplate="<b>%{x}h</b><br>Curt: %{y:,.2f} MWh<extra></extra>",
    ))
    lay = dict(LAY); lay.update(
        title=ed_title("Typical hour: solar resource vs cuts", 20),
        xaxis=dict(title="Hour of day", dtick=2, gridcolor=EL["border"]),
        yaxis=dict(title="GHI (W/m²)", gridcolor=EL["border"],
                    color="#d4a017"),
        yaxis2=dict(title="Avg curtailment (MWh/h)", overlaying="y",
                     side="right", showgrid=False, color=EL["accent"]),
        hovermode="x unified", height=400,
        legend=dict(orientation="h", y=-0.18,
                     font=dict(family="'IBM Plex Mono',monospace", size=11)),
    )
    fig.update_layout(**lay); return fig


def g_heatmap_dow_hora(cf_pivot: pd.DataFrame):
    """Heatmap CF% (hora x dia-da-semana). Revela padrao semanal."""
    if cf_pivot.empty:
        return go.Figure(layout=dict(LAY,
                                       title=ed_title("Weekday × hour", 20)))
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    today_dow = datetime.now().weekday()  # 0=Mon ... 6=Sun
    # Weekend em bold via HTML inline (Plotly suporta em tick labels)
    # Tambem marca dia atual com triangulo discreto.
    x_lab = []
    for d in cf_pivot.columns:
        label = dow_labels[d]
        # Bold em weekend
        if d >= 5:  # Sat (5) or Sun (6)
            label = f"<b>{label}</b>"
        # Marca dia atual
        if d == today_dow:
            label = f"{label}<br><span style='color:#d92e0f;font-size:14px'>▼</span>"
        x_lab.append(label)

    # Valores dentro das celulas — suprime <2% pra reduzir poluicao visual.
    # Tambem decide cor do texto: branco em celulas escuras (z>=30), preto
    # nas claras. Sem isso fica ilegivel em CF alto.
    z_arr = cf_pivot.values
    text_matrix = []
    textcolor_matrix = []  # nao usado direto, mas inspirou a logica abaixo
    for row in z_arr:
        text_row = []
        for v in row:
            if v >= 2.0:
                text_row.append(f"{v:.0f}")
            else:
                text_row.append("")
        text_matrix.append(text_row)

    # Escala de cor mais agressiva nos valores baixos — Mauriti tem CF
    # mediana ~10-25%, entao breakpoints baixos sao onde queremos ver
    # diferenca. Comprime os 60-100% que sao raros.
    colorscale = [
        [0.00, EL["bg"]],
        [0.03, "#fbf3e3"],
        [0.10, "#f1d8b0"],
        [0.25, "#e8b485"],
        [0.45, EL["accent_light"]],
        [0.70, EL["accent"]],
        [1.00, "#5e1d10"],
    ]

    zmax_val = min(80, float(z_arr.max()) if z_arr.size > 0 else 80)
    fig = go.Figure(go.Heatmap(
        z=z_arr, x=x_lab, y=cf_pivot.index,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(family="'IBM Plex Mono', monospace",
                       size=10, color="#1a1a1a"),
        colorscale=colorscale,
        colorbar=dict(title=dict(text="CF%", font=dict(size=11)),
                       outlinewidth=0, ticksuffix="%",
                       tickfont=dict(family="'IBM Plex Mono'", size=10)),
        zmax=zmax_val, zmin=0,
        xgap=1, ygap=1,  # linhas finas entre celulas, ajuda leitura
        hovertemplate="<b>%{x} %{y}h</b><br>CF: %{z:.1f}%<extra></extra>",
    ))

    # Texto BRANCO em celulas com z >= 30% (escuras), via annotations extras
    # sobre as celulas com valor alto — Plotly nao suporta cor de texto por
    # celula nativamente em heatmap, entao usamos add_annotation.
    n_hours, n_cols = z_arr.shape
    threshold_white_text = zmax_val * 0.55  # ~ 55% do zmax = celula escura
    for iy in range(n_hours):
        hora = cf_pivot.index[iy]
        for ix in range(n_cols):
            v = z_arr[iy, ix]
            if v >= threshold_white_text:
                fig.add_annotation(
                    x=x_lab[ix], y=hora,
                    text=f"<b>{v:.0f}</b>",
                    showarrow=False,
                    font=dict(family="'IBM Plex Mono', monospace",
                              size=10, color="#fafaf6"),
                )

    # Linhas pontilhadas delineando a "janela solar" (6h-18h)
    fig.add_shape(
        type="line",
        x0=-0.5, x1=n_cols - 0.5,
        y0=5.5, y1=5.5,
        line=dict(color="#1a1a1a", width=1.2, dash="dot"),
        layer="above",
    )
    fig.add_shape(
        type="line",
        x0=-0.5, x1=n_cols - 0.5,
        y0=18.5, y1=18.5,
        line=dict(color="#1a1a1a", width=1.2, dash="dot"),
        layer="above",
    )
    # Annotation rotulando a janela solar (lateral direita)
    fig.add_annotation(
        x=n_cols - 0.5, y=12,
        text="solar window  6h–18h",
        showarrow=False,
        xanchor="left", yanchor="middle",
        xshift=12, textangle=90,
        font=dict(family="'IBM Plex Mono'", size=9, color="#1a1a1a"),
    )

    # Worst cell highlight — destaca a pior celula com setinha
    if z_arr.size > 0:
        worst_idx = np.unravel_index(np.argmax(z_arr), z_arr.shape)
        worst_y = cf_pivot.index[worst_idx[0]]
        worst_x_col = cf_pivot.columns[worst_idx[1]]
        worst_x_label = dow_labels[worst_x_col]
        worst_val = z_arr[worst_idx]
        # So adiciona setinha se o pico for relevante (>= 15%)
        if worst_val >= 15:
            fig.add_annotation(
                x=x_lab[worst_idx[1]], y=worst_y,
                text=f"<b>worst: {worst_x_label} {worst_y}h<br>{worst_val:.0f}%</b>",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                arrowcolor=EL["accent_today"],
                ax=60, ay=-40,
                bgcolor="#fafaf6", borderpad=6, borderwidth=1,
                bordercolor=EL["accent_today"],
                font=dict(family="'IBM Plex Mono'", size=10,
                          color=EL["accent_today"]),
            )

    # Subtitle dinamico com periodo coberto
    n_buckets = int(np.count_nonzero(z_arr))
    subtitle = (f"<span style='font-size:11px;color:#857d72'>"
                f"cell value = average CF lost to curtailment (%) "
                f"· {n_buckets} active buckets</span>")

    lay = dict(LAY); lay.update(
        title=dict(
            text=(f"<span style='font-size:20px;font-family:Fraunces,serif;"
                  f"color:#1a1715'>Weekday × hour heatmap — weekly pattern</span>"
                  f"<br>{subtitle}"),
            x=0.02, xanchor="left", y=0.97
        ),
        yaxis=dict(title="Hour of day", dtick=2, autorange="reversed",
                    gridcolor=EL["border"], range=[23.5, -0.5]),
        xaxis=dict(title="", gridcolor=EL["border"], side="top"),
        height=480, margin=dict(l=70, r=130, t=90, b=40),
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
<html lang="en" data-lang="en">
<head>
<meta charset="utf-8">
<title data-i18n="page_title">Mauriti — Curtailment & Modulation Study</title>
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
  --warn:#d4a017;
}

/* ===== Dark theme ===== */
html[data-theme="dark"]{
  --bg:#1a1715; --bg-alt:#23201d; --panel:#2a2622;
  --border:#3d3833; --border2:#4a443c;
  --ink:#fafaf6; --ink-2:#e6e1d4; --muted:#a09689;
  --rule:#4a443c; --accent:#d57255; --accent-2:#b85a3d;
  --accent-today:#ff5a3a; --neutral:#a09689; --ok:#5db978;
  --warn:#e8b73e;
}

/* ===== Toolbar fixa (canto superior direito) ===== */
.toolbar{position:fixed;top:24px;right:24px;z-index:1000;
  display:flex;gap:8px;align-items:center;
  background:var(--panel);border:1px solid var(--border2);
  border-radius:4px;padding:6px;box-shadow:0 2px 8px rgba(0,0,0,0.08)}
.toolbar-btn{background:transparent;border:none;cursor:pointer;
  padding:6px 8px;color:var(--muted);font-family:inherit;
  font-size:13px;border-radius:3px;transition:all 0.15s;
  display:inline-flex;align-items:center;gap:4px;line-height:1}
.toolbar-btn:hover{color:var(--ink);background:var(--bg-alt)}
.toolbar-btn.active{background:var(--ink);color:var(--bg)}
.toolbar-btn svg{width:14px;height:14px;flex-shrink:0}
.toolbar-divider{width:1px;height:18px;background:var(--border2)}

/* ===== Freshness badges no masthead ===== */
.freshness{display:inline-flex;gap:12px;font-size:10px;
  color:var(--muted);letter-spacing:0.08em;text-transform:uppercase;
  margin-top:4px;flex-wrap:wrap}
.freshness-item{display:inline-flex;align-items:center;gap:4px}
.freshness-dot{width:6px;height:6px;border-radius:50%;background:var(--ok);
  display:inline-block;animation:pulse 2s infinite}
.freshness-dot.stale{background:var(--warn);animation:none}
.freshness-dot.old{background:var(--accent-today);animation:none}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

/* ===== Tooltips em KPIs ===== */
.kpi-tip{position:relative;display:inline-block;cursor:help;
  border-bottom:1px dotted var(--muted);margin-left:3px}
.kpi-tip::after{content:"?";display:inline-block;width:12px;height:12px;
  margin-left:3px;font-size:9px;color:var(--muted);background:var(--bg-alt);
  border-radius:50%;text-align:center;line-height:12px;font-weight:bold}
.kpi-tip:hover{border-bottom-color:var(--accent)}
.kpi-tip[data-tip]:hover::before{content:attr(data-tip);
  position:absolute;bottom:calc(100% + 8px);left:50%;transform:translateX(-50%);
  background:var(--ink);color:var(--bg);padding:8px 12px;border-radius:4px;
  font-size:11px;font-weight:400;line-height:1.4;letter-spacing:0;
  text-transform:none;white-space:normal;width:260px;z-index:1500;
  box-shadow:0 4px 12px rgba(0,0,0,0.2);
  font-family:'IBM Plex Sans',sans-serif}

/* ===== Modo apresentacao (fullscreen) ===== */
html[data-mode="present"] .toolbar,
html[data-mode="present"] .lang-toggle,
html[data-mode="present"] .tabs,
html[data-mode="present"] .masthead{display:none !important}
html[data-mode="present"] .wrap{max-width:none;padding:24px 48px}
html[data-mode="present"] .tab-pane:not(.active){display:none !important}

/* ===== Versao + changelog ===== */
.version-info{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);margin-top:8px}
.version-info code{background:var(--bg-alt);padding:2px 6px;
  border-radius:2px;color:var(--accent)}
.changelog-toggle{cursor:pointer;color:var(--muted);text-decoration:underline;
  text-decoration-style:dotted;font-size:10px;margin-left:6px}
.changelog-toggle:hover{color:var(--accent)}
.changelog-list{font-size:10px;line-height:1.7;margin:8px 0 0 16px;
  list-style:none;padding:0;color:var(--muted)}
.changelog-list li{margin-bottom:2px;font-family:'IBM Plex Mono',monospace}

/* ===== Credits no footer ===== */
.credits{font-size:10px;color:var(--muted);line-height:1.6;
  margin-top:16px;padding-top:12px;border-top:1px solid var(--border)}
.credits-row{display:flex;gap:16px;flex-wrap:wrap;align-items:center}
.credits-row span{display:inline-flex;align-items:center;gap:4px}

/* ===== Print styles ===== */
@media print{
  .toolbar,.lang-toggle,.tabs{display:none !important}
  .tab-pane:not(.active){display:none !important}
  .wrap{max-width:none;padding:0}
  body{background:white;color:black}
  a{color:black !important;text-decoration:none}
}

/* ===== ONDA 2A: YoY card ===== */
.yoy-card{background:var(--panel);border:1px solid var(--border2);
  border-radius:4px;padding:24px;margin:24px 0}
.yoy-grid{display:grid;grid-template-columns:1fr auto 1fr;gap:32px;
  align-items:center;justify-items:center}
.yoy-cell{text-align:center;display:flex;flex-direction:column;gap:6px;align-items:center}
.yoy-label{font-size:10px;color:var(--muted);letter-spacing:0.14em;
  text-transform:uppercase;font-weight:600}
.yoy-value{font-family:'Fraunces',Georgia,serif;font-size:38px;
  font-weight:500;color:var(--ink);line-height:1.1}
.yoy-value.neg{color:var(--accent)}
.yoy-na{font-size:18px;color:var(--muted);font-style:italic}
.yoy-note{font-size:11px;color:var(--muted);font-family:'IBM Plex Mono',monospace}
.yoy-arrow .arrow{font-size:42px;line-height:1}
.yoy-arrow .arrow.worse{color:var(--accent-today)}
.yoy-arrow .arrow.better{color:var(--ok)}
.yoy-arrow .arrow.equal{color:var(--muted)}
.yoy-delta{font-size:13px;font-family:'IBM Plex Mono',monospace;
  color:var(--ink);margin-top:4px;font-weight:600}
.yoy-prose{margin-top:18px;font-size:13px;color:var(--ink-2);
  line-height:1.6;border-top:1px solid var(--border);padding-top:14px;text-align:center}
.yoy-tag-worse{background:#fce4e0;color:#a8442f;border-color:#f0c5be}
.yoy-tag-better{background:#dff5e8;color:#2d5a3d;border-color:#bfe5d0}
.yoy-tag-equal{background:var(--bg-alt);color:var(--muted);border-color:var(--border)}
@media (max-width:720px){.yoy-grid{grid-template-columns:1fr;gap:16px}}

/* ===== ONDA 2A: ITM/OTM card ===== */
.itm-card{background:var(--panel);border:1px solid var(--border2);
  border-radius:4px;padding:24px;margin:24px 0}
.itm-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px}
.itm-cell{padding:18px;background:var(--bg-alt);border-radius:3px;
  border-left:3px solid var(--rule)}
.itm-cell.itm-ref{border-left-color:var(--neutral);text-align:center}
.itm-cell.itm-pos{border-left-color:var(--ok)}
.itm-cell.itm-neg{border-left-color:var(--accent)}
.itm-tiny{font-size:10px;color:var(--muted);letter-spacing:0.1em;
  text-transform:uppercase;line-height:1.4}
.itm-ref-val{font-family:'Fraunces',Georgia,serif;font-size:30px;
  color:var(--ink);font-weight:500;margin:6px 0}
.itm-ref-val .unit{font-size:12px;color:var(--muted);
  font-family:'IBM Plex Mono',monospace;margin-left:4px}
.itm-label{display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  margin-bottom:10px;font-size:11px;color:var(--ink);
  letter-spacing:0.1em;text-transform:uppercase;font-weight:600}
.itm-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.itm-dot.pos{background:var(--ok)}
.itm-dot.neg{background:var(--accent)}
.itm-grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:8px}
.itm-num{font-family:'Fraunces',Georgia,serif;font-size:28px;
  color:var(--ink);font-weight:500;line-height:1}
.itm-num .unit{font-size:12px;color:var(--muted);
  font-family:'IBM Plex Mono',monospace;margin-left:2px}
.itm-prose{margin-top:18px;font-size:13px;color:var(--ink-2);
  line-height:1.6;border-top:1px solid var(--border);padding-top:14px}
@media (max-width:720px){.itm-grid{grid-template-columns:1fr;gap:12px}}

/* ===== ONDA 2A: Razao breakdown card ===== */
.razao-card{background:var(--panel);border:1px solid var(--border2);
  border-radius:4px;padding:24px;margin:24px 0}
.razao-grid{display:grid;grid-template-columns:340px 1fr;gap:28px;
  align-items:start}
.razao-chart{display:flex;align-items:center;justify-content:center}
.razao-stats{width:100%;border-collapse:collapse;font-size:13px}
.razao-stats th{font-size:10px;color:var(--muted);letter-spacing:0.12em;
  text-transform:uppercase;font-weight:600;text-align:left;
  padding:8px 8px;border-bottom:1px solid var(--border2)}
.razao-stats th.num{text-align:right}
.razao-stats td{padding:10px 8px;border-bottom:1px solid var(--border);
  color:var(--ink-2)}
.razao-stats td.num{text-align:right;font-family:'IBM Plex Mono',monospace}
.razao-stats tfoot td{border-bottom:none;border-top:2px solid var(--border2);
  padding-top:12px;font-weight:600;color:var(--ink)}
.razao-row.razao-rel{background:linear-gradient(to right, rgba(168,68,47,0.04), transparent)}
.razao-row.razao-cnf{background:linear-gradient(to right, rgba(213,114,85,0.04), transparent)}
.razao-tag{display:inline-block;padding:2px 8px;border-radius:10px;
  font-size:10px;font-weight:600;letter-spacing:0.06em}
.razao-tag.elig{background:#dff5e8;color:#2d5a3d}
.razao-tag.noelig{background:var(--bg-alt);color:var(--muted)}
.razao-prose{margin-top:14px;font-size:13px;color:var(--ink-2);
  line-height:1.6;padding-top:10px;border-top:1px solid var(--border)}
@media (max-width:840px){
  .razao-grid{grid-template-columns:1fr}
}

*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
  font-family:'IBM Plex Sans',Georgia,serif;
  -webkit-font-smoothing:antialiased; line-height:1.5}
.wrap{max-width:1100px;margin:0 auto;padding:80px 32px 96px}

/* Language toggle */
.lang-toggle{position:fixed;top:24px;right:24px;z-index:1000;
  background:var(--panel);border:1px solid var(--border2);
  border-radius:2px;padding:4px;display:flex;gap:0;
  font-family:'IBM Plex Mono',monospace;font-size:11px;
  letter-spacing:0.1em;box-shadow:0 1px 4px rgba(0,0,0,0.06)}
.lang-btn{background:transparent;border:none;cursor:pointer;
  padding:6px 12px;color:var(--muted);font-family:inherit;
  font-size:inherit;letter-spacing:inherit;text-transform:uppercase;
  font-weight:500;border-radius:1px;transition:all 0.15s}
.lang-btn:hover{color:var(--ink-2)}
.lang-btn.active{background:var(--ink);color:var(--bg);font-weight:600}

.masthead{display:flex;align-items:center;justify-content:space-between;
  border-bottom:1px solid var(--ink);padding-bottom:16px;margin-bottom:32px;
  font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--ink-2);letter-spacing:0.18em;text-transform:uppercase}
.masthead .vol{font-weight:600}

.tabs{display:flex;gap:0;border-bottom:1px solid var(--rule);
  margin-bottom:48px;flex-wrap:wrap}
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

/* Trend sparkline strip — visual hierarchy with bigger numbers */
.trend-strip{display:grid;grid-template-columns:repeat(3,1fr);
  gap:24px;margin:0 0 48px;padding:28px 32px;
  background:var(--bg-alt);border:1px solid var(--rule);border-radius:2px}
.trend-cell{display:flex;flex-direction:column;gap:6px;position:relative}
.trend-cell .lbl{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:0.18em;text-transform:uppercase;
  display:flex;align-items:center;gap:8px}
.trend-cell .lbl .arrow{display:inline-block;font-size:14px;line-height:1;
  font-weight:600;letter-spacing:0}
.trend-cell .lbl .arrow.up{color:var(--accent-today)}
.trend-cell .lbl .arrow.down{color:var(--ok)}
.trend-cell .lbl .arrow.flat{color:var(--muted)}
.trend-cell .val{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:42px;line-height:1;letter-spacing:-0.015em;
  font-variation-settings:"opsz" 72}
/* Semantic colour: piorando (up) = red, melhorando (down) = green */
.trend-cell.t-up .val{color:var(--accent-today)}
.trend-cell.t-down .val{color:var(--ok)}
.trend-cell.t-flat .val{color:var(--ink)}
.trend-cell .val .unit{font-family:'IBM Plex Sans',sans-serif;font-size:15px;
  color:var(--muted);font-weight:400;margin-left:5px}
.trend-cell .delta{font-family:'IBM Plex Mono',monospace;font-size:11px;
  letter-spacing:0.04em;color:var(--muted)}
.trend-cell .delta.up{color:var(--accent-today);font-weight:500}
.trend-cell .delta.down{color:var(--ok);font-weight:500}
.trend-cell .delta.flat{color:var(--muted)}
.trend-cell .baseline{font-family:'IBM Plex Mono',monospace;font-size:9px;
  color:var(--muted);opacity:0.7;letter-spacing:0.05em;margin-top:4px;
  border-top:1px dashed var(--rule);padding-top:4px}

/* =========================== BENCHMARK TAB =========================== */
.bench-selector{margin:24px 0 32px;padding:22px 24px;background:var(--bg-alt);
  border:1px solid var(--rule);border-radius:2px}
.bench-selector-head{display:flex;align-items:center;justify-content:space-between;
  margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid var(--rule);
  flex-wrap:wrap;gap:10px}
.bench-selector label{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--ink);letter-spacing:0.16em;text-transform:uppercase;font-weight:500}
.bench-selector-meta{display:flex;gap:8px;flex-wrap:wrap}
.bench-btn-mini{padding:6px 12px;border:1px solid var(--rule);background:var(--panel);
  font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.1em;
  color:var(--ink);cursor:pointer;border-radius:2px;text-transform:uppercase;
  transition:all 0.15s}
.bench-btn-mini:hover{background:var(--bg);border-color:var(--accent)}
.bench-checkboxes{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
  gap:8px 18px;margin:8px 0 14px}
.bench-checkbox-row{display:flex;align-items:center;gap:10px;padding:8px 10px;
  border-radius:2px;cursor:pointer;transition:background 0.12s}
.bench-checkbox-row:hover{background:rgba(168,68,47,0.05)}
.bench-checkbox-row input[type=checkbox]{margin:0;cursor:pointer;
  width:14px;height:14px;accent-color:var(--accent)}
.bench-cb-label{display:flex;flex-direction:column;gap:2px;flex:1}
.bench-cb-label > span:first-child{font-family:'IBM Plex Sans',sans-serif;
  font-size:12px;color:var(--ink);font-weight:500}
.bench-cb-meta{font-family:'IBM Plex Mono',monospace;font-size:9px;
  color:var(--muted);letter-spacing:0.04em}
.bench-cb-swatch{display:inline-block;width:10px;height:10px;border-radius:50%;
  flex-shrink:0}
.bench-fleet-toggle{margin-top:14px;padding-top:12px;
  border-top:1px dashed var(--rule)}
.bench-fleet-row{background:rgba(217,46,15,0.03)}
.bench-fleet-row:hover{background:rgba(217,46,15,0.06)}
.bench-meta{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--muted);letter-spacing:0.04em;font-style:italic;margin:14px 0 0}
.bench-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;
  margin:0 0 24px}
@media(max-width:900px){.bench-cards{grid-template-columns:1fr}}
.bench-card{background:var(--panel);border:1px solid var(--rule);
  border-radius:2px;padding:22px 24px;display:flex;flex-direction:column;gap:14px}
.bench-card-mauriti{border-left:3px solid var(--accent)}
.bench-card-peer{border-left:3px solid var(--neutral)}
.bench-card-diff{border-left:3px solid var(--ink);background:var(--bg-alt)}
.bench-card-label{font-family:'IBM Plex Mono',monospace;font-size:10px;
  font-weight:600;letter-spacing:0.22em;text-transform:uppercase;
  color:var(--ink)}
.bench-card-sub{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:0.04em;margin-top:-8px}
.bench-rows{display:flex;flex-direction:column;gap:10px;margin-top:6px}
.bench-row{display:flex;justify-content:space-between;align-items:baseline;
  border-bottom:1px dotted var(--rule);padding-bottom:8px}
.bench-row:last-child{border-bottom:none;padding-bottom:0}
.bench-row .key{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:0.1em;text-transform:uppercase}
.bench-row .val{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:22px;line-height:1;letter-spacing:-0.01em;color:var(--ink);
  font-variation-settings:"opsz" 72}
.bench-row .val .unit{font-family:'IBM Plex Sans',sans-serif;font-size:11px;
  color:var(--muted);font-weight:400;margin-left:3px}
.bench-row.diff-worse .val{color:var(--accent-today)}
.bench-row.diff-better .val{color:var(--ok)}
.bench-row.diff-neutral .val{color:var(--muted)}

.bench-table-wrap{margin:24px 0 0}
.bench-table-wrap h4{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:17px;margin:0 0 8px;color:var(--ink)}
.bench-table-desc{font-family:'IBM Plex Sans',sans-serif;font-size:13px;
  line-height:1.55;color:var(--muted);margin:0 0 14px;max-width:780px}
.bench-table tbody tr{cursor:pointer;transition:background 0.15s}
.bench-table tbody tr:hover{background:var(--bg-alt)}
.bench-table tbody tr.active-bench{background:rgba(168,68,47,0.08)}
.bench-table tbody tr.is-mauriti{background:rgba(168,68,47,0.04);
  font-weight:500}
.bench-table tbody tr.is-ne-fleet{background:rgba(217,46,15,0.04);
  font-style:italic}
.bench-table tbody tr.is-mauriti td:first-child::before{content:"★";
  color:var(--accent)}
.bench-table tbody tr.is-ne-fleet td:first-child::before{content:"⬢";
  color:var(--accent-today)}
.bench-table tbody tr td:first-child{text-align:center;width:24px;
  padding-right:0}
.trend-banner{margin:-24px 0 32px;padding:14px 20px;border-radius:2px;
  font-family:'IBM Plex Mono',monospace;font-size:11px;
  letter-spacing:0.12em;text-transform:uppercase;font-weight:500;
  display:flex;align-items:center;gap:10px}
.trend-banner.up{background:#fbe8e4;color:#7a2818;
  border-left:3px solid var(--accent-today)}
.trend-banner.down{background:#e3eee5;color:#1f4029;
  border-left:3px solid var(--ok)}
.trend-banner.flat{background:#eeece4;color:var(--ink-2);
  border-left:3px solid var(--neutral)}

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

/* REN 1.030 events table */
.csv-actions{display:flex;gap:12px;align-items:center;
  margin:0 0 24px;flex-wrap:wrap}
.btn-csv{background:var(--ink);color:var(--bg);border:none;
  font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:500;
  letter-spacing:0.18em;text-transform:uppercase;padding:10px 18px;
  border-radius:2px;cursor:pointer;text-decoration:none;
  display:inline-flex;align-items:center;gap:8px;transition:opacity 0.15s}
.btn-csv:hover{opacity:0.85}
.events-summary{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--muted);letter-spacing:0.05em}
.events-table-wrap{background:var(--panel);border:1px solid var(--border);
  border-radius:2px;overflow:auto;max-height:520px}
.events-table{width:100%;border-collapse:collapse;
  font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--ink-2)}
.events-table th{background:var(--bg-alt);position:sticky;top:0;
  text-align:left;padding:10px 14px;border-bottom:1px solid var(--rule);
  font-weight:600;letter-spacing:0.1em;text-transform:uppercase;
  font-size:10px;color:var(--ink);white-space:nowrap}
.events-table td{padding:8px 14px;border-bottom:1px solid var(--border);
  white-space:nowrap}
.events-table tr:hover td{background:var(--bg-alt)}
.events-table .razao{font-weight:600}
.events-table .razao.rel{color:var(--accent)}
.events-table .razao.cnf{color:var(--accent-2)}
.events-table .num{text-align:right;font-variant-numeric:tabular-nums}
.empty-msg{padding:48px 32px;text-align:center;color:var(--muted);
  font-family:'Fraunces',Georgia,serif;font-size:17px;font-style:italic}

/* Modulation: recap line above chart + day-by-day table below */
.mod-recap{margin:0 0 18px;padding:14px 20px;
  background:var(--bg-alt);border-left:3px solid var(--accent-today);
  border-radius:2px;
  font-family:'IBM Plex Sans',sans-serif;font-size:13px;line-height:1.7;
  color:var(--ink-2)}
.mod-recap strong{color:var(--ink);font-weight:500}
.mod-recap .recap-bad{color:var(--accent-today);font-weight:600}
.mod-table-wrap{margin:32px 0 0;padding:24px 28px;
  background:var(--bg-alt);border:1px solid var(--rule);border-radius:2px}
.mod-table-head{margin-bottom:18px}
.mod-table-head h4{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:18px;margin:0 0 6px;color:var(--ink)}
.mod-table-sub{display:block;font-family:'IBM Plex Sans',sans-serif;
  font-size:12px;line-height:1.55;color:var(--muted);max-width:620px}
.mod-table tr.row-mid td{background:rgba(212,160,23,0.07)}
.mod-table tr.row-bad td{background:rgba(217,46,15,0.08)}
.mod-table tr.row-bad td.delta{color:var(--accent-today);font-weight:600}
.mod-table tr.row-mid td.delta{color:var(--accent-2);font-weight:600}
.mod-table td.delta{font-variant-numeric:tabular-nums}

/* ============================== */
/* MONTHLY FORECAST (v5.6)        */
/* ============================== */
.forecast-empty{padding:48px;text-align:center;background:var(--bg-alt);
  border:1px solid var(--rule);border-radius:2px;color:var(--muted);
  font-family:'IBM Plex Mono',monospace;font-size:13px}
.forecast-wrap{display:flex;flex-direction:column;gap:24px;margin:24px 0 16px}

/* === Section 1: Projection === */
.forecast-projection{padding:24px 28px;background:var(--bg-alt);
  border:1px solid var(--rule);border-radius:2px}
.forecast-projection-head{display:flex;align-items:center;justify-content:space-between;
  margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid var(--rule);
  flex-wrap:wrap;gap:8px}
.forecast-projection-head h4{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:17px;margin:0;color:var(--ink)}
.forecast-proj-sub{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--muted);letter-spacing:0.06em}
.forecast-proj-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}
@media(max-width:768px){.forecast-proj-grid{grid-template-columns:repeat(2,1fr)}}
.forecast-proj-cell{display:flex;flex-direction:column;gap:6px;padding:8px 0}
.forecast-proj-cell .key{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:0.12em;text-transform:uppercase}
.forecast-proj-cell .val{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:24px;line-height:1;letter-spacing:-0.01em;color:var(--ink)}
.forecast-proj-cell .val .unit{font-family:'IBM Plex Sans',sans-serif;
  font-size:11px;color:var(--muted);font-weight:400;margin-left:4px}
.forecast-proj-cell.forecast-proj-total{padding-left:18px;border-left:3px solid var(--accent)}
.forecast-proj-cell.forecast-proj-total .val{font-size:28px;color:var(--accent)}
.forecast-proj-meta{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);margin-top:2px;letter-spacing:0.04em}

/* === Section 2: Contract setup === */
.forecast-contracts{padding:24px 28px;background:var(--panel);
  border:1px solid var(--rule);border-radius:2px}
.forecast-contracts-head{display:flex;align-items:center;justify-content:space-between;
  margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid var(--rule)}
.forecast-contracts-head h4{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:17px;margin:0;color:var(--ink)}
.forecast-btn-mini{padding:6px 14px;border:1px solid var(--accent);background:var(--panel);
  font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.1em;
  color:var(--accent);cursor:pointer;border-radius:2px;text-transform:uppercase;
  transition:all 0.15s;font-weight:500}
.forecast-btn-mini:hover{background:var(--accent);color:var(--panel)}
.forecast-btn-mini:disabled{opacity:0.4;cursor:not-allowed}
.forecast-btn-mini:disabled:hover{background:var(--panel);color:var(--accent)}
.forecast-contract{padding:16px 0;border-bottom:1px dashed var(--rule)}
.forecast-contract:last-of-type{border-bottom:none}
.forecast-contract-head{display:flex;align-items:center;gap:10px;margin-bottom:12px}
.forecast-contract-tag{display:inline-block;padding:3px 9px;background:var(--ink);
  color:var(--panel);font-family:'IBM Plex Mono',monospace;font-size:10px;
  font-weight:600;letter-spacing:0.1em;border-radius:2px}
.forecast-contract-label{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--ink);letter-spacing:0.06em;font-weight:500}
.forecast-btn-remove{margin-left:auto;padding:4px 10px;border:1px solid var(--rule);
  background:transparent;font-family:'IBM Plex Mono',monospace;font-size:9px;
  letter-spacing:0.1em;color:var(--muted);cursor:pointer;border-radius:2px;
  text-transform:uppercase}
.forecast-btn-remove:hover{color:var(--accent-today);border-color:var(--accent-today)}
.forecast-contract-grid{display:grid;grid-template-columns:2fr 1fr 1fr;gap:18px;
  align-items:start}
@media(max-width:768px){.forecast-contract-grid{grid-template-columns:1fr}}
.forecast-input-group label{display:block;font-family:'IBM Plex Mono',monospace;
  font-size:10px;color:var(--muted);letter-spacing:0.1em;
  text-transform:uppercase;margin-bottom:6px}
.forecast-input-row{display:flex;gap:10px;align-items:center}
.forecast-input-row input[type="range"]{flex:1;-webkit-appearance:none;
  height:4px;background:var(--rule);outline:none;border-radius:2px}
.forecast-input-row input[type="range"]::-webkit-slider-thumb{
  -webkit-appearance:none;width:16px;height:16px;background:var(--accent);
  border-radius:50%;cursor:pointer;border:2px solid var(--panel)}
.forecast-input-row input[type="range"]::-moz-range-thumb{
  width:16px;height:16px;background:var(--accent);border-radius:50%;
  cursor:pointer;border:2px solid var(--panel)}
.forecast-input-row input[type="number"]{width:96px;padding:7px 10px;
  border:1px solid var(--rule);background:var(--panel);
  font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--ink);
  border-radius:2px;text-align:right}
.forecast-input-row input[type="number"]:focus{outline:none;border-color:var(--accent)}
.forecast-select{width:100%;padding:8px 10px;border:1px solid var(--rule);
  background:var(--panel);font-family:'IBM Plex Mono',monospace;font-size:12px;
  color:var(--ink);border-radius:2px;cursor:pointer}
.forecast-select:focus{outline:none;border-color:var(--accent)}
.forecast-hint{font-family:'IBM Plex Mono',monospace;font-size:9px;
  color:var(--muted);letter-spacing:0.04em;margin-top:4px;font-style:italic}
.forecast-portfolio-summary{margin-top:16px;padding:12px 16px;
  background:var(--bg-alt);border:1px solid var(--rule);border-radius:2px;
  font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--ink);
  letter-spacing:0.04em;display:flex;justify-content:space-between;flex-wrap:wrap;
  gap:8px}
.forecast-portfolio-summary strong{font-weight:600;color:var(--accent)}

/* === Section 3: PLD reference === */
.forecast-pld-ref{padding:20px 24px;background:var(--bg-alt);
  border:1px solid var(--rule);border-radius:2px}
.forecast-pld-head{display:flex;align-items:baseline;gap:12px;margin-bottom:14px;
  flex-wrap:wrap}
.forecast-pld-head h4{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:15px;margin:0;color:var(--ink)}
.forecast-pld-sub{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:0.04em;font-style:italic}
.forecast-pld-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));
  gap:10px;margin-bottom:14px}
.forecast-pld-cell{padding:10px 12px;background:var(--panel);border:1px solid var(--rule);
  border-radius:2px;text-align:center}
.forecast-pld-cell .sub{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px}
.forecast-pld-cell .val{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:18px;color:var(--ink);line-height:1}
.forecast-pld-cell .val .unit{font-family:'IBM Plex Sans',sans-serif;
  font-size:9px;color:var(--muted);font-weight:400;margin-left:2px}
.forecast-pld-meta{padding:10px 0 0;border-top:1px dashed var(--rule);
  font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--ink);
  letter-spacing:0.04em}
.forecast-pld-meta strong{color:var(--accent);font-weight:600}
.forecast-pld-extra{color:var(--muted);font-size:10px;margin-left:6px;
  font-style:italic}

/* === Section 4: 3 result cards === */
.forecast-cards-row{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
@media(max-width:900px){.forecast-cards-row{grid-template-columns:1fr}}
.forecast-card{padding:22px 24px;background:var(--panel);border:1px solid var(--rule);
  border-radius:2px;display:flex;flex-direction:column;gap:12px}
.forecast-card-ccee{border-left:3px solid var(--neutral)}
.forecast-card-comm{border-left:3px solid var(--accent)}
.forecast-card-final{border-left:3px solid var(--ink);background:var(--bg-alt);
  padding:26px 28px}
.forecast-card-label{font-family:'IBM Plex Mono',monospace;font-size:10px;
  font-weight:600;letter-spacing:0.22em;text-transform:uppercase;color:var(--ink)}
.forecast-card-sub{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:0.04em;margin-top:-6px}
.forecast-card-detail{display:flex;flex-direction:column;gap:8px;margin-top:4px}
.forecast-card-row{display:flex;justify-content:space-between;align-items:baseline;
  font-family:'IBM Plex Mono',monospace;font-size:11px}
.forecast-card-row .label{color:var(--muted);letter-spacing:0.04em}
.forecast-card-row .value{color:var(--ink);font-weight:500;font-variant-numeric:tabular-nums}
.forecast-card-row.is-neg .value{color:var(--accent-today)}
.forecast-card-row.is-pos .value{color:var(--ok)}
.forecast-card-total{display:flex;justify-content:space-between;align-items:baseline;
  padding-top:12px;border-top:1px solid var(--rule);margin-top:auto}
.forecast-card-total .key{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--ink);letter-spacing:0.1em;text-transform:uppercase;font-weight:600}
.forecast-card-total .val{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:22px;color:var(--ink);line-height:1}
.forecast-card-final .forecast-final-figure{font-family:'Fraunces',Georgia,serif;
  font-weight:500;font-size:42px;line-height:1;letter-spacing:-0.02em;
  color:var(--ink);font-variation-settings:"opsz" 72;margin:8px 0}
.forecast-card-final.is-pos .forecast-final-figure{color:var(--ok)}
.forecast-card-final.is-neg .forecast-final-figure{color:var(--accent-today)}
.forecast-final-meta{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:0.04em;line-height:1.5}

/* === Section 5: Alert === */
.forecast-alert{display:flex;gap:14px;padding:16px 20px;
  background:rgba(217,46,15,0.08);border:1px solid var(--accent-today);
  border-radius:2px;align-items:flex-start}
.forecast-alert-icon{font-size:20px;color:var(--accent-today);line-height:1}
.forecast-alert-body strong{display:block;font-family:'IBM Plex Mono',monospace;
  font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
  color:var(--accent-today);margin-bottom:6px}
.forecast-alert-body p{font-family:'IBM Plex Sans',sans-serif;font-size:13px;
  color:var(--ink);margin:0;line-height:1.5}

/* === Section 6: Sensitivity strip === */
.forecast-sensitivity{padding:22px 24px;background:var(--bg-alt);
  border:1px solid var(--rule);border-radius:2px}
.forecast-sensitivity h4{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:15px;margin:0 0 8px;color:var(--ink)}
.forecast-sens-desc{font-family:'IBM Plex Sans',sans-serif;font-size:12px;
  color:var(--muted);line-height:1.55;margin:0 0 16px;max-width:760px}
.forecast-sens-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:768px){.forecast-sens-grid{grid-template-columns:1fr}}
.forecast-sens-cell{padding:14px 16px;background:var(--panel);
  border:1px solid var(--rule);border-radius:2px;text-align:center}
.forecast-sens-cell.is-base{border-color:var(--accent);background:rgba(168,68,47,0.04)}
.forecast-sens-cell .scenario{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px}
.forecast-sens-cell.is-base .scenario{color:var(--accent);font-weight:600}
.forecast-sens-cell .pld-future{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--muted);margin-bottom:6px}
.forecast-sens-cell .pld-future strong{color:var(--ink)}
.forecast-sens-cell .result{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:22px;color:var(--ink);line-height:1}
.forecast-sens-cell .delta{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:0.04em;margin-top:4px}
.forecast-sens-cell .delta.is-up{color:var(--ok)}
.forecast-sens-cell .delta.is-down{color:var(--accent-today)}

/* === Section 7: Risk decomposition === */
.forecast-decomp{padding:22px 24px;background:var(--panel);
  border:1px solid var(--rule);border-radius:2px}
.forecast-decomp h4{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:15px;margin:0 0 8px;color:var(--ink)}
.forecast-decomp-desc{font-family:'IBM Plex Sans',sans-serif;font-size:12px;
  color:var(--muted);line-height:1.55;margin:0 0 16px;max-width:760px}
.forecast-decomp-grid{display:flex;flex-direction:column;gap:8px}
.forecast-decomp-row{display:grid;grid-template-columns:200px 1fr 100px;
  gap:14px;align-items:center;padding:10px 14px;background:var(--bg-alt);
  border-radius:2px;font-family:'IBM Plex Mono',monospace;font-size:11px}
@media(max-width:768px){.forecast-decomp-row{grid-template-columns:1fr}}
.forecast-decomp-row .label{color:var(--ink);letter-spacing:0.04em}
.forecast-decomp-row .label small{display:block;color:var(--muted);font-size:9px;
  font-style:italic;margin-top:2px}
.forecast-decomp-row .bar-wrap{height:10px;background:var(--rule);border-radius:2px;
  overflow:hidden;position:relative}
.forecast-decomp-row .bar{height:100%;border-radius:2px}
.forecast-decomp-row .bar.is-pos{background:var(--ok)}
.forecast-decomp-row .bar.is-neg{background:var(--accent-today)}
.forecast-decomp-row .value{text-align:right;font-weight:500;font-variant-numeric:tabular-nums}
.forecast-decomp-row .value.is-pos{color:var(--ok)}
.forecast-decomp-row .value.is-neg{color:var(--accent-today)}
.forecast-decomp-row.is-total{background:var(--ink);color:var(--panel);font-weight:600;
  margin-top:8px}
.forecast-decomp-row.is-total .label,
.forecast-decomp-row.is-total .value{color:var(--panel)}

/* === Section 8: History tracking === */
.forecast-history{padding:18px 22px;background:var(--bg-alt);
  border:1px solid var(--rule);border-radius:2px}
.forecast-history h4{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:15px;margin:0 0 12px;color:var(--ink)}
.forecast-hist-empty{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--muted);font-style:italic;letter-spacing:0.04em}
.forecast-hist-row{display:flex;justify-content:space-between;align-items:baseline;
  padding:8px 0;border-bottom:1px dotted var(--rule);
  font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--ink)}
.forecast-hist-row:last-child{border-bottom:none;font-weight:600}
.forecast-hist-row .label{color:var(--muted);letter-spacing:0.04em}
.forecast-hist-row .val{font-variant-numeric:tabular-nums}
.forecast-hist-row .delta{font-family:'IBM Plex Mono',monospace;font-size:10px;
  margin-left:8px}
.forecast-hist-row .delta.is-up{color:var(--ok)}
.forecast-hist-row .delta.is-down{color:var(--accent-today)}

.forecast-disclaimer{margin:8px 0 0;padding:14px 18px;background:transparent;
  border-top:1px dashed var(--rule);font-family:'IBM Plex Sans',sans-serif;
  font-size:11px;color:var(--muted);line-height:1.6;font-style:italic;max-width:780px}


footer{margin-top:96px;padding-top:32px;border-top:1px solid var(--ink);
  font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);
  line-height:1.8;letter-spacing:0.04em}
footer p{margin:0 0 10px;max-width:680px}
footer .colofao{font-family:'Fraunces',Georgia,serif;font-style:italic;
  font-size:13px;color:var(--ink-2);margin-top:24px}

/* Glossary collapsible */
.glossary{margin-top:32px;padding:20px 24px;background:var(--bg-alt);
  border:1px solid var(--rule);border-radius:2px}
.glossary summary{cursor:pointer;font-family:'IBM Plex Mono',monospace;
  font-size:11px;letter-spacing:0.18em;text-transform:uppercase;
  color:var(--ink);font-weight:600;outline:none}
.glossary[open] summary{margin-bottom:16px;
  border-bottom:1px solid var(--rule);padding-bottom:10px}
.glossary dl{margin:0;display:grid;grid-template-columns:120px 1fr;
  gap:8px 20px;font-size:12px;line-height:1.5}
.glossary dt{font-family:'IBM Plex Mono',monospace;font-weight:600;
  color:var(--accent-2);letter-spacing:0.05em}
.glossary dd{margin:0;color:var(--ink-2)}

.tab-pane{display:none}
.tab-pane.active{display:block}

/* Hide elements based on language */
[data-i18n][data-lang-show]{display:none}
html[data-lang="en"] [data-lang-show="en"]{display:initial}
html[data-lang="pt"] [data-lang-show="pt"]{display:initial}
</style>
</head>
<body>

<!-- Toolbar superior (Print, Dark mode, Fullscreen, Lang) -->
<div class="toolbar">
  <button class="toolbar-btn" id="tb-print" title="Print / Save PDF (P)" aria-label="Print">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round">
      <polyline points="6 9 6 2 18 2 18 9"></polyline>
      <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
      <rect x="6" y="14" width="12" height="8"></rect>
    </svg>
  </button>
  <button class="toolbar-btn" id="tb-theme" title="Dark / Light mode (D)" aria-label="Theme">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
    </svg>
  </button>
  <button class="toolbar-btn" id="tb-fullscreen" title="Presentation mode (F)" aria-label="Fullscreen">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round">
      <path d="M8 3H5a2 2 0 0 0-2 2v3"></path>
      <path d="M21 8V5a2 2 0 0 0-2-2h-3"></path>
      <path d="M3 16v3a2 2 0 0 0 2 2h3"></path>
      <path d="M16 21h3a2 2 0 0 0 2-2v-3"></path>
    </svg>
  </button>
  <span class="toolbar-divider"></span>
  <button class="toolbar-btn lang-btn active" data-set-lang="en">EN</button>
  <button class="toolbar-btn lang-btn" data-set-lang="pt">PT</button>
</div>

<!-- (lang-toggle antigo desativado; agora dentro da toolbar) -->
<div class="lang-toggle" style="display:none">
  <button class="lang-btn active" data-set-lang="en">EN</button>
  <button class="lang-btn" data-set-lang="pt">PT</button>
</div>

<div class="wrap">

  <div class="masthead">
    <div class="vol" data-i18n="volume">Mauriti Report — N&deg; 01</div>
    <div>
      {{ periodo }} &nbsp;&middot;&nbsp;
      <span data-i18n="updated">Updated</span>
      {{ gerado_em }}
    </div>
    {% if freshness %}
    <div class="freshness">
      <span class="freshness-item">
        <span class="freshness-dot {{ freshness.ons_status }}"></span>
        <span data-i18n="fresh_ons">Curtailment ONS</span>:&nbsp;<strong>{{ freshness.ons_last }}</strong>
      </span>
      <span class="freshness-item">
        <span class="freshness-dot {{ freshness.pld_status }}"></span>
        <span data-i18n="fresh_pld">PLD CCEE</span>:&nbsp;<strong>{{ freshness.pld_last }}</strong>
      </span>
      <span class="freshness-item">
        <span class="freshness-dot ok"></span>
        <span data-i18n="fresh_next">Next update</span>:&nbsp;<strong>{{ freshness.next_run }}</strong>
      </span>
    </div>
    {% endif %}
  </div>

  <div class="tabs">
    <button class="tab active" data-tab="curt" data-i18n="tab_curt">Curtailment</button>
    <button class="tab" data-tab="mod" data-i18n="tab_mod">Modulation effect</button>
    <button class="tab" data-tab="ren" data-i18n="tab_ren">REN 1.030 tracker</button>
    <button class="tab" data-tab="solar" data-i18n="tab_solar">Solar resource</button>
    <button class="tab" data-tab="bench" data-i18n="tab_bench">Benchmark</button>
  </div>

  <!-- ============================================================ -->
  <!-- TAB: CURTAILMENT                                              -->
  <!-- ============================================================ -->
  <div class="tab-pane active" data-tab="curt">

    <div class="hero">
      <div class="kicker" data-i18n="curt_kicker">PowerChina &middot; Mauriti Solar Complex, Cear&aacute;</div>
      <h1>
        <span data-i18n="curt_h1_a">How much each curtailed</span>
        <em data-i18n="curt_h1_b">megawatt-hour</em>
        <span data-i18n="curt_h1_c">cost Mauriti.</span>
      </h1>
      <p class="lede" data-i18n="curt_lede">
        Constrained-off study for the <strong>Mauriti Solar Complex</strong>
        over {{ periodo }}, benchmarked against wind and solar plants in
        Cear&aacute;, broken down by curtailment reason (REL/CNF/ENE/PAR)
        and translated into lost revenue at the hourly PLD spot price.
      </p>
      <div class="byline">
        <span data-i18n="byline_submarket">Submarket</span> <span>{{ submercado }}</span> &nbsp;&middot;&nbsp;
        <span data-i18n="byline_units">Mauriti units</span> <span>{{ met_m.n_usinas }}</span> &nbsp;&middot;&nbsp;
        <span data-i18n="byline_bench">Benchmark</span> <span>{{ n_grupos }} <span data-i18n="byline_groups">peers</span></span> &nbsp;&middot;&nbsp;
        <span data-i18n="byline_asof">As of</span> <span>{{ gerado_em }}</span>
      </div>
    </div>

    {% if not trend.vazio %}
    {% set tcls = 'up' if trend.tendencia == 'piorando' else ('down' if trend.tendencia == 'melhorando' else 'flat') %}
    <div class="trend-banner {{ tcls }}">
      <span>
        {% if trend.tendencia == 'piorando' %}
          <span data-i18n="trend_worsening">↑ Trend worsening</span>
        {% elif trend.tendencia == 'melhorando' %}
          <span data-i18n="trend_improving">↓ Trend improving</span>
        {% else %}
          <span data-i18n="trend_stable">→ Trend stable</span>
        {% endif %}
      </span>
      <span style="opacity:0.7;font-weight:400">
        Δ {{ "%+.2f"|format(trend.delta_30_vs_365) }} pp
        <span data-i18n="trend_window_label">(30d vs 365d)</span>
      </span>
    </div>
    <div class="trend-strip">
      {# Decide colour direction: piorando (worse) = up/red, melhorando (better) = down/green #}
      {% set tdir = 'up' if trend.tendencia == 'piorando' else ('down' if trend.tendencia == 'melhorando' else 'flat') %}
      {% set tarrow = '↗' if tdir == 'up' else ('↘' if tdir == 'down' else '→') %}
      <div class="trend-cell t-{{ tdir }}">
        <div class="lbl">
          <span data-i18n="trend_30d">Last 30 days</span>
          <span class="arrow {{ tdir }}">{{ tarrow }}</span>
        </div>
        <div class="val">{{ "%.2f"|format(trend.cf_d30) }}<span class="unit">%</span></div>
        <div class="delta">CF · {{ "%.0f"|format(trend.mwh_d30/1000) }} GWh <span data-i18n="trend_curtailed">curtailed</span></div>
        <div class="baseline">365d baseline: {{ "%.2f"|format(trend.cf_d365) }}%</div>
      </div>
      <div class="trend-cell t-flat">
        <div class="lbl">
          <span data-i18n="trend_90d">Last 90 days</span>
        </div>
        <div class="val">{{ "%.2f"|format(trend.cf_d90) }}<span class="unit">%</span></div>
        <div class="delta">CF · {{ "%.0f"|format(trend.mwh_d90/1000) }} GWh <span data-i18n="trend_curtailed">curtailed</span></div>
        <div class="baseline">mid-term window</div>
      </div>
      <div class="trend-cell t-flat">
        <div class="lbl">
          <span data-i18n="trend_365d">Last 365 days</span>
        </div>
        <div class="val">{{ "%.2f"|format(trend.cf_d365) }}<span class="unit">%</span></div>
        <div class="delta">CF · {{ "%.0f"|format(trend.mwh_d365/1000) }} GWh <span data-i18n="trend_curtailed">curtailed</span></div>
        <div class="baseline">structural baseline</div>
      </div>
    </div>
    {% endif %}

    <div class="tracker">
      <div class="liveflag">&bull; <span data-i18n="weekly_update">Weekly update</span></div>
      <h2 data-i18n="tracker_title">Current month tracker</h2>
      <div class="when">
        <span data-i18n="period">Period</span>: {{ tracker.cur_first }} &middot;
        {{ tracker.dias_decorridos }} <span data-i18n="of">of</span>
        {{ tracker.days_in_month }} <span data-i18n="days_elapsed">days elapsed</span>
      </div>

      <div class="tracker-stats">
        <div class="t-stat">
          <div class="lbl">
            <span data-i18n="expected_month">Expected (month)</span>
            <span class="kpi-tip" data-tip="Expected MWh = installed capacity × daytime hours × theoretical CF. Reference for what Mauriti would have generated without curtailment."></span>
          </div>
          <div class="val">{{ "%.1f"|format(tracker.esperada_mes) }}<span class="unit">MWh</span></div>
          <div class="delta">{{ tracker.dias_decorridos }} <span data-i18n="days_short">days</span></div>
        </div>
        <div class="t-stat">
          <div class="lbl">
            <span data-i18n="cut_month">Curtailed (month)</span>
            <span class="kpi-tip" data-tip="MWh curtailed = expected − actual generation. Includes all ONS reasons (REL/CNF/ENE/PAR)."></span>
          </div>
          <div class="val">{{ "%.1f"|format(tracker.cortada_mes) }}<span class="unit">MWh</span></div>
          <div class="delta"><span data-i18n="of_lower">of</span> {{ "%.1f"|format(tracker.esperada_mes) }} <span data-i18n="expected_lower">expected</span></div>
        </div>
        <div class="t-stat alt">
          <div class="lbl">
            <span data-i18n="cf_month">CF% of month</span>
            <span class="kpi-tip" data-tip="Curtailment Factor (CF%) = curtailed MWh ÷ expected MWh × 100. Lower is better. Compared to 22.7% prior-quarter Mauriti average."></span>
          </div>
          <div class="val">{{ "%.2f"|format(tracker.cf_mes) }}<span class="unit">%</span></div>
          {% if tracker.delta_cf is not none %}
          <div class="delta {% if tracker.delta_cf > 0 %}up{% else %}down{% endif %}">
            {% if tracker.delta_cf > 0 %}+{% endif %}{{ "%.2f"|format(tracker.delta_cf) }} pp <span data-i18n="vs_quarter">vs quarter</span>
          </div>
          {% endif %}
        </div>
        <div class="t-stat">
          <div class="lbl" data-i18n="quarter_avg">Quarter avg CF</div>
          <div class="val">{% if tracker.ref_cf is not none %}{{ "%.2f"|format(tracker.ref_cf) }}{% else %}—{% endif %}<span class="unit">%</span></div>
          <div class="delta" data-i18n="ref_90d">90-day baseline</div>
        </div>
        <div class="t-stat">
          <div class="lbl" data-i18n="worst_day">Worst day (CF%)</div>
          <div class="val">{{ "%.1f"|format(tracker.pior_cf) }}<span class="unit">%</span></div>
          <div class="delta"><span data-i18n="day_short">day</span> {{ tracker.pior_dia }}</div>
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
        <h2 data-i18n="curt_bignum_title">Energy not delivered due to operating restrictions</h2>
        <p data-i18n="curt_bignum_p1">
          Equivalent to <strong>R$ {{ "%.1f"|format(met_m.receita_perdida/1e6) }}
          million</strong> in estimated lost revenue at the hourly PLD price
          of submarket {{ submercado }}, with a curtailment factor of
          <strong>{{ "%.2f"|format(met_m.curtailment_factor) }}%</strong>.
        </p>
        <p data-i18n="curt_bignum_p2">
          Of this total, <strong>{{ "%.1f"|format(met_m.pct_ressarcivel) }}%
          potentially recoverable</strong> under ANEEL Resolution
          1.030/2022 (REL and CNF reasons).
        </p>
      </div>
    </div>

    <div class="stats">
      <div class="stat alt">
        <div class="lbl" data-i18n="kpi_cf">Curtailment Factor</div>
        <div class="val">{{ "%.2f"|format(met_m.curtailment_factor) }}<span class="unit">%</span></div>
        <div class="delta" data-i18n="kpi_cf_sub">cut / reference</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="kpi_lost_rev">Lost revenue</div>
        <div class="val">R$ {{ "%.1f"|format(met_m.receita_perdida/1e6) }}<span class="unit">M</span></div>
        <div class="delta">@ PLD {{ submercado }}</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="kpi_recover">Pot. recoverable</div>
        <div class="val">R$ {{ "%.1f"|format(met_m.receita_ressarcivel/1e6) }}<span class="unit">M</span></div>
        <div class="delta">REL + CNF</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="kpi_pld_cut">PLD during cuts</div>
        <div class="val">{{ "%.0f"|format(met_m.pld_durante_corte) }}<span class="unit">R$/MWh</span></div>
        <div class="delta"><span data-i18n="vs_avg">vs avg</span> R$ {{ "%.0f"|format(met_m.pld_geral) }}/MWh</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="kpi_vs_bench">vs Benchmark CE</div>
        {% set delta = met_m.curtailment_factor - met_b.curtailment_factor %}
        <div class="val">{% if delta > 0 %}+{% endif %}{{ "%.2f"|format(delta) }}<span class="unit">pp</span></div>
        <div class="delta" data-i18n="kpi_vs_bench_sub">CF delta</div>
      </div>
    </div>

    <div class="section-head">
      <span class="num">I.</span><h3 data-i18n="curt_s1_title">The cadence of cuts</h3>
      <span class="tag" data-i18n="curt_s1_tag">TIME SERIES</span>
    </div>
    <p class="section-desc" data-i18n="curt_s1_desc">
      The area between the green dotted reference line and the realised
      generation represents energy that could have been produced and was not.
    </p>
    <div class="chart"><div id="serie" style="height:440px"></div></div>

    <div class="section-head">
      <span class="num">II.</span><h3 data-i18n="curt_s2_title">Why we are cut</h3>
      <span class="tag" data-i18n="curt_s2_tag">RESTRICTION REASONS</span>
    </div>
    <p class="section-desc" data-i18n="curt_s2_desc">
      External unavailability (REL) and Reliability (CNF) are typically
      eligible for recovery under REN 1.030/2022; energy reason (ENE)
      is rarely so.
    </p>
    <div class="chart-row">
      <div class="chart"><div id="donut_razao" style="height:380px"></div></div>
      <div class="chart"><div id="razoes_mes" style="height:380px"></div></div>
    </div>

    {% if insights_m %}
    <div class="pullquote">{{ insights_m|safe }}<cite data-i18n="auto_read">Automatic reading</cite></div>
    {% endif %}

    <div class="section-head">
      <span class="num">III.</span><h3 data-i18n="curt_s3_title">Mauriti vs CE peers</h3>
      <span class="tag" data-i18n="curt_s3_tag">FIXED BENCHMARK</span>
    </div>
    <p class="section-desc">
      <span data-i18n="curt_s3_desc">Direct comparison with</span> {{ n_grupos }}
      <span data-i18n="curt_s3_desc2">groups</span>: <strong>{{ grupos_str }}</strong>.
    </p>
    <div class="chart"><div id="comp_cf" style="height:480px"></div></div>

    {% if insights_c %}
    <div class="pullquote">{{ insights_c|safe }}<cite data-i18n="comp_analysis">Comparative analysis</cite></div>
    {% endif %}

    <div class="section-head">
      <span class="num">IV.</span><h3 data-i18n="curt_s4_title">Where in the day cuts happen</h3>
      <span class="tag" data-i18n="curt_s4_tag">HOURLY PATTERN</span>
    </div>
    <p class="section-desc" data-i18n="curt_s4_desc">
      CF% heatmap by hour of day. Cuts at 11h–14h indicate systemic NE
      restriction; scattered patterns suggest local limitation.
    </p>
    <div class="chart"><div id="heatmap" style="height:440px"></div></div>

    <div class="section-head">
      <span class="num">V.</span><h3 data-i18n="curt_s5_title">Weekly pattern</h3>
      <span class="tag" data-i18n="curt_s5_tag">WEEKDAY × HOUR</span>
    </div>
    <p class="section-desc" data-i18n="curt_s5_desc">
      Same heatmap, aggregated by weekday × hour. Reveals if curtailment is
      a structural weekly phenomenon (e.g. weekends with lower industrial load
      mean more solar surplus and more cuts).
    </p>
    <div class="chart"><div id="heatmap_dow" style="height:460px"></div></div>

    <!-- ========================================================= -->
    <!-- ONDA 2A.3: Breakdown REL/CNF/ENE/PAR                       -->
    <!-- ========================================================= -->
    {% if razao_breakdown %}
    <div class="section-head">
      <span class="num">IX.</span>
      <h3 data-i18n="razao_title">Curtailment by reason · ONS classification</h3>
      <span class="tag" data-i18n="razao_tag">REN 1.030 ELIGIBILITY</span>
    </div>
    <p class="section-desc" data-i18n="razao_desc">
      Each curtailment hour is classified by ONS with a reason code. REL
      (external unavailability) and CNF (reliability) are eligible for
      compensation under REN 1.030/2022; ENE (energy/oversupply) and PAR
      (access opinion) are not. Knowing the mix is critical for ressarcement
      strategy.
    </p>

    <div class="razao-card">
      <div class="razao-grid">
        <div class="razao-chart">
          <div class="chart"><div id="razao_donut" style="height:340px"></div></div>
        </div>
        <div class="razao-table">
          <table class="razao-stats">
            <thead>
              <tr>
                <th data-i18n="razao_th_code">Code</th>
                <th data-i18n="razao_th_meaning">Meaning</th>
                <th class="num" data-i18n="razao_th_mwh">MWh</th>
                <th class="num" data-i18n="razao_th_loss">Loss (R$)</th>
                <th class="num" data-i18n="razao_th_pct">% MWh</th>
                <th data-i18n="razao_th_elig">Eligible</th>
              </tr>
            </thead>
            <tbody>
              {% for r in razao_breakdown %}
              <tr class="razao-row razao-{{ r.razao|lower }}">
                <td><strong>{{ r.razao }}</strong></td>
                <td>{{ r.label }}</td>
                <td class="num">{{ "{:,.0f}".format(r.mwh_total).replace(",", " ") }}</td>
                <td class="num">R$ {{ "%.1f"|format(r.perd_rs/1e6) }}M</td>
                <td class="num">{{ "%.1f"|format(r.pct_mwh) }}%</td>
                <td>
                  {% if r.ressarcivel %}
                    <span class="razao-tag elig" data-i18n="razao_yes">Yes</span>
                  {% else %}
                    <span class="razao-tag noelig" data-i18n="razao_no">No</span>
                  {% endif %}
                </td>
              </tr>
              {% endfor %}
            </tbody>
            <tfoot>
              <tr class="razao-foot">
                {% set tot_mwh = razao_breakdown | sum(attribute='mwh_total') %}
                {% set tot_rs  = razao_breakdown | sum(attribute='perd_rs') %}
                {% set ress_mwh = razao_breakdown | selectattr('ressarcivel') | sum(attribute='mwh_total') %}
                {% set ress_pct = (100 * ress_mwh / tot_mwh) if tot_mwh > 0 else 0 %}
                <td colspan="2"><strong data-i18n="razao_total">Total</strong></td>
                <td class="num"><strong>{{ "{:,.0f}".format(tot_mwh).replace(",", " ") }}</strong></td>
                <td class="num"><strong>R$ {{ "%.1f"|format(tot_rs/1e6) }}M</strong></td>
                <td class="num"></td>
                <td>
                  <span class="razao-tag elig">{{ "%.0f"|format(ress_pct) }}%</span>
                </td>
              </tr>
            </tfoot>
          </table>
          <p class="razao-prose">
            {% if ress_pct > 70 %}
              <strong data-i18n="razao_prose_high">{{ "%.0f"|format(ress_pct) }}% of curtailed MWh is eligible for compensation</strong>
              <span data-i18n="razao_prose_high_p"> under REN 1.030/2022. Strong case for ressarcement claims.</span>
            {% elif ress_pct > 40 %}
              <strong>{{ "%.0f"|format(ress_pct) }}%</strong>
              <span data-i18n="razao_prose_mid">of curtailed MWh is potentially eligible for ressarcement. Mixed profile — review case by case.</span>
            {% else %}
              <span data-i18n="razao_prose_low">Most curtailment is ENE/PAR (not eligible for ressarcement). Curtailment driven mainly by oversupply, not by external network failures.</span>
            {% endif %}
          </p>
        </div>
      </div>
    </div>
    {% endif %}

  </div><!-- /tab curt -->

  <!-- ============================================================ -->
  <!-- TAB: MODULATION                                               -->
  <!-- ============================================================ -->
  <div class="tab-pane" data-tab="mod">

    {% if pld_fallback %}
    <div style="background:#fff7e0;border:2px solid #d4a017;
                padding:24px 28px;margin:0 0 40px;border-radius:2px">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                  color:#a07a00;letter-spacing:0.18em;text-transform:uppercase;
                  font-weight:600;margin-bottom:10px">
        &#9888; <span data-i18n="pld_unavail">PLD unavailable — modulation not computed</span>
      </div>
      <div style="font-family:'Fraunces',Georgia,serif;font-size:17px;
                  line-height:1.5;color:#3d3833;margin-bottom:14px"
           data-i18n="pld_unavail_p">
        Could not retrieve hourly PLD from CCEE in this run (CCEE blocks
        cloud provider IPs such as GitHub Actions). Charts below use
        R$ 200/MWh as placeholder, which artificially zeros the modulation
        discount.
      </div>
    </div>
    {% endif %}

    <div class="hero">
      <div class="kicker" data-i18n="mod_kicker">PowerChina &middot; Mauriti &middot; Profile value</div>
      <h1>
        <span data-i18n="mod_h1_a">How much each MWh cost us</span>
        <em data-i18n="mod_h1_b">when</em>
        <span data-i18n="mod_h1_c">we generated it.</span>
      </h1>
      <p class="lede" data-i18n="mod_lede">
        Comparison between Mauriti's actual revenue
        (Σ MWh<sub>hour</sub> × PLD<sub>hour</sub>) and the "flat"
        revenue that would have been earned with constant generation
        throughout the day. The difference is the
        <strong>modulation discount</strong> — cost of timing,
        not of volume.
      </p>
      <div class="byline">
        <span data-i18n="byline_submarket">Submarket</span> <span>{{ submercado }}</span> &nbsp;&middot;&nbsp;
        <span data-i18n="mod_byline_bench">Benchmark</span> <span data-i18n="ne_solar_fleet">NE solar fleet</span> &nbsp;&middot;&nbsp;
        <span data-i18n="byline_period">Period</span> <span>{{ periodo }}</span>
      </div>
    </div>

    {% if not met_mod_m.vazio %}
    <div class="tracker">
      <div class="liveflag">&bull; <span data-i18n="weekly_update">Weekly update</span></div>
      <h2 data-i18n="tracker_title">Current month tracker</h2>
      <div class="when" data-i18n="mod_tracker_when">Actual revenue vs flat revenue per day</div>

      <div class="tracker-stats">
        <div class="t-stat alt">
          <div class="lbl" data-i18n="mod_disc_month">Discount month</div>
          {% if mod_tracker.cur_pct is not none %}
          <div class="val">{{ "%.2f"|format(mod_tracker.cur_pct) }}<span class="unit">%</span></div>
          {% else %}<div class="val">—</div>{% endif %}
          <div class="delta" data-i18n="mod_curr_month">current month</div>
        </div>
        <div class="t-stat">
          <div class="lbl" data-i18n="mod_disc_month_rs">Discount month (R$)</div>
          {% if mod_tracker.cur_rs is not none %}
          <div class="val">{{ "%.0f"|format(mod_tracker.cur_rs/1000) }}<span class="unit">k R$</span></div>
          {% else %}<div class="val">—</div>{% endif %}
          <div class="delta" data-i18n="mod_pot_lost">potential lost revenue</div>
        </div>
        <div class="t-stat">
          <div class="lbl" data-i18n="mod_pld_avg">Month avg PLD</div>
          {% if mod_tracker.cur_pld is not none %}
          <div class="val">{{ "%.0f"|format(mod_tracker.cur_pld) }}<span class="unit">R$/MWh</span></div>
          {% else %}<div class="val">—</div>{% endif %}
          <div class="delta">{{ submercado }}</div>
        </div>
        <div class="t-stat">
          <div class="lbl" data-i18n="ne_bench">NE benchmark</div>
          {% if met_mod_ne.vazio %}<div class="val">—</div>
          {% else %}
          <div class="val">{{ "%.2f"|format(met_mod_ne.desconto_pct) }}<span class="unit">%</span></div>
          {% endif %}
          <div class="delta" data-i18n="ne_fleet_period">NE solar fleet, period</div>
        </div>
        <div class="t-stat">
          <div class="lbl" data-i18n="vs_bench">vs benchmark</div>
          {% if mod_tracker.delta_pp is not none %}
          <div class="delta {% if mod_tracker.delta_pp < 0 %}up{% else %}down{% endif %}"
               style="font-size:30px;font-family:Fraunces,Georgia,serif;
                      letter-spacing:-0.01em;margin-top:0">
            {% if mod_tracker.delta_pp > 0 %}+{% endif %}{{ "%.2f"|format(mod_tracker.delta_pp) }} pp
          </div>
          {% if mod_tracker.delta_pp < 0 %}
          <div class="delta" data-i18n="mauriti_worse">Mauriti worse</div>
          {% else %}
          <div class="delta" data-i18n="mauriti_better">Mauriti better</div>
          {% endif %}
          {% else %}<div class="val">—</div>{% endif %}
        </div>
      </div>

      {% if not mod_summary.vazio %}
      <div class="mod-recap">
        <span data-i18n="mod_recap_days">{{ mod_summary.n_dias }} days elapsed</span> ·
        <span><strong>{{ "{:,.0f}".format(mod_summary.mwh_total).replace(",", " ") }} MWh</strong> <span data-i18n="mod_recap_gen">generated</span></span> ·
        <span data-i18n="mod_recap_real">Real revenue</span>
        <strong>R$ {{ "%.2f"|format(mod_summary.receita_real/1e6) }}M</strong>
        (<span data-i18n="mod_recap_vs">vs flat</span> R$ {{ "%.2f"|format(mod_summary.receita_flat/1e6) }}M) ·
        <span data-i18n="mod_recap_disc">Discount</span>
        <strong class="recap-bad">{{ "%.2f"|format(mod_summary.desconto_pct) }}%</strong>
        (<strong class="recap-bad">−R$ {{ "%.0f"|format((mod_summary.desconto_rs|abs)/1000) }}k</strong>) ·
        <span data-i18n="mod_recap_worst">Worst day</span>
        <strong>{{ mod_summary.pior_dia }}</strong> ({{ "%.1f"|format(mod_summary.pior_pct) }}%)
      </div>
      {% endif %}

      <div style="background:transparent;padding:0">
        <div id="mod_tracker" style="height:460px"></div>
      </div>

      {% if mod_tabela_dias %}
      <div class="mod-table-wrap">
        <div class="mod-table-head">
          <h4 data-i18n="mod_table_title">Day-by-day price detail · R$/MWh</h4>
          <span class="mod-table-sub" data-i18n="mod_table_sub">
            Compares what Mauriti would receive at the daily PLD average ("flat")
            against what it actually received per MWh sold ("effective"). The gap
            is the modulation cost per MWh.
          </span>
        </div>
        <div class="events-table-wrap">
        <table class="events-table mod-table">
          <thead>
            <tr>
              <th data-i18n="mod_th_day">Day</th>
              <th class="num" data-i18n="mod_th_mwh">MWh generated</th>
              <th class="num" data-i18n="mod_th_pld_flat">PLD flat (R$/MWh)</th>
              <th class="num" data-i18n="mod_th_pld_eff">PLD effective (R$/MWh)</th>
              <th class="num" data-i18n="mod_th_delta_mwh">Δ R$/MWh</th>
              <th class="num" data-i18n="mod_th_delta_pct">Δ %</th>
              <th class="num" data-i18n="mod_th_revenue">Real revenue (R$)</th>
            </tr>
          </thead>
          <tbody>
          {% for r in mod_tabela_dias %}
            {% set bad = r.desconto_pct < -40 %}
            {% set mid = r.desconto_pct < -25 and r.desconto_pct >= -40 %}
            <tr{% if bad %} class="row-bad"{% elif mid %} class="row-mid"{% endif %}>
              <td>{{ r.dia }}</td>
              <td class="num">{{ "{:,.0f}".format(r.mwh).replace(",", " ") }}</td>
              <td class="num">{{ "%.0f"|format(r.pld_medio) }}</td>
              <td class="num">{{ "%.0f"|format(r.pld_efetivo) }}</td>
              <td class="num delta">{{ "%+.0f"|format(r.delta_rs_mwh) }}</td>
              <td class="num delta">{{ "%+.2f"|format(r.desconto_pct) }}%</td>
              <td class="num">{{ "{:,.0f}".format(r.receita_real).replace(",", " ") }}</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
        </div>
      </div>
      {% endif %}
    </div>

    <!-- ========================================================= -->
    <!-- ONDA 2A: YoY Modulation comparison + In/Out-the-money    -->
    <!-- ========================================================= -->
    {% if yoy_modulation and not yoy_modulation.vazio %}
    <div class="section-head">
      <span class="num">IV.</span>
      <h3 data-i18n="yoy_title">Year-over-year modulation</h3>
      <span class="tag yoy-tag-{{ yoy_modulation.status }}" data-i18n="yoy_tag_{{ yoy_modulation.status }}">
        {% if yoy_modulation.status == 'worse' %}WORSE{% elif yoy_modulation.status == 'better' %}BETTER{% else %}STABLE{% endif %}
      </span>
    </div>
    <p class="section-desc" data-i18n="yoy_desc">
      Compares the modulation discount of the current partial month against
      the same partial month in the prior year. Apples-to-apples (same number
      of elapsed days) to isolate seasonal vs. structural changes.
    </p>

    <div class="yoy-card">
      <div class="yoy-grid">
        <div class="yoy-cell">
          <div class="yoy-label" data-i18n="yoy_current">Current ({{ yoy_modulation.label_cur }})</div>
          <div class="yoy-value {% if yoy_modulation.cur_pct and yoy_modulation.cur_pct < 0 %}neg{% endif %}">
            {% if yoy_modulation.cur_pct is not none %}
              {{ "%.2f"|format(yoy_modulation.cur_pct) }}%
            {% else %}
              <span class="yoy-na">N/A</span>
            {% endif %}
          </div>
          <div class="yoy-note">{{ yoy_modulation.cur_n_dias }} <span data-i18n="yoy_days">days</span></div>
        </div>

        <div class="yoy-cell yoy-arrow">
          {% if yoy_modulation.delta_pp is not none %}
            {% if yoy_modulation.status == 'worse' %}<span class="arrow worse">▼</span>
            {% elif yoy_modulation.status == 'better' %}<span class="arrow better">▲</span>
            {% else %}<span class="arrow equal">●</span>{% endif %}
            <div class="yoy-delta">{{ "%+.2f"|format(yoy_modulation.delta_pp) }} pp</div>
          {% else %}
            <span class="arrow equal">●</span>
          {% endif %}
        </div>

        <div class="yoy-cell">
          <div class="yoy-label" data-i18n="yoy_prior">Prior year ({{ yoy_modulation.label_prior }})</div>
          <div class="yoy-value {% if yoy_modulation.prior_pct and yoy_modulation.prior_pct < 0 %}neg{% endif %}">
            {% if yoy_modulation.prior_pct is not none %}
              {{ "%.2f"|format(yoy_modulation.prior_pct) }}%
            {% else %}
              <span class="yoy-na">N/A</span>
            {% endif %}
          </div>
          <div class="yoy-note">{{ yoy_modulation.prior_n_dias }} <span data-i18n="yoy_days">days</span></div>
        </div>
      </div>

      <p class="yoy-prose">
        {% if yoy_modulation.cur_pct is not none and yoy_modulation.prior_pct is not none %}
          {% if yoy_modulation.status == 'worse' %}
            <span data-i18n="yoy_prose_worse">Mauriti's modulation discount worsened by</span>
            <strong>{{ "%.2f"|format(yoy_modulation.delta_pp|abs) }} pp</strong>
            <span data-i18n="yoy_prose_worse_p">vs. the same period in the prior year. Investigate if structural causes (more curtailment, peak-hour mix shift) or external (lower PLD spreads).</span>
          {% elif yoy_modulation.status == 'better' %}
            <span data-i18n="yoy_prose_better">Modulation improved by</span>
            <strong>{{ "%.2f"|format(yoy_modulation.delta_pp) }} pp</strong>
            <span data-i18n="yoy_prose_better_p">vs. the same period last year. Reflects more favorable PLD spreads or operational improvements.</span>
          {% else %}
            <span data-i18n="yoy_prose_equal">Modulation is essentially flat YoY, suggesting stable structural conditions.</span>
          {% endif %}
        {% endif %}
      </p>
    </div>
    {% endif %}

    {% if itm_otm and not itm_otm.vazio %}
    <div class="section-head">
      <span class="num">V.</span>
      <h3 data-i18n="itm_title">Hours in/out of the money</h3>
      <span class="tag" data-i18n="itm_tag">SPOT EXPOSURE</span>
    </div>
    <p class="section-desc" data-i18n="itm_desc">
      Of all the hours Mauriti generated this month, what fraction coincided
      with PLD <em>above</em> the monthly mean (valuable hours, in-the-money)
      vs <em>below</em> (cheap hours, out-of-the-money). Quantifies the
      solar timing penalty in hour-by-hour terms.
    </p>

    <div class="itm-card">
      <div class="itm-grid">
        <div class="itm-cell itm-ref">
          <div class="itm-tiny" data-i18n="itm_ref">Monthly PLD reference</div>
          <div class="itm-ref-val">R$ {{ "%.0f"|format(itm_otm.pld_referencia) }}<span class="unit">/MWh</span></div>
          <div class="itm-tiny" data-i18n="itm_total">Total generation hours: {{ itm_otm.total_horas_gen }}</div>
        </div>
        <div class="itm-cell itm-pos">
          <div class="itm-label">
            <span class="itm-dot pos"></span>
            <span data-i18n="itm_in">In-the-money</span>
            <span class="itm-tiny">(PLD &gt; mean)</span>
          </div>
          <div class="itm-grid-2">
            <div>
              <div class="itm-num">{{ itm_otm.itm.n_horas }}<span class="unit">h</span></div>
              <div class="itm-tiny">{{ "%.0f"|format(itm_otm.itm.pct_horas) }}% <span data-i18n="itm_of_hours">of hours</span></div>
            </div>
            <div>
              <div class="itm-num">{{ "%.0f"|format(itm_otm.itm.pct_mwh) }}<span class="unit">%</span></div>
              <div class="itm-tiny" data-i18n="itm_of_mwh">of MWh captured</div>
            </div>
          </div>
        </div>
        <div class="itm-cell itm-neg">
          <div class="itm-label">
            <span class="itm-dot neg"></span>
            <span data-i18n="itm_out">Out-of-the-money</span>
            <span class="itm-tiny">(PLD &le; mean)</span>
          </div>
          <div class="itm-grid-2">
            <div>
              <div class="itm-num">{{ itm_otm.otm.n_horas }}<span class="unit">h</span></div>
              <div class="itm-tiny">{{ "%.0f"|format(itm_otm.otm.pct_horas) }}% <span data-i18n="itm_of_hours">of hours</span></div>
            </div>
            <div>
              <div class="itm-num">{{ "%.0f"|format(itm_otm.otm.pct_mwh) }}<span class="unit">%</span></div>
              <div class="itm-tiny" data-i18n="itm_of_mwh">of MWh captured</div>
            </div>
          </div>
        </div>
      </div>
      <p class="itm-prose">
        {% set itm_skew = itm_otm.otm.pct_mwh - itm_otm.otm.pct_horas %}
        {% if itm_skew > 5 %}
          <span data-i18n="itm_prose_negskew">Solar generation is disproportionately concentrated in low-PLD hours: Mauriti delivered</span>
          <strong>{{ "%.0f"|format(itm_otm.otm.pct_mwh) }}%</strong>
          <span data-i18n="itm_prose_negskew_b">of MWh in only</span>
          <strong>{{ "%.0f"|format(itm_otm.otm.pct_horas) }}%</strong>
          <span data-i18n="itm_prose_negskew_c">of hours when prices were below average. This is the structural cause of modulation discount.</span>
        {% elif itm_skew < -5 %}
          <span data-i18n="itm_prose_posskew">Mauriti is capturing more valuable hours than average — a favorable position.</span>
        {% else %}
          <span data-i18n="itm_prose_balanced">Generation is roughly balanced between valuable and cheap hours.</span>
        {% endif %}
      </p>
    </div>
    {% endif %}

    <!-- ========================================================= -->
    <!-- MONTHLY FORECAST (substitui o PPA What-If antigo)            -->
    <!-- Foco: projecao financeira do mes corrente (CCEE + Comercial) -->
    <!-- ========================================================= -->
    <div class="section-head">
      <span class="num">VI.</span>
      <h3 data-i18n="forecast_title">Monthly forecast</h3>
      <span class="tag" data-i18n="forecast_tag">INTERACTIVE</span>
    </div>
    <p class="section-desc" data-i18n="forecast_desc">
      Projects end-of-month financial outcome based on generation realized
      so far + daily-average pace for remaining days. Set up your hedge
      portfolio (up to 2 contracts) and see how CCEE settlement and commercial
      revenue compose the bottom line.
    </p>

    {% if forecast_data.vazio %}
    <div class="forecast-empty">
      <p data-i18n="forecast_empty">No data for the current month yet. Wait
        until end-of-day to see projection.</p>
    </div>
    {% else %}

    <div class="forecast-wrap" id="forecast-wrap">

      <!-- ========== 1. Generation projection (auto) ========== -->
      <div class="forecast-projection">
        <div class="forecast-projection-head">
          <h4 data-i18n="forecast_proj_title">Generation projection &mdash;
            {{ forecast_data.cur_month_label }}</h4>
          <span class="forecast-proj-sub">
            <span id="forecast-days-info">
              {{ forecast_data.days_elapsed }} of {{ forecast_data.days_total }}
              days elapsed
            </span>
          </span>
        </div>
        <div class="forecast-proj-grid">
          <div class="forecast-proj-cell">
            <div class="key" data-i18n="forecast_realized">Realized MTD</div>
            <div class="val"><span id="fp-realized">—</span><span class="unit"> MWh</span></div>
          </div>
          <div class="forecast-proj-cell">
            <div class="key" data-i18n="forecast_daily">Daily avg</div>
            <div class="val"><span id="fp-daily">—</span><span class="unit"> MWh/day</span></div>
          </div>
          <div class="forecast-proj-cell">
            <div class="key" data-i18n="forecast_projected">Projected (remaining)</div>
            <div class="val">+<span id="fp-projected">—</span><span class="unit"> MWh</span></div>
          </div>
          <div class="forecast-proj-cell forecast-proj-total">
            <div class="key" data-i18n="forecast_total">Total forecast</div>
            <div class="val val-strong"><span id="fp-total">—</span><span class="unit"> MWh</span></div>
            <div class="forecast-proj-meta"><span id="fp-mwm">—</span> <span data-i18n="forecast_mwm">MW avg</span></div>
          </div>
        </div>
      </div>

      <!-- ========== 2. Contract setup (sliders) ========== -->
      <div class="forecast-contracts">
        <div class="forecast-contracts-head">
          <h4 data-i18n="forecast_contracts_title">Contract portfolio</h4>
          <button class="forecast-btn-mini" id="forecast-add-c2" type="button"
                  data-i18n="forecast_btn_add_c2">+ Add 2nd contract</button>
        </div>

        <!-- Contract 1 -->
        <div class="forecast-contract" data-cid="1">
          <div class="forecast-contract-head">
            <span class="forecast-contract-tag">C1</span>
            <span class="forecast-contract-label" data-i18n="forecast_contract_1">Contract 1</span>
          </div>
          <div class="forecast-contract-grid">
            <div class="forecast-input-group">
              <label data-i18n="forecast_volume">Volume (MWh)</label>
              <div class="forecast-input-row">
                <input type="range" min="0" max="100000" step="100" value="37200" id="fc-c1-vol">
                <input type="number" min="0" max="200000" step="100" value="37200" id="fc-c1-vol-num">
              </div>
              <div class="forecast-hint" id="fc-c1-vol-hint"></div>
            </div>
            <div class="forecast-input-group">
              <label data-i18n="forecast_price">Price (R$/MWh)</label>
              <div class="forecast-input-row">
                <input type="range" min="50" max="500" step="1" value="116" id="fc-c1-price">
                <input type="number" min="0" max="2000" step="1" value="116" id="fc-c1-price-num">
              </div>
            </div>
            <div class="forecast-input-group">
              <label data-i18n="forecast_submarket">Submarket</label>
              <select id="fc-c1-sub" class="forecast-select">
                <option value="N">N</option>
                <option value="NE">NE</option>
                <option value="SECO">SE/CO</option>
                <option value="S" selected>S</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Contract 2 (collapsed by default) -->
        <div class="forecast-contract" data-cid="2" id="fc-c2-wrap" style="display:none">
          <div class="forecast-contract-head">
            <span class="forecast-contract-tag">C2</span>
            <span class="forecast-contract-label" data-i18n="forecast_contract_2">Contract 2</span>
            <button class="forecast-btn-remove" id="forecast-remove-c2" type="button" data-i18n="forecast_btn_remove">Remove</button>
          </div>
          <div class="forecast-contract-grid">
            <div class="forecast-input-group">
              <label data-i18n="forecast_volume">Volume (MWh)</label>
              <div class="forecast-input-row">
                <input type="range" min="0" max="100000" step="100" value="0" id="fc-c2-vol">
                <input type="number" min="0" max="200000" step="100" value="0" id="fc-c2-vol-num">
              </div>
              <div class="forecast-hint" id="fc-c2-vol-hint"></div>
            </div>
            <div class="forecast-input-group">
              <label data-i18n="forecast_price">Price (R$/MWh)</label>
              <div class="forecast-input-row">
                <input type="range" min="50" max="500" step="1" value="200" id="fc-c2-price">
                <input type="number" min="0" max="2000" step="1" value="200" id="fc-c2-price-num">
              </div>
            </div>
            <div class="forecast-input-group">
              <label data-i18n="forecast_submarket">Submarket</label>
              <select id="fc-c2-sub" class="forecast-select">
                <option value="N">N</option>
                <option value="NE">NE</option>
                <option value="SECO">SE/CO</option>
                <option value="S" selected>S</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Summary line: total contracted vs forecast -->
        <div class="forecast-portfolio-summary" id="forecast-portfolio-summary"></div>
      </div>

      <!-- ========== 3. PLD reference (auto, MTD) ========== -->
      <div class="forecast-pld-ref">
        <div class="forecast-pld-head">
          <h4 data-i18n="forecast_pld_title">PLD reference (MTD)</h4>
          <span class="forecast-pld-sub" data-i18n="forecast_pld_sub">
            Month-to-date hourly average per submarket</span>
        </div>
        <div class="forecast-pld-grid" id="forecast-pld-grid"></div>
        <div class="forecast-pld-meta">
          <span data-i18n="forecast_pld_eff">Mauriti effective price</span>:
          <strong id="fp-pld-eff">—</strong>
          <span class="forecast-pld-extra" data-i18n="forecast_pld_eff_sub">
            (incorporates curtailment + modulation discount)</span>
        </div>
      </div>

      <!-- ========== 4. CCEE View (detailed, auditable) ========== -->
      <div class="forecast-cards-row">
        <div class="forecast-card forecast-card-ccee">
          <div class="forecast-card-label" data-i18n="forecast_ccee_title">CCEE VIEW</div>
          <div class="forecast-card-sub" data-i18n="forecast_ccee_sub">Spot settlement at submarket PLD</div>
          <div class="forecast-card-detail" id="forecast-ccee-detail"></div>
          <div class="forecast-card-total">
            <span class="key" data-i18n="forecast_ccee_net">Net CCEE</span>
            <span class="val" id="forecast-ccee-net">—</span>
          </div>
        </div>

        <div class="forecast-card forecast-card-comm">
          <div class="forecast-card-label" data-i18n="forecast_comm_title">COMMERCIAL VIEW</div>
          <div class="forecast-card-sub" data-i18n="forecast_comm_sub">PPA contracts revenue</div>
          <div class="forecast-card-detail" id="forecast-comm-detail"></div>
          <div class="forecast-card-total">
            <span class="key" data-i18n="forecast_comm_total">Commercial total</span>
            <span class="val" id="forecast-comm-total">—</span>
          </div>
        </div>

        <div class="forecast-card forecast-card-final" id="forecast-card-final">
          <div class="forecast-card-label" data-i18n="forecast_final_title">FINAL FORECAST</div>
          <div class="forecast-card-sub" data-i18n="forecast_final_sub">Expected revenue {{ forecast_data.cur_month_label }}</div>
          <div class="forecast-final-figure" id="forecast-final-figure">—</div>
          <div class="forecast-final-meta" id="forecast-final-meta"></div>
        </div>
      </div>

      <!-- ========== 5. Short-position alert ========== -->
      <div class="forecast-alert" id="forecast-alert" style="display:none">
        <span class="forecast-alert-icon">⚠</span>
        <div class="forecast-alert-body">
          <strong data-i18n="forecast_alert_title">Short position warning</strong>
          <p id="forecast-alert-text"></p>
        </div>
      </div>

      <!-- ========== 6. PLD sensitivity strip ========== -->
      <div class="forecast-sensitivity">
        <h4 data-i18n="forecast_sens_title">PLD sensitivity (remaining days)</h4>
        <p class="forecast-sens-desc" data-i18n="forecast_sens_desc">
          What if PLD for the rest of the month is different from MTD average?
          The base scenario assumes the same PLD continues. We re-price the
          spot leg of CCEE for the projected remaining MWh.</p>
        <div class="forecast-sens-grid" id="forecast-sens-grid"></div>
      </div>

      <!-- ========== 7. Risk decomposition ========== -->
      <div class="forecast-decomp">
        <h4 data-i18n="forecast_decomp_title">Risk decomposition</h4>
        <p class="forecast-decomp-desc" data-i18n="forecast_decomp_desc">
          Where does the bottom line come from? Each component isolates one
          economic driver of the result.</p>
        <div class="forecast-decomp-grid" id="forecast-decomp-grid"></div>
      </div>

      <!-- ========== 8. Forecast vs yesterday ========== -->
      <div class="forecast-history" id="forecast-history">
        <h4 data-i18n="forecast_hist_title">Forecast tracking</h4>
        <div id="forecast-hist-content"></div>
      </div>

      <p class="forecast-disclaimer" data-i18n="forecast_disclaimer">
        Projection assumes the daily generation pace continues unchanged.
        Real outcomes deviate due to weather, curtailment, and PLD volatility.
        Use as directional reference, not as financial commitment.
      </p>
    </div>

    {% endif %}


    <div class="bignum">
      <div class="figure">
        {{ "%.1f"|format((met_mod_m.desconto_rs|abs)/1e6) }}<span>R$ M</span>
      </div>
      <div class="copy">
        <h2 data-i18n="mod_bignum_title">Potential revenue lost to modulation</h2>
        <p>
          <span data-i18n="mod_bignum_p1a">Mauriti realized</span>
          <strong>R$ {{ "%.1f"|format(met_mod_m.receita_real/1e6) }} M</strong>
          <span data-i18n="mod_bignum_p1b">when, with generation at the daily PLD average, it would have earned</span>
          <strong>R$ {{ "%.1f"|format(met_mod_m.receita_flat/1e6) }} M</strong>.
        </p>
        <p>
          <span data-i18n="mod_bignum_p2a">A discount of</span>
          <strong>{{ "%.2f"|format(met_mod_m.desconto_pct) }}%</strong>
          <span data-i18n="mod_bignum_p2b">of flat revenue — effective price of</span>
          <strong>R$ {{ "%.0f"|format(met_mod_m.preco_efetivo) }}/MWh</strong>
          <span data-i18n="mod_bignum_p2c">against an average PLD of</span>
          <strong>R$ {{ "%.0f"|format(met_mod_m.pld_medio) }}/MWh</strong>.
        </p>
      </div>
    </div>

    <div class="stats">
      <div class="stat alt">
        <div class="lbl" data-i18n="mod_kpi_disc">% Discount Mauriti</div>
        <div class="val">{{ "%.2f"|format(met_mod_m.desconto_pct) }}<span class="unit">%</span></div>
        <div class="delta">real / flat - 1</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="mod_kpi_disc_ne">% Discount NE (fleet)</div>
        {% if met_mod_ne.vazio %}<div class="val">—</div>
        {% else %}<div class="val">{{ "%.2f"|format(met_mod_ne.desconto_pct) }}<span class="unit">%</span></div>{% endif %}
        <div class="delta" data-i18n="ne_fleet_label">NE solar fleet benchmark</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="mod_kpi_vs">vs NE benchmark</div>
        {% if met_mod_ne.vazio %}<div class="val">—</div>
        {% else %}
        {% set d = met_mod_m.desconto_pct - met_mod_ne.desconto_pct %}
        <div class="val">{% if d > 0 %}+{% endif %}{{ "%.2f"|format(d) }}<span class="unit">pp</span></div>
        {% endif %}
        <div class="delta" data-i18n="mod_kpi_vs_sub">% discount delta</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="mod_kpi_eff">Effective price</div>
        <div class="val">R$ {{ "%.0f"|format(met_mod_m.preco_efetivo) }}<span class="unit">/MWh</span></div>
        <div class="delta">real / MWh</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="mod_kpi_pld">Avg PLD</div>
        <div class="val">R$ {{ "%.0f"|format(met_mod_m.pld_medio) }}<span class="unit">/MWh</span></div>
        <div class="delta">{{ submercado }}</div>
      </div>
    </div>

    <div class="section-head">
      <span class="num">I.</span><h3 data-i18n="mod_s1_title">Why we lose revenue</h3>
      <span class="tag" data-i18n="mod_s1_tag">TYPICAL HOURLY PROFILE</span>
    </div>
    <p class="section-desc" data-i18n="mod_s1_desc">
      Average hour of day across the period. Solar peak generation
      coincides with PLD valley in the submarket — we sell most MWh
      when they're worth less.
    </p>
    <div class="chart"><div id="mod_perfil" style="height:420px"></div></div>

    <div class="section-head">
      <span class="num">II.</span><h3 data-i18n="mod_s2_title">Mauriti vs NE solar fleet</h3>
      <span class="tag" data-i18n="mod_s2_tag">MONTHLY HISTORY</span>
    </div>
    <p class="section-desc" data-i18n="mod_s2_desc">
      Month-by-month % discount comparison. Dry periods (PLD usually
      high all day) tend to have smaller discount; light-load periods
      (more solar surplus in the SIN) amplify the effect.
    </p>
    <div class="chart"><div id="mod_hist" style="height:400px"></div></div>

    <div class="section-head">
      <span class="num">III.</span><h3 data-i18n="mod_s3_title">Most painful days</h3>
      <span class="tag" data-i18n="mod_s3_tag">RANKING BY R$</span>
    </div>
    <p class="section-desc" data-i18n="mod_s3_desc">
      Days with largest absolute discount (R$). Usually coincide with
      very low midday PLD (excess solar supply on the SIN).
    </p>
    <div class="chart"><div id="mod_top" style="height:520px"></div></div>

    <div class="disclaimer">
      <p data-i18n="mod_disclaimer"><strong>Important:</strong> this analysis shows the <em>theoretical</em>
      discount assuming all generation settles at hourly PLD. The real
      revenue impact for PowerChina depends on each unit's commercial
      regime: fixed PPA (R$/MWh) absorbs the effect but loses spot value;
      MCP settlement captures the full impact; CCEE shape products
      neutralize the discount. Numbers here are
      <strong>directional indicator</strong> of the premium that would
      be worth paying for a perfect modulation hedge.</p>
    </div>

    {% endif %}

  </div><!-- /tab mod -->

  <!-- ============================================================ -->
  <!-- TAB: REN 1.030                                                -->
  <!-- ============================================================ -->
  <div class="tab-pane" data-tab="ren">

    <div class="hero">
      <div class="kicker" data-i18n="ren_kicker">REN 1.030/2022 &middot; Recoverable curtailment events</div>
      <h1>
        <span data-i18n="ren_h1_a">Cuts that the regulator</span>
        <em data-i18n="ren_h1_b">may compensate.</em>
      </h1>
      <p class="lede" data-i18n="ren_lede">
        ANEEL's Resolution 1.030/2022 establishes that curtailment classified
        as <strong>REL (external unavailability)</strong> or
        <strong>CNF (reliability)</strong> may be eligible for financial
        compensation. This view lists every such event for the Mauriti
        complex — the raw input for the formal claim process.
      </p>
      <div class="byline">
        <span data-i18n="byline_period">Period</span> <span>{{ periodo }}</span> &nbsp;&middot;&nbsp;
        <span data-i18n="byline_filter">Filter</span> <span>REL + CNF only</span>
      </div>
    </div>

    {% if met_ren.vazio %}
    <div class="empty-msg" data-i18n="ren_empty">
      No eligible events were found in the analysis period.
    </div>
    {% else %}

    <div class="stats">
      <div class="stat alt">
        <div class="lbl" data-i18n="ren_kpi_events">Eligible events</div>
        <div class="val">{{ met_ren.n_eventos }}</div>
        <div class="delta">
          {{ met_ren.n_eventos_rel }} REL · {{ met_ren.n_eventos_cnf }} CNF
        </div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="ren_kpi_mwh">Total eligible MWh</div>
        <div class="val">{{ "%.0f"|format(met_ren.mwh_total) }}<span class="unit">MWh</span></div>
        <div class="delta">{{ "%.1f"|format(met_ren.pct_total) }}% <span data-i18n="ren_pct_total">of total curtailment</span></div>
      </div>
      <div class="stat">
        <div class="lbl">REL</div>
        <div class="val">{{ "%.0f"|format(met_ren.rel_mwh) }}<span class="unit">MWh</span></div>
        <div class="delta" data-i18n="ren_rel_sub">external unavailability</div>
      </div>
      <div class="stat">
        <div class="lbl">CNF</div>
        <div class="val">{{ "%.0f"|format(met_ren.cnf_mwh) }}<span class="unit">MWh</span></div>
        <div class="delta" data-i18n="ren_cnf_sub">reliability</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="ren_kpi_days">Days affected</div>
        <div class="val">{{ met_ren.n_dias }}</div>
        <div class="delta">{{ met_ren.n_usinas }} <span data-i18n="ren_units_short">UFVs</span></div>
      </div>
    </div>

    <div class="section-head">
      <span class="num">I.</span><h3 data-i18n="ren_s1_title">Eligible volume by month</h3>
      <span class="tag" data-i18n="ren_s1_tag">REL + CNF</span>
    </div>
    <p class="section-desc" data-i18n="ren_s1_desc">
      Stacked bars show how the recoverable volume distributes across REL
      and CNF — a useful split for the regulatory dossier.
    </p>
    <div class="chart"><div id="ren_mensal" style="height:380px"></div></div>

    <div class="section-head">
      <span class="num">II.</span><h3 data-i18n="ren_s2_title">Top causes</h3>
      <span class="tag" data-i18n="ren_s2_tag">RESTRICTION ORIGIN</span>
    </div>
    <p class="section-desc" data-i18n="ren_s2_desc">
      Origin classification as reported by ONS:
      <strong>LOC</strong> = local (transmission line, substation or bay
      near the plant; specific counterparty can be held responsible) ·
      <strong>SIS</strong> = systemic (regional grid limitations, e.g.
      saturated NE → SE/CO export corridor; structural issue affecting
      the entire region). The split tells you how much of the claim
      has an addressable counterparty vs how much is a systemic constraint.
    </p>
    <div class="chart"><div id="ren_origem" style="height:380px"></div></div>

    <div class="section-head">
      <span class="num">III.</span><h3 data-i18n="ren_s3_title">Event log</h3>
      <span class="tag" data-i18n="ren_s3_tag">EXPORTABLE</span>
    </div>
    <p class="section-desc" data-i18n="ren_s3_desc">
      Every eligible event identified in the period, with start/end
      timestamp, plant, reason, origin and curtailed energy. Use the
      CSV export for further processing in spreadsheets or claim software.
    </p>

    <div class="csv-actions">
      <a href="eventos_elegiveis_ren1030.csv" class="btn-csv" download>
        ↓ <span data-i18n="ren_export_csv">Download CSV</span>
      </a>
      <span class="events-summary">
        {{ met_ren.n_eventos }} <span data-i18n="ren_events_lower">events</span> ·
        {{ "%.0f"|format(met_ren.mwh_total) }} MWh ·
        {{ met_ren.n_dias }} <span data-i18n="ren_days_lower">days</span>
      </span>
    </div>

    <div class="events-table-wrap">
      <table class="events-table">
        <thead>
          <tr>
            <th data-i18n="th_start">Start</th>
            <th data-i18n="th_end">End</th>
            <th data-i18n="th_dur">Duration</th>
            <th data-i18n="th_plant">Plant</th>
            <th data-i18n="th_reason">Reason</th>
            <th data-i18n="th_origin">Origin</th>
            <th class="num" data-i18n="th_mwh">MWh cut</th>
            <th class="num" data-i18n="th_pld">PLD avg</th>
          </tr>
        </thead>
        <tbody>
          {% for ev in eventos_top %}
          <tr>
            <td>{{ ev.data_inicio_str }} {{ ev.hora_inicio }}</td>
            <td>{{ ev.data_fim_str }} {{ ev.hora_fim }}</td>
            <td>{{ "%.1f"|format(ev.duracao_h) }}h</td>
            <td>{{ ev.usina }}</td>
            <td class="razao {{ ev.razao|lower }}">{{ ev.razao }}</td>
            <td>{{ ev.origem_label }}</td>
            <td class="num">{{ "%.1f"|format(ev.mwh_cortado) }}</td>
            <td class="num">{{ "%.0f"|format(ev.pld_medio) }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% if met_ren.n_eventos > eventos_top|length %}
    <p style="margin-top:14px;font-family:'IBM Plex Mono',monospace;
              font-size:11px;color:var(--muted);letter-spacing:0.05em">
      <span data-i18n="ren_showing">Showing</span> {{ eventos_top|length }}
      <span data-i18n="ren_of">of</span> {{ met_ren.n_eventos }}.
      <span data-i18n="ren_full_csv">Full list in the CSV export above.</span>
    </p>
    {% endif %}

    <div class="disclaimer">
      <p><strong data-i18n="ren_caveat">Important:</strong>
      <span data-i18n="ren_caveat_p">
      this view identifies events that meet the formal eligibility test
      under REN 1.030/2022 (REL/CNF reasons). The actual recovery value
      depends on each contract's declared cost (EAR/COD), the Recoverable
      Margin Allocation (MRA) rules, and CMO discount mechanics — which
      require regulatory and contractual data outside this dataset. We
      deliberately do not show a R$ value here to avoid misleading
      anchoring. The CSV is the input for the legal/regulatory team to
      build the formal claim.
      </span></p>
    </div>

    {% endif %}

  </div><!-- /tab ren -->

  <!-- ============================================================ -->
  <!-- TAB: SOLAR RESOURCE                                           -->
  <!-- ============================================================ -->
  <div class="tab-pane" data-tab="solar">

    <div class="hero">
      <div class="kicker" data-i18n="solar_kicker">NASA POWER &middot; Local irradiance</div>
      <h1>
        <span data-i18n="solar_h1_a">Are we cutting</span>
        <em data-i18n="solar_h1_b">when the sun shines most?</em>
      </h1>
      <p class="lede" data-i18n="solar_lede">
        Cross-reference between Mauriti's <strong>ONS-classified
        curtailment</strong> (reasons REL/CNF/ENE/PAR) and Global
        Horizontal Irradiance (GHI) from NASA POWER for the plant
        coordinates. Reveals whether cuts concentrate on high-irradiance
        hours — when the opportunity cost of curtailment is largest.
        Unclassified underperformance (morning ramp-up, optimistic ONS
        baseline) is excluded so the signal reflects only "real"
        curtailment.
      </p>
      <div class="byline">
        <span data-i18n="solar_byline_src">Source</span> <span>NASA POWER</span> &nbsp;&middot;&nbsp;
        <span data-i18n="solar_byline_coord">Coordinates</span> <span>{{ mauriti_lat }}°S, {{ mauriti_lon|abs }}°W</span> &nbsp;&middot;&nbsp;
        <span data-i18n="solar_byline_lat">Latency</span> <span>~5-7 days</span>
      </div>
    </div>

    {% if met_irr.vazio %}
    <div class="empty-msg" data-i18n="solar_empty">
      Could not retrieve irradiance data from NASA POWER for the period.
    </div>
    {% else %}

    <div class="stats">
      <div class="stat alt">
        <div class="lbl" data-i18n="solar_kpi_pct_cut_sun">% sunny hours with cut</div>
        <div class="val">{{ "%.1f"|format(met_irr.pct_horas_corte_em_sol) }}<span class="unit">%</span></div>
        <div class="delta" data-i18n="solar_kpi_pct_sub">GHI > 600 W/m²</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="solar_kpi_cf_sun">CF in clear-sky hours</div>
        <div class="val">{{ "%.2f"|format(met_irr.cf_em_sol_pleno) }}<span class="unit">%</span></div>
        <div class="delta" data-i18n="solar_kpi_cf_sub">vs daily CF</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="solar_kpi_ghi">Avg daytime GHI</div>
        <div class="val">{{ "%.0f"|format(met_irr.ghi_medio_diurno) }}<span class="unit">W/m²</span></div>
        <div class="delta" data-i18n="solar_kpi_ghi_sub">period mean</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="solar_kpi_peak">Peak GHI</div>
        <div class="val">{{ "%.0f"|format(met_irr.ghi_pico) }}<span class="unit">W/m²</span></div>
        <div class="delta" data-i18n="solar_kpi_peak_sub">max in period</div>
      </div>
      <div class="stat">
        <div class="lbl" data-i18n="solar_kpi_temp">Avg temp during cuts</div>
        <div class="val">{{ "%.1f"|format(met_irr.temp_media_corte) }}<span class="unit">°C</span></div>
        <div class="delta">2m air</div>
      </div>
    </div>

    <div class="section-head">
      <span class="num">I.</span><h3 data-i18n="solar_s1_title">CF by irradiance bin</h3>
      <span class="tag" data-i18n="solar_s1_tag">RELATIONSHIP</span>
    </div>
    <p class="section-desc" data-i18n="solar_s1_desc">
      Curtailment factor grouped by GHI bin. If high-GHI bins show
      higher CF, it confirms cuts concentrate on the most valuable
      production hours.
    </p>
    <div class="chart"><div id="irr_scatter" style="height:380px"></div></div>

    <div class="section-head">
      <span class="num">II.</span><h3 data-i18n="solar_s2_title">Typical hour: sun vs cuts</h3>
      <span class="tag" data-i18n="solar_s2_tag">DIURNAL PROFILE</span>
    </div>
    <p class="section-desc" data-i18n="solar_s2_desc">
      Average hour-by-hour profile. The shape of the GHI curve and the
      curtailment curve tells you whether the plant is being cut when
      the resource is best (the worst case).
    </p>
    <div class="chart"><div id="irr_perfil" style="height:400px"></div></div>

    <div class="disclaimer">
      <p data-i18n="solar_disclaimer">
      <strong>Note:</strong> NASA POWER provides reanalysis-derived
      hourly irradiance at 0.5° × 0.625° resolution, sufficient for
      monthly-pattern analysis but coarser than on-site measurements.
      The plant's actual measured irradiance (if available from local
      pyranometers) would yield more precise correlations. NASA POWER
      data has 5-7 days latency, so the most recent week is excluded.
      </p>
    </div>

    {% endif %}

  </div><!-- /tab solar -->

  <!-- ============================================================ -->
  <!-- TAB: BENCHMARK BUILDER                                        -->
  <!-- ============================================================ -->
  <div class="tab-pane" data-tab="bench">

    <div class="hero">
      <div class="kicker" data-i18n="bench_kicker">CE peer comparison &middot; pre-defined groups</div>
      <h1>
        <span data-i18n="bench_h1_a">How does Mauriti compare</span>
        <em data-i18n="bench_h1_b">against its peers?</em>
      </h1>
      <p class="lede" data-i18n="bench_lede">Pick any of the 7 pre-defined peer groups in CE
      and see Mauriti's KPIs against that benchmark, side by side. Use the
      dropdown to switch between groups and see all metrics update instantly.</p>
    </div>

    <!-- Group selector -->
    <div class="bench-selector">
      <div class="bench-selector-head">
        <label data-i18n="bench_select_label">Compare Mauriti against:</label>
        <div class="bench-selector-meta">
          <button class="bench-btn-mini" id="bench-select-all" type="button"
                  data-i18n="bench_btn_all">Select all</button>
          <button class="bench-btn-mini" id="bench-select-none" type="button"
                  data-i18n="bench_btn_none">Clear</button>
          <button class="bench-btn-mini" id="bench-select-ufv" type="button"
                  data-i18n="bench_btn_ufv">Solar only</button>
        </div>
      </div>
      <div class="bench-checkboxes" id="bench-checkboxes">
        <!-- checkboxes inserted by JS -->
      </div>
      <div class="bench-fleet-toggle">
        <label class="bench-checkbox-row bench-fleet-row">
          <input type="checkbox" id="bench-show-ne-fleet" checked>
          <span class="bench-cb-label">
            <span data-i18n="bench_show_ne">Show NE solar fleet average</span>
            <span class="bench-cb-meta" id="bench-ne-meta"></span>
          </span>
        </label>
      </div>
      <p class="bench-meta" id="bench-meta"></p>
    </div>

    <!-- 3 KPI cards: Mauriti / Selected avg / Diff -->
    <div class="bench-cards">
      <div class="bench-card bench-card-mauriti">
        <div class="bench-card-label" data-i18n="bench_card_mauriti">MAURITI</div>
        <div class="bench-card-sub" id="bench-mauriti-sub">9 UFVs</div>
        <div class="bench-rows" id="bench-mauriti-rows"></div>
      </div>
      <div class="bench-card bench-card-peer">
        <div class="bench-card-label" id="bench-peer-label" data-i18n="bench_card_selected">SELECTED PEERS</div>
        <div class="bench-card-sub" id="bench-peer-sub"></div>
        <div class="bench-rows" id="bench-peer-rows"></div>
      </div>
      <div class="bench-card bench-card-diff">
        <div class="bench-card-label" data-i18n="bench_card_diff">DIFFERENCE</div>
        <div class="bench-card-sub" data-i18n="bench_card_diff_sub">Mauriti − Peer avg</div>
        <div class="bench-rows" id="bench-diff-rows"></div>
      </div>
    </div>

    <!-- Time series chart: monthly CF since jul/2025 -->
    <div id="g_bench_monthly" style="width:100%;height:480px;margin:32px 0 16px"></div>

    <!-- Bar comparison chart (annual totals) -->
    <div id="g_bench_compare" style="width:100%;height:380px;margin:16px 0 32px"></div>

    <!-- All groups context table -->
    <div class="bench-table-wrap">
      <h4 data-i18n="bench_table_title">All peer groups — annual KPIs</h4>
      <p class="bench-table-desc" data-i18n="bench_table_desc">
        Reference table with all peer groups including Mauriti and the
        NE solar fleet aggregate. Click any row to toggle its selection.</p>
      <div class="events-table-wrap">
        <table class="events-table bench-table">
          <thead>
            <tr>
              <th></th>
              <th data-i18n="bench_th_group">Group</th>
              <th data-i18n="bench_th_source">Source</th>
              <th class="num" data-i18n="bench_th_n_plants"># plants</th>
              <th class="num" data-i18n="bench_th_gen">Total gen (GWh)</th>
              <th class="num" data-i18n="bench_th_curt">Curt (GWh)</th>
              <th class="num" data-i18n="bench_th_cf">CF (%)</th>
            </tr>
          </thead>
          <tbody id="bench-all-rows"></tbody>
        </table>
      </div>
    </div>

  </div><!-- /tab bench -->

  <!-- Glossary -->
  <details class="glossary">
    <summary data-i18n="glossary_title">Glossary &amp; technical notes</summary>
    <dl>
      <dt>PLD</dt>
      <dd data-i18n="gloss_pld">Hourly settlement price (R$/MWh) set by CCEE for each
          submarket. Reference for spot energy settlement in Brazil.</dd>
      <dt>ONS</dt>
      <dd data-i18n="gloss_ons">National System Operator. Publishes constrained-off
          data (curtailment) for wind and solar plants.</dd>
      <dt>CCEE</dt>
      <dd data-i18n="gloss_ccee">Electric Energy Commercialization Chamber.
          Publishes hourly PLD and operates the spot market.</dd>
      <dt>NE</dt>
      <dd data-i18n="gloss_ne">Northeast submarket of the SIN (National
          Interconnected System), where Mauriti is located.</dd>
      <dt>REN 1.030</dt>
      <dd data-i18n="gloss_ren">ANEEL Resolution 1.030/2022. Defines
          eligibility of curtailment for compensation. REL and CNF
          reasons are eligible; ENE (energy) and PAR (access opinion)
          generally are not.</dd>
      <dt>CF (%)</dt>
      <dd data-i18n="gloss_cf">Curtailment Factor. Curtailed energy
          divided by reference (estimated) generation, expressed as %.</dd>
      <dt>REL / CNF / ENE / PAR</dt>
      <dd data-i18n="gloss_reasons">Curtailment reason codes. REL =
          external unavailability, CNF = reliability, ENE = energy
          (oversupply), PAR = access opinion limit.</dd>
      <dt>LOC / SIS</dt>
      <dd data-i18n="gloss_origins">Restriction origin codes.
          <strong>LOC</strong> = local (constraint near the plant —
          transmission line, substation or bay; a specific transmission
          asset is responsible). <strong>SIS</strong> = systemic
          (regional grid constraint — typically saturated inter-regional
          export limits or stability requirements; affects multiple
          plants simultaneously).</dd>
      <dt>GHI</dt>
      <dd data-i18n="gloss_ghi">Global Horizontal Irradiance. Total
          solar power received per square meter on a horizontal surface.
          From NASA POWER reanalysis.</dd>
      <dt data-i18n="gloss_modulation_dt">Modulation discount</dt>
      <dd data-i18n="gloss_modulation">Difference between actual revenue
          (Σ MWh<sub>h</sub> × PLD<sub>h</sub>) and "flat" revenue
          (MWh<sub>day</sub> × PLD<sub>daily avg</sub>). Captures the
          opportunity cost of generating during low-PLD hours.</dd>
    </dl>
  </details>

  <footer>
    <p><strong data-i18n="footer_sources">Sources</strong>
       <span data-i18n="footer_sources_p">Constrained-off Operating Restriction (ONS,
       semi-hourly base, plant detail + consolidated with reasons).
       Hourly PLD (CCEE). Solar irradiance (NASA POWER, hourly
       reanalysis at plant coordinates).</span></p>
    <p><strong data-i18n="footer_defs">Definitions</strong>
       <span data-i18n="footer_defs_p">Curtailment = max(0, val_geracaoestimada −
       val_geracaoverificada). Lost revenue = curtailment_MWh ×
       hourly_PLD. CF = curtailment / expected (%). Modulation discount
       = (Σ MWh<sub>h</sub> × PLD<sub>h</sub>) − (Σ MWh<sub>day</sub> ×
       PLD<sub>daily_avg</sub>).</span></p>
    <p><strong data-i18n="footer_bench">NE benchmark</strong>
       <span data-i18n="footer_bench_p">Modulation benchmark aggregates all UFVs of
       NE submarket as a single fleet weighted by generation. Shows
       the discount the NE fleet collectively suffered.</span></p>
    <p class="colofao">Mauriti — Curtailment, Modulation, REN 1.030 &amp;
       Solar Resource report. <span data-i18n="footer_built">Built</span>
       {{ gerado_em }}.</p>

    <div class="credits">
      <div class="credits-row">
        <span><strong data-i18n="cred_data">Data sources</strong>:</span>
        <span>ONS <span data-i18n="cred_ons">Open Data</span></span>
        <span>&middot;</span>
        <span>CCEE <span data-i18n="cred_ccee">Open Data Portal</span></span>
        <span>&middot;</span>
        <span>NASA POWER (GHI)</span>
      </div>
      <div class="credits-row" style="margin-top:6px">
        <span><strong data-i18n="cred_tech">Tech stack</strong>:</span>
        <span>Python &middot; pandas &middot; Plotly &middot; curl_cffi (TLS bypass)</span>
        <span>&middot;</span>
        <span>GitHub Actions (daily 09:15 BRT)</span>
      </div>
      <div class="version-info">
        <span data-i18n="cred_version">Version</span>:
        <code>v{{ dash_version }}</code>
        <span data-i18n="cred_released">released</span> {{ dash_version_date }}
        <span class="changelog-toggle" onclick="toggleChangelog()">
          [<span data-i18n="cred_changelog">changelog</span>]
        </span>
      </div>
      <ul class="changelog-list" id="changelog-list" style="display:none">
        {% for entry in dash_changes %}
        <li>{{ entry }}</li>
        {% endfor %}
      </ul>
    </div>
  </footer>

</div>

<script>
const FIGS = {{ figs_json|safe }};
const MOD_MENSAL = {{ mod_mensal_json|safe }};
const FORECAST = {{ forecast_json|safe }};
const BENCH_KPIS = {{ bench_kpis_json|safe }};

// I18N dictionary
const I18N = {
  pt: {
    page_title: "Mauriti — Estudo de Curtailment & Modulação",
    volume: "Mauriti Report — Nº 01",
    updated: "atualizado",
    tab_curt: "Curtailment",
    tab_mod: "Efeito Modulação",
    tab_ren: "Tracker REN 1.030",
    tab_solar: "Recurso Solar",
    tab_bench: "Benchmark",
    forecast_title: "Previsão mensal",
    forecast_tag: "INTERATIVO",
    forecast_desc: "Projeta o resultado financeiro do mês baseado na geração realizada até agora + ritmo diário pros dias restantes. Configure seu portfólio de hedge (até 2 contratos) e veja como liquidação CCEE e receita comercial compõem o resultado.",
    forecast_empty: "Sem dados pro mês corrente ainda. Aguarde o fim do dia pra ver a projeção.",
    forecast_proj_title: "Projeção de geração",
    forecast_realized: "Realizado MTD",
    forecast_daily: "Média diária",
    forecast_projected: "Projeção (restante)",
    forecast_total: "Total previsto",
    forecast_mwm: "MW médios",
    forecast_contracts_title: "Portfólio de contratos",
    forecast_btn_add_c2: "+ Adicionar 2º contrato",
    forecast_btn_remove: "Remover",
    forecast_contract_1: "Contrato 1",
    forecast_contract_2: "Contrato 2",
    forecast_volume: "Volume (MWh)",
    forecast_price: "Preço (R$/MWh)",
    forecast_submarket: "Submercado",
    forecast_pld_title: "PLD referência (MTD)",
    forecast_pld_sub: "Média horária do mês até hoje por submercado",
    forecast_pld_eff: "Preço efetivo Mauriti",
    forecast_pld_eff_sub: "(considera curtailment + modulação)",
    forecast_ccee_title: "VISÃO CCEE",
    forecast_ccee_sub: "Liquidação no spot ao PLD do submercado",
    forecast_ccee_net: "Líquido CCEE",
    forecast_comm_title: "VISÃO COMERCIAL",
    forecast_comm_sub: "Receita dos contratos PPA",
    forecast_comm_total: "Total comercial",
    forecast_final_title: "PREVISÃO FINAL",
    forecast_final_sub: "Receita esperada",
    forecast_alert_title: "Alerta: posição curta",
    forecast_sens_title: "Sensibilidade ao PLD (dias restantes)",
    forecast_sens_desc: "E se o PLD do restante do mês for diferente da média MTD? O cenário base assume que o mesmo PLD continua. Repreçamos a parte spot da CCEE pros MWh remanescentes projetados.",
    forecast_decomp_title: "Decomposição de risco",
    forecast_decomp_desc: "De onde vem o resultado? Cada componente isola um driver econômico distinto.",
    forecast_hist_title: "Tracking do forecast",
    forecast_disclaimer: "A projeção assume que o ritmo diário continua inalterado. Resultados reais variam por clima, curtailment e volatilidade do PLD. Use como referência direcional, não como compromisso financeiro.",
    bench_kicker: "Comparação peer no CE · grupos pré-definidos",
    bench_h1_a: "Como Mauriti se compara",
    bench_h1_b: "contra seus peers?",
    bench_lede: "Escolha qualquer um dos 7 grupos peer pré-definidos no CE e veja os KPIs de Mauriti contra esse benchmark, lado a lado. Use o dropdown para alternar entre grupos e ver todas as métricas atualizarem instantaneamente.",
    bench_select_label: "Comparar Mauriti com:",
    bench_card_mauriti: "MAURITI",
    bench_card_diff: "DIFERENÇA",
    bench_card_diff_sub: "Mauriti − Peer",
    bench_table_title: "Todos os 7 grupos peer — KPIs num só lugar",
    bench_table_desc: "Tabela referência com todos os grupos peer incluindo Mauriti. Clique em qualquer linha para tornar aquele grupo a comparação ativa acima.",
    bench_th_group: "Grupo",
    bench_th_source: "Fonte",
    bench_th_n_plants: "# usinas",
    bench_th_gen: "Geração total (GWh)",
    bench_th_curt: "Cortado (GWh)",
    bench_th_cf: "CF (%)",
    bench_card_selected: "PEERS SELECIONADOS",
    bench_btn_all: "Todos",
    bench_btn_none: "Limpar",
    bench_btn_ufv: "Só solar",
    bench_show_ne: "Mostrar média da frota NE solar",
    period: "Período",
    of: "de",
    of_lower: "de",
    days_elapsed: "dias decorridos",
    days_short: "dias",
    weekly_update: "Atualizado semanalmente",
    tracker_title: "Acompanhamento — mês corrente",
    expected_month: "Esperada (mês)",
    expected_lower: "esperados",
    cut_month: "Cortada (mês)",
    cf_month: "CF% do mês",
    vs_quarter: "vs trim.",
    quarter_avg: "CF medio trim.",
    ref_90d: "referência 90 dias",
    worst_day: "Pior dia (CF%)",
    day_short: "dia",
    auto_read: "Leitura automática",
    comp_analysis: "Análise comparativa",
    vs_avg: "vs médio",
    vs_bench: "vs benchmark",
    ne_bench: "Benchmark NE",
    ne_solar_fleet: "frota UFV NE",
    ne_fleet_period: "frota UFV NE no período",
    ne_fleet_label: "benchmark frota UFV NE",
    mauriti_worse: "Mauriti pior",
    mauriti_better: "Mauriti melhor",
    pld_unavail: "PLD Indisponível — Modulação não calculada",
    pld_unavail_p: "Não foi possível obter o PLD horário da CCEE nesta execução. Os gráficos abaixo usam R$ 200/MWh como placeholder.",
    byline_submarket: "Submercado",
    byline_units: "UFVs Mauriti",
    byline_bench: "Benchmark",
    byline_groups: "grupos",
    byline_asof: "Apurado em",
    byline_period: "Período",
    byline_filter: "Filtro",
    mod_byline_bench: "Benchmark",

    trend_worsening: "↑ Tendência piorando",
    trend_improving: "↓ Tendência melhorando",
    trend_stable: "→ Tendência estável",
    trend_window_label: "(30d vs 365d)",
    trend_30d: "Últimos 30 dias",
    trend_90d: "Últimos 90 dias",
    trend_365d: "Últimos 365 dias",
    trend_curtailed: "cortados",

    curt_kicker: "PowerChina · Complexo Fotovoltaico Mauriti, Ceará",
    curt_h1_a: "Quanto custou ao Mauriti",
    curt_h1_b: "cada megawatt-hora",
    curt_h1_c: "cortado.",
    curt_lede: "Estudo de constrained-off do <strong>Complexo Fotovoltaico Mauriti</strong> no período de {{ periodo }}, com benchmark contra usinas eólicas e fotovoltaicas do Ceará, quebra por razão do corte (REL/CNF/ENE/PAR) e estimativa de receita perdida a PLD horário.",
    curt_bignum_title: "Energia não entregue por restrição de operação",
    curt_bignum_p1: "Equivale a <strong>R$ {{ '%.1f'|format(met_m.receita_perdida/1e6) }} milhões</strong> em receita perdida estimada a PLD horário do submercado {{ submercado }}, com curtailment factor de <strong>{{ '%.2f'|format(met_m.curtailment_factor) }}%</strong>.",
    curt_bignum_p2: "Desse total, <strong>{{ '%.1f'|format(met_m.pct_ressarcivel) }}% potencialmente ressarcíveis</strong> sob a REN ANEEL 1.030/2022 (razões REL e CNF).",
    kpi_cf: "Curtailment Factor",
    kpi_cf_sub: "cortado / referência",
    kpi_lost_rev: "Receita perdida",
    kpi_recover: "Pot. ressarcível",
    kpi_pld_cut: "PLD durante corte",
    kpi_vs_bench: "vs Benchmark CE",
    kpi_vs_bench_sub: "delta de CF",
    curt_s1_title: "O ritmo dos cortes",
    curt_s1_tag: "SÉRIE TEMPORAL",
    curt_s1_desc: "A área entre a referência (linha pontilhada verde) e a geração realizada representa a energia que poderia ter sido produzida e não foi.",
    curt_s2_title: "Por que se corta",
    curt_s2_tag: "RAZÕES DA RESTRIÇÃO",
    curt_s2_desc: "Indisponibilidade externa (REL) e Confiabilidade (CNF) são tipicamente tratáveis sob a REN 1.030/2022; razão energética (ENE) raramente o é.",
    curt_s3_title: "Mauriti vs ativos do CE",
    curt_s3_tag: "BENCHMARK FIXO",
    curt_s3_desc: "Comparação direta com",
    curt_s3_desc2: "grupos",
    curt_s4_title: "Onde, no dia, se corta",
    curt_s4_tag: "PADRÃO HORÁRIO",
    curt_s4_desc: "Heatmap do CF% por hora do dia. Cortes em 11h–14h indicam restrição sistêmica do NE; pulverizados sugerem limitação local.",
    curt_s5_title: "Padrão semanal",
    curt_s5_tag: "DIA DA SEMANA × HORA",
    curt_s5_desc: "Mesmo heatmap, agregado por dia-da-semana × hora. Revela se o curtailment é fenômeno semanal estrutural (ex: fins-de-semana com carga industrial menor = mais excedente solar = mais corte).",

    mod_recap_days: "dias decorridos",
    mod_recap_gen: "gerados",
    mod_recap_real: "Receita real",
    mod_recap_vs: "vs flat",
    mod_recap_disc: "Desconto",
    mod_recap_worst: "Pior dia",
    mod_table_title: "Detalhe diário de preços · R$/MWh",
    mod_table_sub: "Compara o que Mauriti receberia se vendesse à média diária do PLD (\"flat\") com o que efetivamente recebeu por MWh vendido (\"efetivo\"). A diferença é o custo de modulação por MWh.",
    mod_th_day: "Dia",
    mod_th_mwh: "MWh gerados",
    mod_th_pld_flat: "PLD flat (R$/MWh)",
    mod_th_pld_eff: "PLD efetivo (R$/MWh)",
    mod_th_delta_mwh: "Δ R$/MWh",
    mod_th_delta_pct: "Δ %",
    mod_th_revenue: "Receita real (R$)",
    ppa_title: "Simulador PPA — análise hipotética",
    ppa_tag: "INTERATIVO",
    ppa_desc: "Simula qual seria a receita de Mauriti sob diferentes estruturas de PPA: um volume mensal fixo entregue de forma uniforme (flat) hora a hora a um preço acordado, opcionalmente em outro submercado. Compara com a receita real do MCP modulado no período.",
    ppa_type_a: "Volume limitado (A)",
    ppa_type_b: "Toda geração (B)",
    ppa_price_label: "Preço PPA (R$/MWh)",
    ppa_volume_label: "Volume contratado (MWh/mês, flat)",
    ppa_sub_label: "Submercado de entrega",
    ppa_sub_ne: "NE — Nordeste (mesmo da usina)",
    ppa_sub_seco: "SE/CO — Sudeste/Centro-Oeste",
    ppa_sub_s: "S — Sul",
    ppa_sub_n: "N — Norte",
    ppa_add_second: "+ Adicionar 2º PPA",
    ppa_add_hint: "Combine dois contratos em submercados diferentes",
    ppa_second_label: "2º PPA",
    ppa_remove_second: "Remover 2º PPA",
    ppa_sens_title: "Matriz de sensibilidade: preço × volume",
    ppa_sens_desc: "Δ receita (PPA vs MCP) para a grade completa de preço e volume contratados no submercado selecionado. Zona verde = PPA vence; zona vermelha = MCP vence. A linha preta marca o break-even (Δ = 0). A estrela ★ indica o ponto atual do simulador.",
    ppa_sens_hint: "Passe o mouse sobre qualquer célula para ver o Δ exato. Use os sliders acima para mover a ★ e explorar. A matriz recalcula quando você muda o submercado ou tipo de contrato.",
    ppa_kpi_mcp: "Receita real MCP (sem PPA)",
    ppa_kpi_mcp_hint: "Conforme observado no período",
    ppa_kpi_ppa: "Receita simulada (com PPA)",
    ppa_kpi_breakeven: "Preço PPA break-even",
    ppa_kpi_breakeven_hint: "Acima disso, PPA vence o MCP",
    ppa_monthly_title: "Comparação mês a mês",
    ppa_th_month: "Mês",
    ppa_th_gen: "MWh gerados",
    ppa_th_pld_sub: "PLD sub (R$/MWh)",
    ppa_th_mcp: "MCP real (R$M)",
    ppa_th_ppa: "Com PPA (R$M)",
    ppa_th_delta: "Δ (R$M)",
    ppa_th_better: "Melhor",
    ppa_disclaimer: "Modelo simplificado. PPAs reais incluem flexibilidade sazonal, multas por desvio, encargos regulatórios (TUST/TUSD), prêmio de risco de submercado e termos de crédito não modelados aqui. Use como referência direcional, não para precificação contratual.",
    mod_kicker: "PowerChina · Mauriti · Valor de perfil",
    mod_h1_a: "Quanto custou",
    mod_h1_b: "quando",
    mod_h1_c: "geramos cada MWh.",
    mod_lede: "Comparação entre a receita real do Mauriti (Σ MWh<sub>hora</sub> × PLD<sub>hora</sub>) e a receita \"flat\" que se obteria se a geração fosse plana ao longo do dia. A diferença é o <strong>desconto de modulação</strong> — custo do timing, não do volume.",
    mod_tracker_when: "Receita real vs receita flat por dia",
    mod_disc_month: "Desconto mês",
    mod_curr_month: "no mês corrente",
    mod_disc_month_rs: "Desconto mês (R$)",
    mod_pot_lost: "receita potencial perdida",
    mod_pld_avg: "PLD médio mês",
    mod_bignum_title: "Receita potencial perdida pela modulação",
    mod_bignum_p1a: "O Mauriti realizou",
    mod_bignum_p1b: "quando, com geração à média diária do PLD, teria realizado",
    mod_bignum_p2a: "Isso é um desconto de",
    mod_bignum_p2b: "da receita flat — preço efetivo de",
    mod_bignum_p2c: "contra um PLD médio de",
    mod_kpi_disc: "% Desconto Mauriti",
    mod_kpi_disc_ne: "% Desconto NE (frota)",
    mod_kpi_vs: "vs Benchmark NE",
    mod_kpi_vs_sub: "delta de % desconto",
    mod_kpi_eff: "Preço efetivo",
    mod_kpi_pld: "PLD médio",
    mod_s1_title: "Por que perdemos receita",
    mod_s1_tag: "PERFIL HORÁRIO TÍPICO",
    mod_s1_desc: "Dia tipico (média de cada hora ao longo do período). O pico de geração solar coincide com o vale do PLD do submercado — vendemos a maior parte do MWh quando ele vale menos.",
    mod_s2_title: "Mauriti vs frota NE solar",
    mod_s2_tag: "HISTÓRICO MENSAL",
    mod_s2_desc: "Comparação do % de desconto mês a mês. Períodos secos (PLD geralmente alto o dia todo) tendem a ter desconto menor; períodos de carga baixa (mais excedente solar no SIN) ampliam o efeito.",
    mod_s3_title: "Top dias mais doloridos",
    mod_s3_tag: "RANKING POR R$",
    mod_s3_desc: "Os dias em que o desconto absoluto (em R$) foi maior. Geralmente coincidem com PLD muito baixo no meio do dia (excesso de oferta solar no SIN).",
    mod_disclaimer: "<strong>Importante:</strong> esta análise mostra o desconto <em>teórico</em> assumindo que toda a geração é liquidada no PLD horário. O impacto real na receita do PowerChina depende do regime de comercialização de cada UFV: PPA fixo (R$/MWh) absorve o efeito mas perde valor no spot; liquidação no MCP captura o impacto integral; produtos shape com a CCEE neutralizam o desconto. Os números aqui são <strong>indicador direcional</strong> do prêmio que valeria a pena pagar por um hedge perfeito de modulação.",

    ren_kicker: "REN 1.030/2022 · Eventos ressarcíveis de curtailment",
    ren_h1_a: "Cortes que o regulador",
    ren_h1_b: "pode ressarcir.",
    ren_lede: "A REN 1.030/2022 da ANEEL estabelece que cortes classificados como <strong>REL (indisponibilidade externa)</strong> ou <strong>CNF (confiabilidade)</strong> podem ser elegíveis a ressarcimento. Esta visão lista cada evento desse tipo no complexo Mauriti — input bruto pro processo formal de pleito.",
    ren_empty: "Não foram encontrados eventos elegíveis no período analisado.",
    ren_kpi_events: "Eventos elegíveis",
    ren_kpi_mwh: "MWh elegíveis total",
    ren_pct_total: "do curtailment total",
    ren_rel_sub: "indisponibilidade externa",
    ren_cnf_sub: "confiabilidade",
    ren_kpi_days: "Dias afetados",
    ren_units_short: "UFVs",
    ren_s1_title: "Volume elegível por mês",
    ren_s1_tag: "REL + CNF",
    ren_s1_desc: "Barras empilhadas mostram como o volume ressarcível se distribui entre REL e CNF — quebra útil pra montagem do dossiê.",
    ren_s2_title: "Top causas",
    ren_s2_tag: "ORIGEM DA RESTRIÇÃO",
    ren_s2_desc: "Classificação de origem como reportada pelo ONS: <strong>LOC</strong> = local (linha de transmissão, subestação ou bay próximo da usina; há uma contraparte específica que pode ser cobrada) · <strong>SIS</strong> = sistêmico (limitação da rede regional, ex: corredor de exportação NE → SE/CO saturado; problema estrutural que afeta a região inteira). A divisão mostra quanto do pleito tem contraparte direta vs quanto é restrição sistêmica.",
    ren_s3_title: "Log de eventos",
    ren_s3_tag: "EXPORTÁVEL",
    ren_s3_desc: "Cada evento elegível identificado no período, com timestamp de início/fim, usina, razão, origem e energia cortada. Use o export CSV para processamento adicional.",
    ren_export_csv: "Baixar CSV",
    ren_events_lower: "eventos",
    ren_days_lower: "dias",
    ren_showing: "Exibindo",
    ren_of: "de",
    ren_full_csv: "Lista completa no CSV acima.",
    ren_caveat: "Importante:",
    ren_caveat_p: "esta visão identifica eventos que atendem ao teste formal de elegibilidade da REN 1.030/2022 (razões REL/CNF). O valor exato a recuperar depende do custo declarado de cada contrato (EAR/COD), das regras da Margem de Ressarcimento Atribuível (MRA), e da mecânica de descontos de CMO — informações que estão fora deste dataset. Não mostramos valor R$ aqui de propósito, pra evitar âncora enganosa. O CSV é o input pro time legal/regulatório montar o pleito formal.",
    th_start: "Início",
    th_end: "Fim",
    th_dur: "Duração",
    th_plant: "Usina",
    th_reason: "Razão",
    th_origin: "Origem",
    th_mwh: "MWh cortado",
    th_pld: "PLD médio",

    solar_kicker: "NASA POWER · Irradiância local",
    solar_h1_a: "Estamos cortando",
    solar_h1_b: "quando o sol mais brilha?",
    solar_lede: "Cruzamento entre o <strong>curtailment classificado pelo ONS</strong> (razões REL/CNF/ENE/PAR) do Mauriti e a Irradiância Horizontal Global (GHI) da NASA POWER nas coordenadas da usina. Revela se os cortes se concentram em horas de alta irradiância — quando o custo de oportunidade é maior. Subgeração não-classificada (ramp-up matinal, baseline otimista do ONS) é excluída pra o sinal refletir apenas curtailment 'verdadeiro'.",
    solar_byline_src: "Fonte",
    solar_byline_coord: "Coordenadas",
    solar_byline_lat: "Latência",
    solar_empty: "Não foi possível obter dados de irradiância da NASA POWER pra o período.",
    solar_kpi_pct_cut_sun: "% horas de sol com corte",
    solar_kpi_pct_sub: "GHI > 600 W/m²",
    solar_kpi_cf_sun: "CF em céu limpo",
    solar_kpi_cf_sub: "vs CF diário",
    solar_kpi_ghi: "GHI médio diurno",
    solar_kpi_ghi_sub: "média do período",
    solar_kpi_peak: "GHI pico",
    solar_kpi_peak_sub: "máx no período",
    solar_kpi_temp: "Temp média durante cortes",
    solar_s1_title: "CF por bin de irradiância",
    solar_s1_tag: "RELAÇÃO",
    solar_s1_desc: "Curtailment factor agrupado por bin de GHI. Se bins de alta GHI mostram CF maior, confirma que os cortes se concentram nas horas de produção mais valiosas.",
    solar_s2_title: "Hora típica: sol vs cortes",
    solar_s2_tag: "PERFIL DIURNO",
    solar_s2_desc: "Perfil médio hora-a-hora. A forma da curva de GHI e da curva de curtailment dizem se a usina está sendo cortada quando o recurso é melhor (pior cenário).",
    solar_disclaimer: "<strong>Nota:</strong> NASA POWER fornece irradiância horária derivada de reanálise em resolução 0.5° × 0.625°, suficiente pra análise de padrão mensal mas mais grosseira que medições in-situ. A irradiância real medida na planta (se disponível por piranômetros locais) daria correlações mais precisas. Dados NASA POWER têm latência de 5-7 dias, então a semana mais recente é excluída.",

    glossary_title: "Glossário e notas técnicas",
    gloss_pld: "Preço de Liquidação das Diferenças horário (R$/MWh) definido pela CCEE para cada submercado. Referência para liquidação de energia no spot brasileiro.",
    gloss_ons: "Operador Nacional do Sistema. Publica dados de constrained-off (curtailment) para usinas eólicas e fotovoltaicas.",
    gloss_ccee: "Câmara de Comercialização de Energia Elétrica. Publica o PLD horário e opera o mercado spot.",
    gloss_ne: "Submercado Nordeste do SIN (Sistema Interligado Nacional), onde está o Mauriti.",
    gloss_ren: "Resolução Normativa ANEEL 1.030/2022. Define elegibilidade de curtailment para ressarcimento. Razões REL e CNF são elegíveis; ENE (energética) e PAR (parecer de acesso) geralmente não.",
    gloss_cf: "Curtailment Factor. Energia cortada dividida pela referência (estimada), em %.",
    gloss_reasons: "Códigos de razão do corte. REL = indisponibilidade externa, CNF = confiabilidade, ENE = energética (sobre-oferta), PAR = limite por parecer de acesso.",
    gloss_origins: "Códigos de origem da restrição. <strong>LOC</strong> = local (restrição próxima da usina — linha de transmissão, subestação ou bay; um ativo de transmissão específico é responsável). <strong>SIS</strong> = sistêmico (restrição da rede regional — tipicamente saturação dos limites de exportação entre regiões ou requisitos de estabilidade; afeta várias usinas ao mesmo tempo).",
    gloss_ghi: "Global Horizontal Irradiance. Potência solar total recebida por m² em superfície horizontal. Da reanálise NASA POWER.",
    gloss_modulation_dt: "Desconto de modulação",
    gloss_modulation: "Diferença entre receita real (Σ MWh<sub>h</sub> × PLD<sub>h</sub>) e receita \"flat\" (MWh<sub>dia</sub> × PLD<sub>médio diário</sub>). Captura o custo de oportunidade de gerar em horas de PLD baixo.",

    footer_sources: "Fontes",
    footer_sources_p: "Restrição de Operação por Constrained-off (ONS, base semi-horária, detalhamento por usina + consolidado com razões). PLD horário (CCEE). Irradiância solar (NASA POWER, reanálise horária nas coordenadas da usina).",
    footer_defs: "Definições",
    footer_defs_p: "Curtailment = max(0, val_geracaoestimada − val_geracaoverificada). Receita perdida = curtailment_MWh × PLD_horário. CF = curtailment / esperada (%). Desconto modulação = (Σ MWh<sub>h</sub> × PLD<sub>h</sub>) − (Σ MWh<sub>dia</sub> × PLD<sub>medio_dia</sub>).",
    footer_bench: "Benchmark NE",
    footer_bench_p: "O benchmark de modulação agrega todas as UFVs do submercado NE como uma frota única, ponderando pela geração. Mostra o desconto que a frota NE coletivamente sofreu.",
    footer_built: "Gerado em",
    // ===== ONDA 1 =====
    fresh_ons: "Curtailment ONS",
    fresh_pld: "PLD CCEE",
    fresh_next: "Próxima atualização",
    cred_data: "Fontes de dados",
    cred_ons: "Dados Abertos",
    cred_ccee: "Portal Dados Abertos",
    cred_tech: "Stack técnico",
    cred_version: "Versão",
    cred_released: "lançada em",
    cred_changelog: "histórico de mudanças",
    // ===== ONDA 2A =====
    yoy_title: "Modulação ano a ano",
    yoy_desc: "Compara o desconto de modulação do mês corrente (parcial) com o mesmo mês parcial do ano anterior. Comparação cabeça a cabeça (mesmo número de dias decorridos) pra isolar mudanças sazonais vs. estruturais.",
    yoy_current: "Atual",
    yoy_prior: "Ano anterior",
    yoy_days: "dias",
    yoy_tag_worse: "PIOR",
    yoy_tag_better: "MELHOR",
    yoy_tag_equal: "ESTÁVEL",
    yoy_tag_no_data: "—",
    yoy_prose_worse: "O desconto de modulação de Mauriti piorou em",
    yoy_prose_worse_p: "vs. o mesmo período do ano anterior. Investigar causas estruturais (mais curtailment, mudança no mix horário) ou externas (spreads de PLD menores).",
    yoy_prose_better: "A modulação melhorou em",
    yoy_prose_better_p: "vs. o mesmo período do ano passado. Reflete spreads de PLD mais favoráveis ou melhorias operacionais.",
    yoy_prose_equal: "Modulação essencialmente estável YoY, sugerindo condições estruturais constantes.",
    itm_title: "Horas in/out of the money",
    itm_desc: "Das horas que Mauriti gerou no mês, quantas coincidiram com PLD acima da média (horas valiosas, in-the-money) vs. abaixo (horas baratas, out-of-the-money). Quantifica o custo de timing solar em base hora-a-hora.",
    itm_tag: "EXPOSIÇÃO SPOT",
    itm_ref: "PLD médio mensal",
    itm_total: "Total horas geração:",
    itm_in: "In-the-money",
    itm_out: "Out-of-the-money",
    itm_of_hours: "das horas",
    itm_of_mwh: "dos MWh capturados",
    itm_prose_negskew: "Geração solar está desproporcionalmente concentrada em horas de PLD baixo: Mauriti entregou",
    itm_prose_negskew_b: "dos MWh em apenas",
    itm_prose_negskew_c: "das horas em que os preços estavam abaixo da média. Esta é a causa estrutural do desconto de modulação.",
    itm_prose_posskew: "Mauriti está capturando mais horas valiosas que a média — posição favorável.",
    itm_prose_balanced: "A geração está razoavelmente balanceada entre horas valiosas e baratas.",
    razao_title: "Curtailment por razão · classificação ONS",
    razao_desc: "Cada hora de curtailment é classificada pelo ONS com um código de razão. REL (indisponibilidade externa) e CNF (confiabilidade) são elegíveis a ressarcimento pela REN 1.030/2022; ENE (energético) e PAR (parecer de acesso) não. Conhecer o mix é crítico pra estratégia de pleito.",
    razao_tag: "ELEGIBILIDADE REN 1.030",
    razao_th_code: "Código",
    razao_th_meaning: "Significado",
    razao_th_mwh: "MWh",
    razao_th_loss: "Perda (R$)",
    razao_th_pct: "% MWh",
    razao_th_elig: "Elegível",
    razao_yes: "Sim",
    razao_no: "Não",
    razao_total: "Total",
    razao_prose_high: "dos MWh cortados são elegíveis a ressarcimento",
    razao_prose_high_p: " pela REN 1.030/2022. Caso forte para pleitos de ressarcimento.",
    razao_prose_mid: "dos MWh cortados são potencialmente elegíveis a ressarcimento. Perfil misto — revisar caso a caso.",
    razao_prose_low: "A maior parte do curtailment é ENE/PAR (não elegível a ressarcimento). Curtailment dirigido por excesso de oferta, não por falhas externas de rede."
  }
};

// Apply language
function applyLang(lang) {
  document.documentElement.setAttribute('data-lang', lang);
  document.documentElement.setAttribute('lang', lang === 'pt' ? 'pt-br' : 'en');
  // Update buttons
  document.querySelectorAll('.lang-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.setLang === lang);
  });
  // Translate elements
  if (lang === 'en') return; // EN is the default in HTML
  const dict = I18N[lang] || {};
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (dict[key] !== undefined) {
      el.innerHTML = dict[key];
    }
  });
  // Title
  if (dict.page_title) document.title = dict.page_title;
}

// Cache original EN strings on first load (so we can switch back)
const EN_STRINGS = {};
document.querySelectorAll('[data-i18n]').forEach(el => {
  EN_STRINGS[el.getAttribute('data-i18n')] = el.innerHTML;
});

function applyLangFull(lang) {
  document.documentElement.setAttribute('data-lang', lang);
  document.documentElement.setAttribute('lang', lang === 'pt' ? 'pt-br' : 'en');
  document.querySelectorAll('.lang-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.setLang === lang);
  });
  const dict = lang === 'en' ? EN_STRINGS : (I18N[lang] || {});
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (dict[key] !== undefined) {
      el.innerHTML = dict[key];
    }
  });
  // Page title
  if (lang === 'pt' && I18N.pt && I18N.pt.page_title) {
    document.title = I18N.pt.page_title;
  } else {
    document.title = "Mauriti — Curtailment & Modulation Study";
  }
  try { localStorage.setItem('mauriti-lang', lang); } catch(e) {}
}

// Render charts on load
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
    setTimeout(() => {
      document.querySelectorAll(`.tab-pane.active [id]`).forEach(div => {
        if (window.Plotly && div._fullData) Plotly.Plots.resize(div);
      });
    }, 60);
  });
});

// Lang toggle buttons
document.querySelectorAll('.lang-btn').forEach(btn => {
  btn.addEventListener('click', () => applyLangFull(btn.dataset.setLang));
});

// Initial language: stored preference or default EN
try {
  const stored = localStorage.getItem('mauriti-lang');
  if (stored === 'pt' || stored === 'en') {
    applyLangFull(stored);
  }
} catch(e) {}

// ============================================================
// MONTHLY FORECAST (v5.6) — substitui o PPA What-If Simulator
// ============================================================
const forecastState = {
  c1: { vol: 37200, price: 116, sub: 'S' },
  c2: { active: false, vol: 0, price: 200, sub: 'S' },
};

function _fmtRs(n) {
  const sign = n < 0 ? '-' : '+';
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${sign}R$ ${(abs/1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}R$ ${(abs/1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}R$ ${(abs/1e3).toFixed(0)}k`;
  return `${sign}R$ ${abs.toFixed(0)}`;
}
function _fmtRsNoSign(n) {
  const abs = Math.abs(n);
  if (abs >= 1e9) return `R$ ${(abs/1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `R$ ${(abs/1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `R$ ${(abs/1e3).toFixed(0)}k`;
  return `R$ ${abs.toFixed(0)}`;
}
function _fmtMwh(n) {
  return n.toLocaleString('en-US', {maximumFractionDigits: 0});
}
function _fmtMwm(n) { return n.toFixed(1); }
function _pldSub(sub) {
  if (!FORECAST || !FORECAST.pld_mtd_por_sub) return 0;
  return FORECAST.pld_mtd_por_sub[sub] || 0;
}

// ===== Core forecast calculation =====
// Returns object with all financial outputs given current state.
//
// IMPORTANT — sensitivity model:
// The PLD MTD is ALREADY REALIZED for the elapsed days — it's a fact,
// not a variable. The pldRemainingMultiplier (only != 1.0 in sensitivity
// scenarios) applies ONLY to the projected remaining MWh:
//
//   Component                Realized MTD (fixed)    Remaining (variable)
//   ─────────────────────────────────────────────────────────────────
//   G × PLD_eff              G_realized × PLD_eff    G_remaining × PLD_eff × mult
//                            (uses MTD effective)    (assumes same curt %)
//   C × PLD_sub              C_share_MTD × PLD_sub   C_share_rem × PLD_sub × mult
//   Exposure × PLD_NE        Exp_MTD × PLD_NE        Exp_rem × PLD_NE × mult
//
// Contract volume is delivered FLAT across the month, so we split the
// contracted volume in same proportion: elapsed_days / total_days.
function computeForecast(pldRemainingMultiplier) {
  if (!FORECAST || FORECAST.vazio) return null;
  pldRemainingMultiplier = pldRemainingMultiplier || 1.0;
  const mult = pldRemainingMultiplier;

  // Time split
  const days_total = FORECAST.days_total;
  const days_elapsed = FORECAST.days_elapsed;
  const days_remaining = FORECAST.days_remaining;
  const frac_elapsed = days_total > 0 ? (days_elapsed / days_total) : 1;
  const frac_remaining = 1 - frac_elapsed;

  // Generation split
  const G_realized = FORECAST.mwh_realized;
  const G_remaining = FORECAST.mwh_projected_remaining;
  const G = G_realized + G_remaining;

  // PLD references (MTD = realized = fixed)
  const PLD_NE_mtd  = FORECAST.pld_medio_ne;          // NE PLD MTD (fixed)
  const PLD_eff_mtd = FORECAST.pld_efetivo_mauriti;   // Mauriti PLD eff MTD (fixed)
  // Future projections (vary with sensitivity multiplier)
  const PLD_NE_remaining = PLD_NE_mtd * mult;
  // PLD_eff future: assume same curtailment % as MTD applies going forward
  const curt_factor_implied = PLD_NE_mtd > 0
    ? (PLD_eff_mtd / PLD_NE_mtd) : 1;
  const PLD_eff_remaining = PLD_NE_remaining * curt_factor_implied;

  // Effective blended PLD for display
  const PLD_eff_blended = G > 0
    ? (G_realized * PLD_eff_mtd + G_remaining * PLD_eff_remaining) / G
    : PLD_eff_mtd;
  const PLD_NE_blended = G > 0
    ? (G_realized * PLD_NE_mtd + G_remaining * PLD_NE_remaining) / G
    : PLD_NE_mtd;

  // ===== Contracts =====
  const c1 = forecastState.c1;
  const c2 = forecastState.c2;
  const C1_vol  = c1.vol;
  const C1_price = c1.price;
  const C1_sub_pld_mtd = _pldSub(c1.sub);
  const C1_sub_pld_rem = C1_sub_pld_mtd * mult;
  // Contract volume split: flat delivery — share proportional to time elapsed
  const C1_vol_mtd = C1_vol * frac_elapsed;
  const C1_vol_rem = C1_vol * frac_remaining;
  // Blended PLD applied to contract (for display)
  const C1_pld_blended = C_total_helper(C1_vol_mtd, C1_sub_pld_mtd,
                                          C1_vol_rem, C1_sub_pld_rem);

  const C2_vol  = c2.active ? c2.vol : 0;
  const C2_price = c2.active ? c2.price : 0;
  const C2_sub_pld_mtd = c2.active ? _pldSub(c2.sub) : 0;
  const C2_sub_pld_rem = C2_sub_pld_mtd * mult;
  const C2_vol_mtd = C2_vol * frac_elapsed;
  const C2_vol_rem = C2_vol * frac_remaining;
  const C2_pld_blended = c2.active
    ? C_total_helper(C2_vol_mtd, C2_sub_pld_mtd, C2_vol_rem, C2_sub_pld_rem)
    : 0;

  // ===== Exposure split =====
  // Exposure_MTD       = G_realized × frac_realized_in_total — share of mauriti
  // Simpler: total exposure = G − (C1 + C2). Split by time fraction.
  const C_total = C1_vol + C2_vol;
  const Exposure = G - C_total;
  const Exp_mtd = Exposure * frac_elapsed;
  const Exp_rem = Exposure * frac_remaining;

  // ===== CCEE View (detailed) =====
  // Each line is sum of (realized leg) + (remaining leg with mult).
  const ccee_spot_mtd = G_realized * PLD_eff_mtd;
  const ccee_spot_rem = G_remaining * PLD_eff_remaining;
  const ccee_total_spot = ccee_spot_mtd + ccee_spot_rem;

  const ccee_c1_buy_mtd = -(C1_vol_mtd * C1_sub_pld_mtd);
  const ccee_c1_buy_rem = -(C1_vol_rem * C1_sub_pld_rem);
  const ccee_c1_buy = ccee_c1_buy_mtd + ccee_c1_buy_rem;

  const ccee_c2_buy_mtd = -(C2_vol_mtd * C2_sub_pld_mtd);
  const ccee_c2_buy_rem = -(C2_vol_rem * C2_sub_pld_rem);
  const ccee_c2_buy = ccee_c2_buy_mtd + ccee_c2_buy_rem;

  const ccee_exp_mtd = Exp_mtd * PLD_NE_mtd;
  const ccee_exp_rem = Exp_rem * PLD_NE_remaining;
  const ccee_exposure = ccee_exp_mtd + ccee_exp_rem;

  const ccee_net = ccee_total_spot + ccee_c1_buy + ccee_c2_buy + ccee_exposure;

  // ===== Commercial View =====
  // Commercial leg uses fixed contract price independent of PLD
  const comm_c1 = C1_vol * C1_price;
  const comm_c2 = C2_vol * C2_price;
  const comm_total = comm_c1 + comm_c2;

  const final_total = ccee_net + comm_total;

  // ===== Risk decomposition =====
  const PLD_sub_blended_weighted = C_total > 0
    ? ((C1_vol * C1_pld_blended) + (C2_vol * C2_pld_blended)) / C_total
    : PLD_NE_blended;
  const price_weighted = C_total > 0
    ? ((C1_vol * C1_price) + (C2_vol * C2_price)) / C_total : 0;
  const decomp_hedge      = C_total * (price_weighted - PLD_sub_blended_weighted);
  const decomp_spot_long  = Math.max(0, Exposure) * PLD_NE_blended;
  const decomp_spot_short = Math.min(0, Exposure) * PLD_NE_blended;
  const decomp_basis      = C_total * (PLD_NE_blended - PLD_sub_blended_weighted);
  const decomp_curt       = G * (PLD_eff_blended - PLD_NE_blended);

  return {
    G, G_realized, G_remaining,
    PLD_NE: PLD_NE_mtd, PLD_NE_blended,
    PLD_eff: PLD_eff_mtd, PLD_eff_blended,
    C1_vol, C1_price, C1_pld: C1_pld_blended, C1_sub: c1.sub,
    C2_vol, C2_price, C2_pld: C2_pld_blended, C2_sub: c2.sub,
    c2_active: c2.active,
    C_total, Exposure,
    ccee_total_spot, ccee_c1_buy, ccee_c2_buy, ccee_exposure, ccee_net,
    comm_c1, comm_c2, comm_total,
    final_total,
    decomp_hedge, decomp_spot_long, decomp_spot_short, decomp_basis, decomp_curt,
  };
}

// Helper: weighted-average PLD across MTD and remaining legs of a contract
function C_total_helper(vol_mtd, pld_mtd, vol_rem, pld_rem) {
  const total = vol_mtd + vol_rem;
  if (total <= 0) return 0;
  return (vol_mtd * pld_mtd + vol_rem * pld_rem) / total;
}

// ===== Render functions =====
function forecastRender() {
  if (!FORECAST || FORECAST.vazio) return;
  const lang = document.body.dataset.lang || 'en';
  const I = (lang === 'pt') ? {
    mwm: 'MW médios',
    portfolio: 'Portfólio',
    portfolio_total: 'Total contratado',
    of_forecast: 'da projeção',
    exposure: 'Exposição',
    long: '(excedente)', short: '(curta)',
    alert: 'Posição curta',
    alert_p: pre => `Você terá que comprar ${pre} MWh no spot pra honrar contratos. Aporte CCEE: `,
    pld_eff: 'Preço efetivo Mauriti',
    pld_eff_sub: '(incorpora curtailment + modulação)',
    revenue_total: 'Faturamento total previsto',
    scen_low: 'PLD −20%', scen_base: 'Base (MTD)', scen_high: 'PLD +20%',
    pld_future_label: 'PLD remanescente',
    delta_vs_base: 'vs base',
    decomp_hedge: 'Hedge dos PPAs',
    decomp_hedge_sub: 'C × (Preço − PLD do submercado)',
    decomp_long: 'Excedente vendido',
    decomp_long_sub: 'Exposição positiva × PLD NE',
    decomp_short: 'Compra de cobertura',
    decomp_short_sub: 'Exposição negativa × PLD NE',
    decomp_basis: 'Diferencial de submercado',
    decomp_basis_sub: 'C × (PLD NE − PLD do contrato)',
    decomp_curt: 'Efeito do curtailment',
    decomp_curt_sub: 'G × (PLD efetivo − PLD NE)',
    total: 'Total',
    hist_no_data: 'Coletando histórico — comparativo dia-a-dia aparece após 24h.',
    hist_today: 'Forecast hoje',
    hist_yesterday: 'Forecast ontem',
    hist_change: 'Variação',
  } : {
    mwm: 'MW avg',
    portfolio: 'Portfolio',
    portfolio_total: 'Total contracted',
    of_forecast: 'of forecast',
    exposure: 'Exposure',
    long: '(long)', short: '(short)',
    alert: 'Short position warning',
    alert_p: pre => `You will have to buy ${pre} MWh on the spot market to honor contracts. Estimated CCEE call: `,
    pld_eff: 'Mauriti effective price',
    pld_eff_sub: '(incorporates curtailment + modulation)',
    revenue_total: 'Total forecasted revenue',
    scen_low: 'PLD −20%', scen_base: 'Base (MTD)', scen_high: 'PLD +20%',
    pld_future_label: 'PLD remaining',
    delta_vs_base: 'vs base',
    decomp_hedge: 'PPA hedge gain',
    decomp_hedge_sub: 'C × (Price − PLD of submarket)',
    decomp_long: 'Spot long revenue',
    decomp_long_sub: 'Positive exposure × NE PLD',
    decomp_short: 'Spot short cover',
    decomp_short_sub: 'Negative exposure × NE PLD',
    decomp_basis: 'Submarket basis',
    decomp_basis_sub: 'C × (NE PLD − contract PLD)',
    decomp_curt: 'Curtailment penalty',
    decomp_curt_sub: 'G × (effective PLD − NE PLD)',
    total: 'Total',
    hist_no_data: 'Collecting history — daily comparison will appear after 24h.',
    hist_today: 'Forecast today',
    hist_yesterday: 'Forecast yesterday',
    hist_change: 'Change',
  };

  // ===== 1. Projection card =====
  document.getElementById('fp-realized').textContent = _fmtMwh(FORECAST.mwh_realized);
  document.getElementById('fp-daily').textContent = _fmtMwh(FORECAST.daily_avg);
  document.getElementById('fp-projected').textContent = _fmtMwh(FORECAST.mwh_projected_remaining);
  document.getElementById('fp-total').textContent = _fmtMwh(FORECAST.mwh_total_forecast);
  const mwm_total = FORECAST.mwh_total_forecast / Math.max(FORECAST.n_horas_mes, 1);
  document.getElementById('fp-mwm').textContent = _fmtMwm(mwm_total);

  // Hints for sliders (MW médios equivalente)
  const horas = FORECAST.n_horas_mes;
  const c1_mwm = horas > 0 ? (forecastState.c1.vol / horas) : 0;
  const c2_mwm = horas > 0 ? (forecastState.c2.vol / horas) : 0;
  document.getElementById('fc-c1-vol-hint').textContent =
    `≈ ${_fmtMwm(c1_mwm)} ${I.mwm}`;
  if (forecastState.c2.active) {
    document.getElementById('fc-c2-vol-hint').textContent =
      `≈ ${_fmtMwm(c2_mwm)} ${I.mwm}`;
  }

  // ===== 3. PLD reference =====
  const pldGrid = document.getElementById('forecast-pld-grid');
  if (pldGrid) {
    const subs = FORECAST.pld_mtd_por_sub || {};
    pldGrid.innerHTML = Object.entries(subs).map(([sub, pld]) =>
      `<div class="forecast-pld-cell">
        <div class="sub">${sub}</div>
        <div class="val">${Math.round(pld)}<span class="unit"> R$/MWh</span></div>
      </div>`
    ).join('');
  }
  document.getElementById('fp-pld-eff').textContent =
    `R$ ${Math.round(FORECAST.pld_efetivo_mauriti)}/MWh`;

  // ===== Core computation =====
  const fc = computeForecast(1.0);

  // ===== 2. Portfolio summary line =====
  const portfolioSum = document.getElementById('forecast-portfolio-summary');
  if (portfolioSum) {
    const pct = FORECAST.mwh_total_forecast > 0
      ? (100 * fc.C_total / FORECAST.mwh_total_forecast) : 0;
    const expClass = fc.Exposure >= 0 ? 'long' : 'short';
    portfolioSum.innerHTML =
      `<span><strong>${I.portfolio_total}:</strong> ${_fmtMwh(fc.C_total)} MWh (${pct.toFixed(0)}% ${I.of_forecast})</span>
       <span><strong>${I.exposure}:</strong> ${fc.Exposure >= 0 ? '+' : ''}${_fmtMwh(fc.Exposure)} MWh ${I[expClass]}</span>`;
  }

  // ===== 4. CCEE view detail =====
  const cceeDetail = document.getElementById('forecast-ccee-detail');
  if (cceeDetail) {
    cceeDetail.innerHTML = [
      {label: `G × PLD_eff`, val: fc.ccee_total_spot, cls: fc.ccee_total_spot >= 0 ? 'is-pos' : 'is-neg'},
      {label: `− C1 × PLD<sub>${fc.C1_sub}</sub>`, val: fc.ccee_c1_buy, cls: fc.ccee_c1_buy >= 0 ? 'is-pos' : 'is-neg'},
      ...(fc.c2_active ? [{label: `− C2 × PLD<sub>${fc.C2_sub}</sub>`, val: fc.ccee_c2_buy, cls: fc.ccee_c2_buy >= 0 ? 'is-pos' : 'is-neg'}] : []),
      {label: `${fc.Exposure >= 0 ? '+' : ''} Exp × PLD<sub>NE</sub>`, val: fc.ccee_exposure, cls: fc.ccee_exposure >= 0 ? 'is-pos' : 'is-neg'},
    ].map(r =>
      `<div class="forecast-card-row ${r.cls}">
        <span class="label">${r.label}</span>
        <span class="value">${_fmtRs(r.val)}</span>
      </div>`
    ).join('');
  }
  document.getElementById('forecast-ccee-net').textContent = _fmtRs(fc.ccee_net);

  // ===== Commercial view detail =====
  const commDetail = document.getElementById('forecast-comm-detail');
  if (commDetail) {
    let html = `<div class="forecast-card-row is-pos">
        <span class="label">C1: ${_fmtMwh(fc.C1_vol)} × R$ ${fc.C1_price}</span>
        <span class="value">${_fmtRs(fc.comm_c1)}</span></div>`;
    if (fc.c2_active) {
      html += `<div class="forecast-card-row is-pos">
        <span class="label">C2: ${_fmtMwh(fc.C2_vol)} × R$ ${fc.C2_price}</span>
        <span class="value">${_fmtRs(fc.comm_c2)}</span></div>`;
    }
    commDetail.innerHTML = html;
  }
  document.getElementById('forecast-comm-total').textContent = _fmtRs(fc.comm_total);

  // ===== Final card (semantic colour) =====
  const cardFinal = document.getElementById('forecast-card-final');
  cardFinal.classList.remove('is-pos', 'is-neg');
  cardFinal.classList.add(fc.final_total >= 0 ? 'is-pos' : 'is-neg');
  const finalFig = document.getElementById('forecast-final-figure');
  finalFig.textContent = _fmtRs(fc.final_total);
  document.getElementById('forecast-final-meta').innerHTML =
    `${I.revenue_total}<br>${I.pld_eff}: <strong>R$ ${Math.round(fc.PLD_eff_blended)}/MWh</strong>`;

  // ===== 5. Alert short position =====
  const alertEl = document.getElementById('forecast-alert');
  if (fc.Exposure < 0) {
    const shortMwh = Math.abs(fc.Exposure);
    const aporteEstimado = shortMwh * fc.PLD_NE_blended;
    alertEl.style.display = 'flex';
    document.getElementById('forecast-alert-text').innerHTML =
      I.alert_p(_fmtMwh(shortMwh)) + `<strong>${_fmtRsNoSign(aporteEstimado)}</strong>`;
  } else {
    alertEl.style.display = 'none';
  }

  // ===== 6. PLD sensitivity strip =====
  const sensGrid = document.getElementById('forecast-sens-grid');
  if (sensGrid) {
    const scenarios = [
      {key: 'low',  label: I.scen_low,  mult: 0.8,  cls: ''},
      {key: 'base', label: I.scen_base, mult: 1.0,  cls: 'is-base'},
      {key: 'high', label: I.scen_high, mult: 1.2,  cls: ''},
    ];
    const base = fc.final_total;
    sensGrid.innerHTML = scenarios.map(s => {
      const fs = computeForecast(s.mult);
      if (!fs) return '';
      const delta = fs.final_total - base;
      const deltaCls = s.mult === 1.0 ? ''
        : (delta > 0 ? 'is-up' : 'is-down');
      const pldFuture = Math.round(FORECAST.pld_medio_ne * s.mult);
      return `<div class="forecast-sens-cell ${s.cls}">
        <div class="scenario">${s.label}</div>
        <div class="pld-future">${I.pld_future_label}: <strong>R$ ${pldFuture}/MWh</strong></div>
        <div class="result">${_fmtRs(fs.final_total)}</div>
        ${s.mult !== 1.0 ? `<div class="delta ${deltaCls}">${delta >= 0 ? '+' : ''}${_fmtRsNoSign(delta)} ${I.delta_vs_base}</div>` : ''}
      </div>`;
    }).join('');
  }

  // ===== 7. Risk decomposition =====
  const decompGrid = document.getElementById('forecast-decomp-grid');
  if (decompGrid) {
    const components = [
      {label: I.decomp_hedge, sub: I.decomp_hedge_sub, val: fc.decomp_hedge},
      {label: I.decomp_long,  sub: I.decomp_long_sub,  val: fc.decomp_spot_long},
      ...(fc.decomp_spot_short < 0
        ? [{label: I.decomp_short, sub: I.decomp_short_sub, val: fc.decomp_spot_short}]
        : []),
      {label: I.decomp_basis, sub: I.decomp_basis_sub, val: fc.decomp_basis},
      {label: I.decomp_curt,  sub: I.decomp_curt_sub,  val: fc.decomp_curt},
    ];
    const maxAbs = Math.max(...components.map(c => Math.abs(c.val)));
    const renderRow = (c) => {
      const pct = maxAbs > 0 ? (100 * Math.abs(c.val) / maxAbs) : 0;
      const cls = c.val >= 0 ? 'is-pos' : 'is-neg';
      return `<div class="forecast-decomp-row">
        <span class="label">${c.label}<small>${c.sub}</small></span>
        <span class="bar-wrap"><span class="bar ${cls}" style="width:${pct}%"></span></span>
        <span class="value ${cls}">${_fmtRs(c.val)}</span>
      </div>`;
    };
    const sumDecomp = components.reduce((s, c) => s + c.val, 0);
    decompGrid.innerHTML = components.map(renderRow).join('') +
      `<div class="forecast-decomp-row is-total">
        <span class="label">${I.total}</span>
        <span></span>
        <span class="value">${_fmtRs(sumDecomp)}</span>
      </div>`;
  }

  // ===== 8. History (forecast today vs yesterday) =====
  const histContent = document.getElementById('forecast-hist-content');
  if (histContent) {
    if (!FORECAST.yesterday) {
      histContent.innerHTML = `<p class="forecast-hist-empty">${I.hist_no_data}</p>`;
    } else {
      // Comparison: forecast today's mwh_total vs yesterday's forecast.
      // We can't recompute yesterday's full revenue (we didn't save the
      // contracts setup), so we compare just the GENERATION forecast,
      // which is operationally meaningful by itself.
      const tToday = FORECAST.mwh_total_forecast;
      const tYest = FORECAST.yesterday.mwh_total_forecast;
      const dGen = tToday - tYest;
      const dGenCls = dGen >= 0 ? 'is-up' : 'is-down';
      // Compute revenue under same contract setup BUT using yesterday's
      // PLD/efetivo for comparison. Simpler: just compare forecasts of G.
      histContent.innerHTML = `
        <div class="forecast-hist-row">
          <span class="label">${I.hist_today} (generation)</span>
          <span class="val">${_fmtMwh(tToday)} MWh</span>
        </div>
        <div class="forecast-hist-row">
          <span class="label">${I.hist_yesterday} (generation)</span>
          <span class="val">${_fmtMwh(tYest)} MWh</span>
        </div>
        <div class="forecast-hist-row">
          <span class="label">${I.hist_change}</span>
          <span class="val">${dGen >= 0 ? '+' : ''}${_fmtMwh(dGen)} MWh
            <span class="delta ${dGenCls}">(${dGen >= 0 ? '+' : ''}${(100*dGen/Math.max(tYest,1)).toFixed(1)}%)</span>
          </span>
        </div>`;
    }
  }
}

function forecastInit() {
  if (!FORECAST || FORECAST.vazio) return;

  // Wire up sliders: sync range <-> number, then re-render
  function wireSlider(rangeId, numId, getter, setter) {
    const r = document.getElementById(rangeId);
    const n = document.getElementById(numId);
    if (!r || !n) return;
    r.value = getter(); n.value = getter();
    r.addEventListener('input', () => {
      n.value = r.value; setter(parseFloat(r.value)); forecastRender();
    });
    n.addEventListener('input', () => {
      const v = parseFloat(n.value) || 0;
      r.value = v; setter(v); forecastRender();
    });
  }
  wireSlider('fc-c1-vol', 'fc-c1-vol-num',
    () => forecastState.c1.vol, v => forecastState.c1.vol = v);
  wireSlider('fc-c1-price', 'fc-c1-price-num',
    () => forecastState.c1.price, v => forecastState.c1.price = v);
  wireSlider('fc-c2-vol', 'fc-c2-vol-num',
    () => forecastState.c2.vol, v => forecastState.c2.vol = v);
  wireSlider('fc-c2-price', 'fc-c2-price-num',
    () => forecastState.c2.price, v => forecastState.c2.price = v);

  document.getElementById('fc-c1-sub').addEventListener('change', e => {
    forecastState.c1.sub = e.target.value; forecastRender();
  });
  document.getElementById('fc-c2-sub').addEventListener('change', e => {
    forecastState.c2.sub = e.target.value; forecastRender();
  });

  // Toggle 2nd contract
  const addBtn = document.getElementById('forecast-add-c2');
  const removeBtn = document.getElementById('forecast-remove-c2');
  const c2Wrap = document.getElementById('fc-c2-wrap');
  if (addBtn) {
    addBtn.addEventListener('click', () => {
      forecastState.c2.active = true;
      c2Wrap.style.display = 'block';
      addBtn.style.display = 'none';
      forecastRender();
    });
  }
  if (removeBtn) {
    removeBtn.addEventListener('click', () => {
      forecastState.c2.active = false;
      c2Wrap.style.display = 'none';
      addBtn.style.display = 'inline-block';
      forecastRender();
    });
  }

  forecastRender();
}

forecastInit();
window.addEventListener('mauriti-lang-changed', forecastRender);


// ============================================================
// BENCHMARK BUILDER (Tab: bench) — multi-select + time series
// ============================================================
const benchState = {
  selectedIds: new Set(),   // peers ticados pelo usuario
  showNeFleet: true,        // toggle linha NE fleet
};

// Palette of distinct colours for selected peers in the chart
const BENCH_PALETTE = [
  '#5b6b7d',  // gray-blue
  '#9a8467',  // earth
  '#6b9080',  // sage
  '#a87a55',  // tan
  '#7a6b8e',  // muted purple
  '#5e7c8a',  // slate
  '#8a6f5d',  // brown
  '#6b8278',  // dark sage
];
function colourFor(idx) { return BENCH_PALETTE[idx % BENCH_PALETTE.length]; }

function _fmtGwh(mwh) { return (mwh / 1000).toFixed(1); }
function _fmtPct(v) { return v.toFixed(2); }

// Aggregate KPIs from a list of peer entries (weighted by gen if relevant)
function aggregatePeers(peerList) {
  if (peerList.length === 0) {
    return null;
  }
  let totalGen = 0, totalCurt = 0, totalPlants = 0;
  for (const p of peerList) {
    totalGen += p.mwh_gen;
    totalCurt += p.mwh_curt;
    totalPlants += p.n_plants;
  }
  const cf = totalGen > 0 ? (100 * totalCurt / totalGen) : 0;
  return {
    mwh_gen: totalGen, mwh_curt: totalCurt, n_plants: totalPlants, cf: cf,
    n_groups: peerList.length,
  };
}

// Build monthly aggregated time series from multiple peer entries
function aggregateMonthly(peerList) {
  const byMonth = new Map();
  for (const p of peerList) {
    if (!p.monthly || p.monthly.length === 0) continue;
    for (const m of p.monthly) {
      if (!byMonth.has(m.mes)) {
        byMonth.set(m.mes, {mes: m.mes, mwh_gen: 0, mwh_curt: 0});
      }
      const acc = byMonth.get(m.mes);
      acc.mwh_gen += m.mwh_gen;
      acc.mwh_curt += m.mwh_curt;
    }
  }
  const result = [...byMonth.values()].sort((a, b) => a.mes.localeCompare(b.mes));
  for (const r of result) {
    r.cf = r.mwh_gen > 0 ? (100 * r.mwh_curt / r.mwh_gen) : 0;
  }
  return result;
}

function benchPopulateCheckboxes() {
  const wrap = document.getElementById('bench-checkboxes');
  if (!wrap || !BENCH_KPIS || BENCH_KPIS.length === 0) return;
  // Exclui Mauriti e NE fleet (eles tem UI propria)
  const peers = BENCH_KPIS.filter(g => !g.is_mauriti && !g.is_ne_fleet);
  // Pre-seleciona os primeiros 2 peers por default
  if (benchState.selectedIds.size === 0 && peers.length > 0) {
    benchState.selectedIds.add(peers[0].id);
    if (peers.length > 1) benchState.selectedIds.add(peers[1].id);
  }
  wrap.innerHTML = peers.map((g, idx) => {
    const checked = benchState.selectedIds.has(g.id) ? 'checked' : '';
    return `
      <label class="bench-checkbox-row" data-peer-id="${g.id}">
        <input type="checkbox" value="${g.id}" ${checked}
               class="bench-peer-cb">
        <span class="bench-cb-label">
          <span><span class="bench-cb-swatch" data-cb-swatch="${g.id}"></span>
            ${g.label}</span>
          <span class="bench-cb-meta">${g.fonte} · ${g.n_plants} plants · CF ${g.cf.toFixed(1)}%</span>
        </span>
      </label>`;
  }).join('');
  // Attach handlers
  wrap.querySelectorAll('.bench-peer-cb').forEach(cb => {
    cb.addEventListener('change', () => {
      if (cb.checked) benchState.selectedIds.add(cb.value);
      else benchState.selectedIds.delete(cb.value);
      benchRender();
    });
  });
}

function _benchSetSelection(filterFn) {
  const peers = BENCH_KPIS.filter(g => !g.is_mauriti && !g.is_ne_fleet);
  benchState.selectedIds.clear();
  for (const p of peers) {
    if (filterFn(p)) benchState.selectedIds.add(p.id);
  }
  benchPopulateCheckboxes();
  benchRender();
}

function benchRender() {
  if (!BENCH_KPIS || BENCH_KPIS.length === 0) return;
  const lang = document.body.dataset.lang || 'en';
  const mauriti = BENCH_KPIS.find(g => g.is_mauriti);
  const neFleet = BENCH_KPIS.find(g => g.is_ne_fleet);
  const selectedPeers = BENCH_KPIS.filter(
    g => benchState.selectedIds.has(g.id));

  const I = (lang === 'pt')
    ? {cf:'CF curtailment',gen:'Geração',curt:'Cortado',plants:'usinas',
       gwh:'GWh',pp:'pp',avg:'média',sel:'selecionados',noSel:'Nenhum peer selecionado',
       groups:'grupos'}
    : {cf:'Curtailment CF',gen:'Generation',curt:'Curtailed',plants:'plants',
       gwh:'GWh',pp:'pp',avg:'avg',sel:'selected',noSel:'No peers selected',
       groups:'groups'};

  // ===== NE Fleet swatch update =====
  const neMeta = document.getElementById('bench-ne-meta');
  if (neMeta && neFleet) {
    neMeta.textContent = `${neFleet.n_plants} ${I.plants} · CF ${neFleet.cf.toFixed(2)}%`;
  }
  // Update color swatches for checkboxes
  let pIdx = 0;
  for (const p of BENCH_KPIS) {
    if (p.is_mauriti || p.is_ne_fleet) continue;
    const sw = document.querySelector(`[data-cb-swatch="${p.id}"]`);
    if (sw) {
      if (benchState.selectedIds.has(p.id)) {
        const colour = colourFor(pIdx);
        sw.style.background = colour;
        sw.style.border = `1px solid ${colour}`;
      } else {
        sw.style.background = 'transparent';
        sw.style.border = '1px solid var(--rule)';
      }
    }
    if (benchState.selectedIds.has(p.id)) pIdx++;
  }

  function row(key, val, unit, extraCls) {
    extraCls = extraCls || '';
    return `<div class="bench-row ${extraCls}"><span class="key">${key}</span>
            <span class="val">${val}<span class="unit">${unit||''}</span></span></div>`;
  }

  // ===== Mauriti card =====
  document.getElementById('bench-mauriti-sub').textContent =
    `${mauriti.n_plants} ${I.plants}`;
  document.getElementById('bench-mauriti-rows').innerHTML =
    row(I.cf, _fmtPct(mauriti.cf), '%') +
    row(I.gen, _fmtGwh(mauriti.mwh_gen), ' ' + I.gwh) +
    row(I.curt, _fmtGwh(mauriti.mwh_curt), ' ' + I.gwh);

  // ===== Peer aggregated card =====
  const peerLabelEl = document.getElementById('bench-peer-label');
  const peerSubEl = document.getElementById('bench-peer-sub');
  const peerRowsEl = document.getElementById('bench-peer-rows');
  const agg = aggregatePeers(selectedPeers);
  if (agg && agg.n_groups > 0) {
    if (agg.n_groups === 1) {
      peerLabelEl.textContent = selectedPeers[0].label.toUpperCase();
      peerSubEl.textContent = `${selectedPeers[0].fonte} · ${agg.n_plants} ${I.plants}`;
    } else {
      peerLabelEl.textContent = (lang === 'pt'
        ? `${agg.n_groups} GRUPOS SELECIONADOS (MÉDIA PONDERADA)`
        : `${agg.n_groups} SELECTED GROUPS (WEIGHTED AVG)`);
      peerSubEl.textContent = `${agg.n_plants} ${I.plants} · ${agg.n_groups} ${I.groups}`;
    }
    peerRowsEl.innerHTML =
      row(I.cf, _fmtPct(agg.cf), '%') +
      row(I.gen, _fmtGwh(agg.mwh_gen), ' ' + I.gwh) +
      row(I.curt, _fmtGwh(agg.mwh_curt), ' ' + I.gwh);

    // Diff card
    const dCf = mauriti.cf - agg.cf;
    const dGen = mauriti.mwh_gen - agg.mwh_gen;
    const dCurt = mauriti.mwh_curt - agg.mwh_curt;
    const cfCls = dCf > 0.5 ? 'diff-worse' : (dCf < -0.5 ? 'diff-better' : 'diff-neutral');
    const curtCls = 'diff-neutral';  // diff de curt absoluto depende da escala dos grupos
    document.getElementById('bench-diff-rows').innerHTML =
      row(I.cf, (dCf >= 0 ? '+' : '') + _fmtPct(dCf), ' ' + I.pp, cfCls) +
      row(I.gen, (dGen >= 0 ? '+' : '') + _fmtGwh(dGen), ' ' + I.gwh) +
      row(I.curt, (dCurt >= 0 ? '+' : '') + _fmtGwh(dCurt), ' ' + I.gwh, curtCls);
  } else {
    peerLabelEl.textContent = I.noSel.toUpperCase();
    peerSubEl.textContent = '';
    peerRowsEl.innerHTML = '<div class="bench-row"><span class="key">—</span></div>';
    document.getElementById('bench-diff-rows').innerHTML =
      '<div class="bench-row"><span class="key">—</span></div>';
  }

  // ===== Meta info =====
  const meta = document.getElementById('bench-meta');
  if (meta) {
    meta.textContent = (lang === 'pt'
      ? `${selectedPeers.length} peer(s) selecionado(s) · ${agg ? agg.n_plants : 0} usinas no benchmark`
      : `${selectedPeers.length} peer(s) selected · ${agg ? agg.n_plants : 0} plants in benchmark`);
  }

  // ===== Time series chart (monthly) =====
  benchRenderMonthly(mauriti, selectedPeers, neFleet, lang);

  // ===== Annual comparison bar chart =====
  benchRenderCompare(mauriti, agg, selectedPeers, lang);

  // ===== Reference table =====
  benchRenderTable(lang);
}

function benchRenderMonthly(mauriti, selectedPeers, neFleet, lang) {
  const el = document.getElementById('g_bench_monthly');
  if (!el) return;
  const traces = [];

  // Trace 1: Mauriti (sempre presente, em destaque)
  traces.push({
    type: 'scatter', mode: 'lines+markers', name: '<b>Mauriti</b>',
    x: mauriti.monthly.map(m => m.mes),
    y: mauriti.monthly.map(m => m.cf),
    line: {color: '#a8442f', width: 3.5},
    marker: {size: 9, color: '#a8442f', line: {color: '#fafaf6', width: 2}},
    hovertemplate: '<b>Mauriti</b><br>%{x}: %{y:.2f}%<extra></extra>',
  });

  // Trace 2: Selected peers (cada um com cor distinta)
  selectedPeers.forEach((peer, idx) => {
    if (!peer.monthly || peer.monthly.length === 0) return;
    const c = colourFor(idx);
    traces.push({
      type: 'scatter', mode: 'lines+markers', name: peer.label,
      x: peer.monthly.map(m => m.mes),
      y: peer.monthly.map(m => m.cf),
      line: {color: c, width: 1.6, dash: 'solid'},
      marker: {size: 6, color: c, line: {color: '#fafaf6', width: 1}},
      hovertemplate: `<b>${peer.label}</b><br>%{x}: %{y:.2f}%<extra></extra>`,
      opacity: 0.85,
    });
  });

  // Trace 3: Average of selected peers (curva agregada)
  if (selectedPeers.length >= 2) {
    const aggMonthly = aggregateMonthly(selectedPeers);
    if (aggMonthly.length > 0) {
      traces.push({
        type: 'scatter', mode: 'lines+markers',
        name: (lang === 'pt'
          ? `<b>Média dos ${selectedPeers.length} selecionados</b>`
          : `<b>Avg of ${selectedPeers.length} selected</b>`),
        x: aggMonthly.map(m => m.mes),
        y: aggMonthly.map(m => m.cf),
        line: {color: '#1a1715', width: 2.5, dash: 'dash'},
        marker: {size: 8, color: '#1a1715',
                  line: {color: '#fafaf6', width: 1.5}, symbol: 'diamond'},
        hovertemplate: (lang === 'pt'
          ? '<b>Média selecionados</b><br>%{x}: %{y:.2f}%<extra></extra>'
          : '<b>Selected avg</b><br>%{x}: %{y:.2f}%<extra></extra>'),
      });
    }
  }

  // Trace 4: NE Solar Fleet (referencia se toggled)
  if (benchState.showNeFleet && neFleet && neFleet.monthly &&
      neFleet.monthly.length > 0) {
    traces.push({
      type: 'scatter', mode: 'lines+markers',
      name: (lang === 'pt'
        ? `<b>Frota NE solar</b> (${neFleet.n_plants} UFVs)`
        : `<b>NE solar fleet</b> (${neFleet.n_plants} UFVs)`),
      x: neFleet.monthly.map(m => m.mes),
      y: neFleet.monthly.map(m => m.cf),
      line: {color: '#d92e0f', width: 2.5, dash: 'dot'},
      marker: {size: 7, color: '#d92e0f',
                line: {color: '#fafaf6', width: 1.5}, symbol: 'square'},
      hovertemplate: (lang === 'pt'
        ? `<b>Frota NE</b> (${neFleet.n_plants} UFVs)<br>%{x}: %{y:.2f}%<extra></extra>`
        : `<b>NE fleet</b> (${neFleet.n_plants} UFVs)<br>%{x}: %{y:.2f}%<extra></extra>`),
      opacity: 0.9,
    });
  }

  const layout = {
    title: {
      text: (lang === 'pt'
        ? 'Curtailment factor mensal — Mauriti vs peers selecionados'
        : 'Monthly curtailment factor — Mauriti vs selected peers'),
      font: {family: 'Fraunces, serif', size: 18, color: '#1a1a1a'},
      x: 0.02, xanchor: 'left', y: 0.96
    },
    xaxis: {
      title: '', gridcolor: '#e8e2d4',
      tickfont: {family: 'IBM Plex Mono', size: 10}
    },
    yaxis: {
      title: {text: 'CF (%)', font: {family: 'IBM Plex Mono', size: 11}},
      gridcolor: '#e8e2d4', ticksuffix: '%',
      tickfont: {family: 'IBM Plex Mono', size: 10}
    },
    margin: {l: 70, r: 30, t: 60, b: 100},
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    hovermode: 'x unified',
    legend: {orientation: 'h', x: 0.5, y: -0.18, xanchor: 'center',
             font: {family: 'IBM Plex Mono', size: 10}},
    height: 480,
  };
  Plotly.react(el, traces, layout, {
    responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d','select2d','autoScale2d','zoomIn2d','zoomOut2d']
  });
}

function benchRenderCompare(mauriti, agg, selectedPeers, lang) {
  const el = document.getElementById('g_bench_compare');
  if (!el) return;
  const labels = (lang === 'pt')
    ? ['CF curtailment (%)', 'Geração (GWh)', 'Cortado (GWh)']
    : ['Curtailment CF (%)', 'Generation (GWh)', 'Curtailed (GWh)'];

  const trace1 = {
    type: 'bar', name: 'Mauriti', x: labels,
    y: [mauriti.cf, mauriti.mwh_gen/1000, mauriti.mwh_curt/1000],
    marker: {color: '#a8442f'},
    text: [_fmtPct(mauriti.cf) + '%',
           _fmtGwh(mauriti.mwh_gen) + ' GWh',
           _fmtGwh(mauriti.mwh_curt) + ' GWh'],
    textposition: 'outside',
    textfont: {family: 'IBM Plex Mono', size: 11},
    hovertemplate: '<b>Mauriti</b><br>%{x}: %{y:.2f}<extra></extra>',
  };
  const traces = [trace1];
  if (agg) {
    const peerName = selectedPeers.length === 1
      ? selectedPeers[0].label
      : (lang === 'pt'
          ? `Média ${selectedPeers.length} selecionados`
          : `Avg of ${selectedPeers.length} selected`);
    const trace2 = {
      type: 'bar', name: peerName, x: labels,
      y: [agg.cf, agg.mwh_gen/1000, agg.mwh_curt/1000],
      marker: {color: '#5b6b7d'},
      text: [_fmtPct(agg.cf) + '%',
             _fmtGwh(agg.mwh_gen) + ' GWh',
             _fmtGwh(agg.mwh_curt) + ' GWh'],
      textposition: 'outside',
      textfont: {family: 'IBM Plex Mono', size: 11},
      hovertemplate: `<b>${peerName}</b><br>%{x}: %{y:.2f}<extra></extra>`,
    };
    traces.push(trace2);
  }
  const layout = {
    title: {
      text: (lang === 'pt'
        ? 'Totais do período — comparação direta'
        : 'Period totals — direct comparison'),
      font: {family: 'Fraunces, serif', size: 16, color: '#1a1a1a'},
      x: 0.02, xanchor: 'left', y: 0.95
    },
    barmode: 'group', bargap: 0.3, bargroupgap: 0.15,
    xaxis: {tickfont: {family: 'IBM Plex Sans', size: 12}},
    yaxis: {title: '', gridcolor: '#e8e2d4',
            tickfont: {family: 'IBM Plex Mono', size: 10}},
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: {l: 60, r: 30, t: 50, b: 80},
    legend: {orientation: 'h', x: 0.5, y: -0.18, xanchor: 'center',
             font: {family: 'IBM Plex Mono', size: 11}},
    height: 380,
  };
  Plotly.react(el, traces, layout, {
    responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d','select2d','autoScale2d','zoomIn2d','zoomOut2d']
  });
}

function benchRenderTable(lang) {
  const tbody = document.getElementById('bench-all-rows');
  if (!tbody) return;
  const rows = BENCH_KPIS.slice().sort((a, b) => {
    if (a.is_mauriti) return -1;
    if (b.is_mauriti) return 1;
    if (a.is_ne_fleet) return -1;
    if (b.is_ne_fleet) return 1;
    return b.cf - a.cf;  // peers ordenados por CF descendente
  });
  tbody.innerHTML = rows.map(g => {
    const isPeer = !g.is_mauriti && !g.is_ne_fleet;
    let cls = '';
    if (g.is_mauriti) cls = 'is-mauriti';
    else if (g.is_ne_fleet) cls = 'is-ne-fleet';
    else if (benchState.selectedIds.has(g.id)) cls = 'active-bench';
    return `
      <tr class="${cls}" data-bench-id="${g.id}"
          ${isPeer ? 'role="button"' : ''}>
        <td></td>
        <td>${g.label}</td>
        <td>${g.fonte}</td>
        <td class="num">${g.n_plants}</td>
        <td class="num">${_fmtGwh(g.mwh_gen)}</td>
        <td class="num">${_fmtGwh(g.mwh_curt)}</td>
        <td class="num">${_fmtPct(g.cf)}</td>
      </tr>`;
  }).join('');
  // Click on peer rows toggles selection
  tbody.querySelectorAll('tr[data-bench-id]').forEach(tr => {
    if (tr.classList.contains('is-mauriti') ||
        tr.classList.contains('is-ne-fleet')) return;
    tr.addEventListener('click', () => {
      const id = tr.dataset.benchId;
      if (benchState.selectedIds.has(id)) benchState.selectedIds.delete(id);
      else benchState.selectedIds.add(id);
      // Re-build checkboxes to keep them in sync
      const cb = document.querySelector(`input.bench-peer-cb[value="${id}"]`);
      if (cb) cb.checked = benchState.selectedIds.has(id);
      benchRender();
    });
  });
}

function benchInit() {
  if (!BENCH_KPIS || BENCH_KPIS.length === 0) {
    console.warn('Benchmark: no data available');
    return;
  }
  benchPopulateCheckboxes();

  // Select all / none / UFV-only buttons
  document.getElementById('bench-select-all').addEventListener('click', () => {
    _benchSetSelection(p => true);
  });
  document.getElementById('bench-select-none').addEventListener('click', () => {
    _benchSetSelection(p => false);
  });
  document.getElementById('bench-select-ufv').addEventListener('click', () => {
    _benchSetSelection(p => p.fonte === 'UFV');
  });
  // NE fleet toggle
  const neToggle = document.getElementById('bench-show-ne-fleet');
  if (neToggle) {
    neToggle.addEventListener('change', () => {
      benchState.showNeFleet = neToggle.checked;
      benchRender();
    });
  }
  benchRender();
}

benchInit();
window.addEventListener('mauriti-lang-changed', benchRender);

// =============================================================================
// ONDA 1: Toolbar handlers (print, theme, fullscreen, keyboard, changelog)
// =============================================================================

// ===== Tema (dark/light) =====
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem('mauriti-theme', theme); } catch (e) {}
}
// Carrega tema salvo (ou light por padrao)
try {
  const savedTheme = localStorage.getItem('mauriti-theme') || 'light';
  applyTheme(savedTheme);
} catch (e) { applyTheme('light'); }

document.getElementById('tb-theme').addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme') || 'light';
  applyTheme(cur === 'dark' ? 'light' : 'dark');
});

// ===== Modo apresentacao (fullscreen) =====
function togglePresentMode() {
  const html = document.documentElement;
  const cur = html.getAttribute('data-mode');
  if (cur === 'present') {
    html.removeAttribute('data-mode');
    if (document.exitFullscreen) document.exitFullscreen().catch(() => {});
  } else {
    html.setAttribute('data-mode', 'present');
    if (html.requestFullscreen) html.requestFullscreen().catch(() => {});
  }
}
document.getElementById('tb-fullscreen').addEventListener('click', togglePresentMode);

// Detecta saida do fullscreen via Esc
document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement) {
    document.documentElement.removeAttribute('data-mode');
  }
});

// ===== Print / Save PDF =====
document.getElementById('tb-print').addEventListener('click', () => {
  window.print();
});

// ===== Atalhos de teclado =====
document.addEventListener('keydown', (e) => {
  // Ignora se usuario esta digitando em input/textarea/select
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' ||
      e.target.tagName === 'SELECT' || e.ctrlKey || e.metaKey || e.altKey) {
    return;
  }
  const key = e.key.toLowerCase();
  const tabMap = {
    'c': 'curt', 'm': 'mod', 'r': 'ren', 's': 'solar', 'b': 'bench',
  };
  if (key in tabMap) {
    const tab = document.querySelector(`.tab[data-tab="${tabMap[key]}"]`);
    if (tab) { tab.click(); e.preventDefault(); }
  } else if (key === 'd') {
    document.getElementById('tb-theme').click();
    e.preventDefault();
  } else if (key === 'f') {
    document.getElementById('tb-fullscreen').click();
    e.preventDefault();
  } else if (key === 'p') {
    document.getElementById('tb-print').click();
    e.preventDefault();
  } else if (key === '?') {
    // Mostra modal com lista de atalhos
    alert(
      'Atalhos de teclado:\n\n' +
      '  C - Aba Curtailment\n' +
      '  M - Aba Modulation effect\n' +
      '  R - Aba REN 1.030 tracker\n' +
      '  S - Aba Solar resource\n' +
      '  B - Aba Benchmark\n\n' +
      '  D - Toggle Dark/Light mode\n' +
      '  F - Toggle Presentation mode\n' +
      '  P - Print / Save PDF\n' +
      '  ? - Mostra atalhos'
    );
  }
});

// ===== Changelog toggle =====
function toggleChangelog() {
  const list = document.getElementById('changelog-list');
  if (list) {
    list.style.display = list.style.display === 'none' ? 'block' : 'none';
  }
}
</script>

</body>
</html>
"""


# =============================================================================
#  RENDER
# =============================================================================

def gerar_html(mauriti: Selecao, grupos: list[Grupo], pld: pd.DataFrame,
                ne_horario: pd.DataFrame, irradiancia: pd.DataFrame,
                pld_sub: str, periodo: str, output: Path, today: date,
                cfg: dict) -> None:
    met_m = metricas(mauriti.df)
    bench_df = (pd.concat([g.df for g in grupos], ignore_index=True)
                if grupos else pd.DataFrame())
    met_b = metricas(bench_df) if not bench_df.empty else {"vazio": True,
                                                             "n_usinas": 0,
                                                             "curtailment_factor": 0}

    # ========== KPIS POR GRUPO PARA BENCHMARK BUILDER ==========
    # Calcula CF, MWh gen, MWh curt para cada grupo individual + Mauriti +
    # frota NE solar (universo). Inclui SERIE MENSAL pra graficos temporais.
    # Estrutura sera exportada como JSON pro JS interativo da aba Benchmark.
    def _slug(s: str) -> str:
        return (str(s).lower().strip().replace(" ", "_")
                .replace("/", "_").replace("-", "_")
                .replace("ç", "c").replace("ã", "a").replace("á", "a")
                .replace("é", "e").replace("í", "i").replace("ó", "o")
                .replace("ú", "u").replace("ô", "o").replace("â", "a"))

    def _serie_mensal(df_in):
        """Agrega geracao/curt/cf por mes (YYYY-MM). Retorna lista de dicts."""
        if df_in is None or df_in.empty:
            return []
        d = df_in.copy()
        d["mes"] = d["din_instante"].dt.to_period("M").astype(str)
        m = (d.groupby("mes")
              .agg(mwh_gen=("estimada_mwh", "sum"),
                   mwh_curt=("curtailment_mwh", "sum"))
              .reset_index())
        m["cf"] = (100 * m["mwh_curt"]
                    / m["mwh_gen"].replace(0, np.nan)).fillna(0)
        return [{"mes": r["mes"], "mwh_gen": float(r["mwh_gen"]),
                 "mwh_curt": float(r["mwh_curt"]), "cf": float(r["cf"])}
                for _, r in m.iterrows()]

    def _serie_mensal_ne(ne_df):
        """Agrega geracao/curt/cf por mes para a frota NE solar."""
        if ne_df is None or ne_df.empty or "mwh_estim_ne" not in ne_df.columns:
            return []
        d = ne_df.copy()
        d["mes"] = d["hora"].dt.to_period("M").astype(str)
        m = (d.groupby("mes")
              .agg(mwh_gen=("mwh_estim_ne", "sum"),
                   mwh_curt=("mwh_curt_ne", "sum"))
              .reset_index())
        m["cf"] = (100 * m["mwh_curt"]
                    / m["mwh_gen"].replace(0, np.nan)).fillna(0)
        return [{"mes": r["mes"], "mwh_gen": float(r["mwh_gen"]),
                 "mwh_curt": float(r["mwh_curt"]), "cf": float(r["cf"])}
                for _, r in m.iterrows()]

    bench_kpis_data = []
    # Mauriti como entrada destacada
    bench_kpis_data.append({
        "id": "mauriti",
        "label": "Mauriti",
        "is_mauriti": True,
        "fonte": "UFV",
        "n_plants": int(met_m.get("n_usinas", 0)),
        "mwh_gen": float(met_m.get("total_estimada_mwh", 0)),
        "mwh_curt": float(met_m.get("total_curt_mwh", 0)),
        "cf": float(met_m.get("curtailment_factor", 0)),
        "receita_perdida": float(met_m.get("receita_perdida", 0)),
        "monthly": _serie_mensal(mauriti.df),
    })
    # Cada grupo do benchmark
    for g in grupos:
        met_g = metricas(g.df)
        bench_kpis_data.append({
            "id": _slug(g.label),
            "label": g.label,
            "is_mauriti": False,
            "fonte": g.fonte,
            "n_plants": int(met_g.get("n_usinas", 0)),
            "mwh_gen": float(met_g.get("total_estimada_mwh", 0)),
            "mwh_curt": float(met_g.get("total_curt_mwh", 0)),
            "cf": float(met_g.get("curtailment_factor", 0)),
            "receita_perdida": float(met_g.get("receita_perdida", 0)),
            "monthly": _serie_mensal(g.df),
        })

    # Frota NE solar (universo) — referencia sempre disponivel
    ne_monthly = _serie_mensal_ne(ne_horario)
    ne_n_plants = (int(ne_horario["n_usinas"].max())
                    if not ne_horario.empty and "n_usinas" in ne_horario.columns
                    else 0)
    ne_total_gen = sum(m["mwh_gen"] for m in ne_monthly)
    ne_total_curt = sum(m["mwh_curt"] for m in ne_monthly)
    ne_total_cf = (100 * ne_total_curt / ne_total_gen) if ne_total_gen > 0 else 0
    ne_fleet = {
        "id": "ne_solar_fleet",
        "label": "NE solar fleet",
        "is_mauriti": False,
        "is_ne_fleet": True,  # flag especial pra UI
        "fonte": "UFV",
        "n_plants": ne_n_plants,
        "mwh_gen": ne_total_gen,
        "mwh_curt": ne_total_curt,
        "cf": ne_total_cf,
        "receita_perdida": 0,
        "monthly": ne_monthly,
    }
    print(f"  Benchmark KPIs: {len(bench_kpis_data)} grupos "
          f"(incluindo Mauriti) + NE solar fleet ({ne_n_plants} UFVs)")
    # NE fleet vai pro final da lista (UI sabe tratar diferente via flag)
    bench_kpis_data.append(ne_fleet)


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

    # ========== TENDENCIA (sparklines) ==========
    print("\n[*] Calculando tendencia 30/90/365 dias...")
    trend = calcular_tendencia(mauriti.df, today)
    if not trend.get("vazio"):
        print(f"  CF 30d:  {trend['cf_d30']:.2f}%  ({trend['mwh_d30']:,.0f} MWh)")
        print(f"  CF 90d:  {trend['cf_d90']:.2f}%  ({trend['mwh_d90']:,.0f} MWh)")
        print(f"  CF 365d: {trend['cf_d365']:.2f}% ({trend['mwh_d365']:,.0f} MWh)")
        print(f"  Tendencia: {trend['tendencia']} "
              f"(delta {trend['delta_30_vs_365']:+.2f} pp)")

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
        cur_m = diario_m[diario_m["dia"] >= cur_first].copy()
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
        cur_m = pd.DataFrame()
        cur_pct, cur_rs, cur_pld = None, None, None
    delta_pp = ((cur_pct - met_mod_ne["desconto_pct"])
                if cur_pct is not None and not met_mod_ne.get("vazio") else None)

    # ========== RESUMO MACRO MES CORRENTE + TABELA DIA-A-DIA (v5.3) ==========
    # Resumo macro: 1 linha de prosa acima do grafico mostrando totais
    mod_summary = {"vazio": True}
    mod_tabela_dias: list[dict] = []
    mod_mensal_data: list[dict] = []  # pro simulador PPA (preenchido abaixo)
    if not cur_m.empty:
        mwh_total = float(cur_m["mwh_dia"].sum())
        receita_real_total = float(cur_m["receita_real"].sum())
        receita_flat_total = float(cur_m["receita_flat"].sum())
        pld_medio_mes = float(cur_m["pld_avg"].mean())
        pld_efetivo_mes = (receita_real_total / mwh_total
                            if mwh_total > 0 else 0.0)
        n_dias = int(len(cur_m))
        # Pior dia (% mais negativo, ou seja, com pior desconto)
        idx_pior = cur_m["desconto_pct"].idxmin()
        pior = cur_m.loc[idx_pior]
        mod_summary = dict(
            vazio=False,
            n_dias=n_dias,
            mwh_total=mwh_total,
            receita_real=receita_real_total,
            receita_flat=receita_flat_total,
            desconto_rs=receita_real_total - receita_flat_total,
            desconto_pct=cur_pct or 0.0,
            pld_medio=pld_medio_mes,
            pld_efetivo=pld_efetivo_mes,
            pior_dia=pior["dia"].strftime("%d"),
            pior_pct=float(pior["desconto_pct"]),
        )
        # Tabela: 1 linha por dia (mes corrente)
        for _, r in cur_m.sort_values("dia").iterrows():
            mod_tabela_dias.append(dict(
                dia=r["dia"].strftime("%d"),
                mwh=float(r["mwh_dia"]),
                pld_medio=float(r["pld_avg"]),
                pld_efetivo=float(r["preco_efetivo"]) if pd.notna(r["preco_efetivo"]) else 0.0,
                delta_rs_mwh=float(r["preco_efetivo"] - r["pld_avg"])
                    if pd.notna(r["preco_efetivo"]) else 0.0,
                desconto_pct=float(r["desconto_pct"]) if pd.notna(r["desconto_pct"]) else 0.0,
                receita_real=float(r["receita_real"]),
            ))
        print(f"\n[*] Resumo modulacao mes corrente "
              f"({cur_first.strftime('%b/%Y')}): "
              f"{n_dias} dias, {mwh_total:,.0f} MWh, "
              f"PLD med {pld_medio_mes:.0f} vs efetivo {pld_efetivo_mes:.0f} "
              f"R$/MWh, desconto {cur_pct:.2f}%, "
              f"pior dia {pior['dia'].strftime('%d')} ({pior['desconto_pct']:.1f}%)")

    # ========== ONDA 2A.1: YoY Modulacao ==========
    yoy_data = calcular_yoy_modulacao(diario_m, today)
    if not yoy_data.get("vazio"):
        cur_p = yoy_data.get("cur_pct")
        prior_p = yoy_data.get("prior_pct")
        delta = yoy_data.get("delta_pp")
        print(f"[*] YoY modulacao: {yoy_data['label_cur']} "
              f"{cur_p:.2f}% vs {yoy_data['label_prior']} "
              f"{prior_p if prior_p is not None else 'N/A':.2f}% "
              f"-> delta {delta:+.2f} pp ({yoy_data['status']})"
              if cur_p is not None and prior_p is not None
              else f"[*] YoY modulacao: {yoy_data['label_cur']}={cur_p}, "
                   f"{yoy_data['label_prior']}={prior_p}")

    # ========== ONDA 2A.2: In/Out-the-money do PLD ==========
    itm_data = calcular_in_out_money(hor_m, pld, cur_first)
    if not itm_data.get("vazio"):
        print(f"[*] In/Out-the-money mes corrente: "
              f"PLD ref R$ {itm_data['pld_referencia']:.0f}/MWh, "
              f"ITM {itm_data['itm']['n_horas']}h ({itm_data['itm']['pct_horas']:.0f}%) "
              f"capturou {itm_data['itm']['pct_mwh']:.0f}% dos MWh, "
              f"OTM {itm_data['otm']['n_horas']}h capturou "
              f"{itm_data['otm']['pct_mwh']:.0f}% dos MWh")

    # ========== ONDA 2A.3: Breakdown REL/CNF/ENE/PAR ==========
    df_razao = calcular_curtailment_por_razao(mauriti.df, pld)
    razao_data = []
    if not df_razao.empty:
        for _, r in df_razao.iterrows():
            razao_data.append(dict(
                razao=r["razao"],
                label=r["label"],
                ressarcivel=bool(r["ressarcivel"]),
                mwh_total=float(r["mwh_total"]),
                n_eventos=int(r["n_eventos"]),
                perd_rs=float(r["perd_rs"]),
                pct_mwh=float(r["pct_mwh"]),
                pct_rs=float(r["pct_rs"]),
            ))
        total_mwh = sum(r["mwh_total"] for r in razao_data)
        total_rs = sum(r["perd_rs"] for r in razao_data)
        pct_ress_mwh = sum(r["mwh_total"] for r in razao_data
                            if r["ressarcivel"]) / total_mwh * 100 if total_mwh > 0 else 0
        print(f"[*] Curtailment por razao: {len(razao_data)} categorias, "
              f"total {total_mwh:,.0f} MWh / R$ {total_rs/1e6:.1f}M perdidos, "
              f"{pct_ress_mwh:.0f}% ressarcivel (REL+CNF)")

    # ========== AGREGADO MENSAL PARA O SIMULADOR PPA ==========
    # Para cada mes, agrega geração+receita de Mauriti + PLD de todos os
    # submercados. JS no client-side usa pra simular cenarios de PPA.
    if not diario_m.empty:
        d_m = diario_m.copy()
        d_m["mes"] = pd.to_datetime(d_m["dia"]).dt.to_period("M").dt.to_timestamp()
        # PLD por submercado mensal (vindo do attrs do PLD)
        pld_mensal_sub = pld.attrs.get("mensal_por_sub", pd.DataFrame())
        # Pra cada mes
        for mes, grp in d_m.groupby("mes"):
            mwh_total_mes = float(grp["mwh_dia"].sum())
            rev_real = float(grp["receita_real"].sum())
            rev_flat = float(grp["receita_flat"].sum())
            pld_avg = float(grp["pld_avg"].mean())  # NE medio
            pld_efetivo = (rev_real / mwh_total_mes
                            if mwh_total_mes > 0 else 0.0)
            n_dias_mes = int(len(grp))
            # PLD por submercado naquele mes
            pld_por_sub = {}
            if not pld_mensal_sub.empty:
                sub_mes = pld_mensal_sub[pld_mensal_sub["mes"] == mes]
                for _, r in sub_mes.iterrows():
                    pld_por_sub[r["sub_code"]] = {
                        "mean": round(float(r["mean_pld"]), 2),
                        "sum": round(float(r["sum_pld"]), 0),
                        "n_horas": int(r["n_horas"]),
                    }
            # n_horas do mes (pra calculos de PPA flat)
            n_horas_mes = (pld_por_sub.get("NE", {}).get("n_horas")
                            or pld_por_sub.get("SECO", {}).get("n_horas")
                            or (n_dias_mes * 24))
            mod_mensal_data.append(dict(
                month=mes.strftime("%Y-%m"),
                month_label=mes.strftime("%b/%y"),
                mwh_total=round(mwh_total_mes, 1),
                receita_real=round(rev_real, 0),
                receita_flat=round(rev_flat, 0),
                pld_medio_ne=round(pld_avg, 2),
                pld_efetivo=round(pld_efetivo, 2),
                n_dias=n_dias_mes,
                n_horas_mes=int(n_horas_mes),
                pld_por_sub=pld_por_sub,
            ))
        if mod_mensal_data:
            print(f"\n[*] Agregado mensal para simulador PPA: "
                  f"{len(mod_mensal_data)} meses, submercados disponiveis: "
                  f"{sorted(set(s for m in mod_mensal_data for s in m['pld_por_sub']))}")

    # ========== MONTHLY FORECAST DATA (v5.6) ==========
    # Substitui o simulador PPA historico por uma projecao do mes corrente:
    # geracao projetada (MTD + media_diaria * dias_restantes), PLD MTD por
    # submercado (pra contratos), PLD efetivo Mauriti, e snapshot do dia
    # salvo em forecast_history.json pra comparacao dia-a-dia.
    forecast_data = {"vazio": True}
    if not cur_m.empty:
        # Dias no mes total e ja decorridos
        if cur_first.month == 12:
            next_first = date(cur_first.year + 1, 1, 1)
        else:
            next_first = date(cur_first.year, cur_first.month + 1, 1)
        days_total_mes = (next_first - cur_first).days
        days_elapsed = int(len(cur_m))
        days_remaining = max(days_total_mes - days_elapsed, 0)

        mwh_realized = float(cur_m["mwh_dia"].sum())
        daily_avg = mwh_realized / max(days_elapsed, 1)
        mwh_projected_remaining = daily_avg * days_remaining
        mwh_total_forecast = mwh_realized + mwh_projected_remaining

        # PLD MTD por submercado (filtra PLD raw pelo mes corrente)
        pld_mtd_por_sub = {}
        if not pld.empty:
            pld_mes = pld[(pld["hora"] >= pd.Timestamp(cur_first)) &
                            (pld["hora"] < pd.Timestamp(today + timedelta(days=1)))].copy()
            if not pld_mes.empty and "sub" in pld_mes.columns:
                # Normalizacao consistente com selecionar_grupos
                sub_map = {
                    "NORDESTE": "NE", "NE": "NE",
                    "SUDESTE/CENTROOESTE": "SECO", "SUDESTE": "SECO",
                    "SECO": "SECO", "SE/CO": "SECO", "SE": "SECO",
                    "SUL": "S", "S": "S",
                    "NORTE": "N", "N": "N",
                }
                pld_mes["sub_norm"] = (pld_mes["sub"].astype(str)
                                            .str.upper().str.strip())
                pld_mes["sub_code"] = (pld_mes["sub_norm"].map(sub_map)
                                            .fillna(pld_mes["sub_norm"]))
                for sub_code, grp in pld_mes.groupby("sub_code"):
                    pld_mtd_por_sub[str(sub_code)] = round(
                        float(grp["pld"].mean()), 2)
        # Fallback se PLD nao tiver: usar PLD do mod_mensal_data do mes
        if not pld_mtd_por_sub and mod_mensal_data:
            mes_str = cur_first.strftime("%Y-%m")
            for m in mod_mensal_data:
                if m["month"] == mes_str:
                    for sub, info in m.get("pld_por_sub", {}).items():
                        pld_mtd_por_sub[sub] = info.get("mean", 0)
                    break

        # PLD efetivo Mauriti no mes (ja calculado em mod_summary)
        pld_efetivo_mauriti = float(pld_efetivo_mes) if mwh_total > 0 else 0.0
        pld_medio_ne = pld_mtd_por_sub.get("NE", float(pld_medio_mes))

        # MW medios equivalente (pro hint dos sliders no JS)
        n_horas_mes = days_total_mes * 24

        forecast_data = dict(
            vazio=False,
            cur_month=cur_first.strftime("%Y-%m"),
            cur_month_label=cur_first.strftime("%B/%Y"),
            days_total=days_total_mes,
            days_elapsed=days_elapsed,
            days_remaining=days_remaining,
            n_horas_mes=n_horas_mes,
            mwh_realized=round(mwh_realized, 1),
            daily_avg=round(daily_avg, 1),
            mwh_projected_remaining=round(mwh_projected_remaining, 1),
            mwh_total_forecast=round(mwh_total_forecast, 1),
            pld_mtd_por_sub=pld_mtd_por_sub,
            pld_efetivo_mauriti=round(pld_efetivo_mauriti, 2),
            pld_medio_ne=round(pld_medio_ne, 2),
            today_str=today.strftime("%Y-%m-%d"),
        )
        print(f"\n[*] Monthly forecast: G_realizada={mwh_realized:,.0f} MWh "
              f"({days_elapsed}/{days_total_mes} dias) "
              f"-> G_projetada={mwh_total_forecast:,.0f} MWh")
        print(f"    PLD MTD por sub: {pld_mtd_por_sub}, "
              f"PLD_eff Mauriti: {pld_efetivo_mauriti:.0f} R$/MWh")

        # ===== Persistencia historica do forecast =====
        # Salva snapshot do dia em forecast_history.json (append). Janela
        # de 60 dias - suficiente pra comparar com ontem/semana/mes anterior.
        hist_path = Path(cfg["output_html"]).parent / "forecast_history.json"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        history_list = []
        if hist_path.exists():
            try:
                with open(hist_path, "r", encoding="utf-8") as f:
                    history_list = json.load(f)
            except Exception:
                history_list = []
        # Remove qualquer snapshot anterior do MESMO dia (evita duplicacao)
        history_list = [h for h in history_list
                          if h.get("today_str") != forecast_data["today_str"]]
        # Adiciona snapshot de hoje (apenas campos relevantes pra comparacao)
        history_list.append({
            "today_str": forecast_data["today_str"],
            "cur_month": forecast_data["cur_month"],
            "mwh_realized": forecast_data["mwh_realized"],
            "mwh_total_forecast": forecast_data["mwh_total_forecast"],
            "daily_avg": forecast_data["daily_avg"],
            "pld_efetivo_mauriti": forecast_data["pld_efetivo_mauriti"],
            "pld_mtd_por_sub": forecast_data["pld_mtd_por_sub"],
        })
        # Mantem ultimos 60 dias
        history_list = sorted(history_list,
                                key=lambda h: h["today_str"])[-60:]
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history_list, f, ensure_ascii=False, indent=2)
        print(f"    forecast_history.json: {len(history_list)} snapshots "
              f"({history_list[0]['today_str']} a {history_list[-1]['today_str']})")
        # Snapshot de ontem (se existir) pra exibir card "vs yesterday"
        yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        snap_yesterday = next((h for h in history_list
                                  if h["today_str"] == yesterday_str), None)
        forecast_data["yesterday"] = snap_yesterday  # pode ser None

    # ========== REN 1.030 ==========
    print("\n[*] Identificando eventos elegiveis REN 1.030/2022...")
    eventos = eventos_elegiveis_ren1030(mauriti.df)
    met_ren = metricas_ren1030(eventos, mauriti.df)
    if not met_ren.get("vazio"):
        print(f"  Eventos elegiveis: {met_ren['n_eventos']} "
              f"({met_ren['n_eventos_rel']} REL + "
              f"{met_ren['n_eventos_cnf']} CNF)")
        print(f"  MWh elegivel total: {met_ren['mwh_total']:,.1f} "
              f"({met_ren['pct_total']:.1f}% do curtailment total)")
        print(f"  Dias afetados: {met_ren['n_dias']}")
        # Salva CSV
        csv_out = Path(cfg["output_csv_ren1030"]).resolve()
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        eventos_csv = eventos.copy()
        # eventos tem colunas: data_inicio_str, hora_inicio, data_fim_str,
        # hora_fim, duracao_h, usina, razao, origem (cod), origem_label,
        # mwh_cortado, pld_medio
        eventos_csv.columns = [
            "date_start", "time_start", "date_end", "time_end",
            "duration_h", "plant", "reason",
            "origin_code", "origin_label",
            "mwh_curtailed", "pld_avg_rs_mwh"
        ]
        eventos_csv.to_csv(csv_out, index=False, encoding="utf-8",
                            float_format="%.3f")
        print(f"  CSV salvo em: {csv_out}")

    # Top 50 eventos pra exibir na tabela HTML (full vai pro CSV)
    eventos_top = (eventos.head(50).to_dict(orient="records")
                    if not eventos.empty else [])

    # ========== IRRADIANCIA ==========
    print("\n[*] Cruzando curtailment x GHI NASA POWER...")
    print("    (curtailment filtrado: apenas razoes ONS REL/CNF/ENE/PAR)")
    cruz_irr = cruzar_irradiancia(mauriti.df, irradiancia)
    met_irr = metricas_irradiancia(cruz_irr)
    if not met_irr.get("vazio"):
        print(f"  GHI medio diurno: {met_irr['ghi_medio_diurno']:.0f} W/m2")
        print(f"  Horas em ceu limpo (GHI > 600 W/m2): "
              f"{met_irr['n_horas_pleno']}")
        print(f"  % com corte: {met_irr['pct_horas_corte_em_sol']:.1f}%")
        print(f"  CF em ceu limpo: {met_irr['cf_em_sol_pleno']:.2f}%")

    # ========== DIAGNOSTICO HORARIO (DEBUG) ==========
    # Imprime tabela hora_dia x (n_REL, n_CNF, n_ENE, sum_curt_filt, sum_curt_bruto, ghi)
    print("\n[DEBUG] Distribuicao horaria de curtailment vs irradiancia:")
    print("       hora_dia | n_REL+CNF+ENE+PAR | curt_filt(MWh) | curt_bruto(MWh) | ghi_avg(W/m2)")
    dfd = mauriti.df.copy()
    dfd["hora_dia"] = dfd["din_instante"].dt.hour
    razoes_oficiais_dbg = ["REL", "CNF", "ENE", "PAR"]
    dfd["e_classificado"] = dfd["cod_razaorestricao"].isin(razoes_oficiais_dbg)
    dfd["curt_filt"] = dfd["curtailment_mwh"].where(dfd["e_classificado"], 0.0)
    # GHI por hora_dia (perfil tipico)
    ghi_perfil = {}
    if not irradiancia.empty:
        irr_d = irradiancia.copy()
        irr_d["hora_dia"] = irr_d["hora"].dt.hour
        ghi_perfil = irr_d.groupby("hora_dia")["ghi"].mean().to_dict()
    for h in range(24):
        sub = dfd[dfd["hora_dia"] == h]
        n_class = int(sub["e_classificado"].sum())
        sum_filt = float(sub["curt_filt"].sum())
        sum_bruto = float(sub["curtailment_mwh"].sum())
        ghi = ghi_perfil.get(h, 0.0)
        print(f"         {h:>2}h    | {n_class:>17,} | {sum_filt:>14,.0f} | "
              f"{sum_bruto:>15,.0f} | {ghi:>13.0f}")
    # Sample din_instante pra verificar timezone
    if not dfd.empty:
        amostras = dfd["din_instante"].sample(min(3, len(dfd)), random_state=42)
        print(f"\n[DEBUG] Amostra din_instante: {amostras.tolist()}")
        print(f"[DEBUG] din_instante dtype: {dfd['din_instante'].dtype}")
    if not irradiancia.empty:
        amos_irr = irradiancia["hora"].sample(min(3, len(irradiancia)), random_state=42)
        print(f"[DEBUG] Amostra NASA hora (apos UTC->BRT): {amos_irr.tolist()}")

    # ========== HEATMAP DOW ==========
    cf_dow = heatmap_dow_hora(mauriti.df)

    # ========== FIGURAS ==========
    figs: dict[str, Any] = {"tracker": json.loads(pio.to_json(fig_tracker))}
    if not met_m.get("vazio"):
        figs["serie"]       = json.loads(pio.to_json(g_serie(mauriti.df,
            "Daily generation — estimated vs realised")))
        figs["donut_razao"] = json.loads(pio.to_json(g_donut_razao(mauriti.df,
            "Reasons for the cuts")))
        figs["razoes_mes"]  = json.loads(pio.to_json(g_razoes_mes(mauriti.df,
            "Reasons by month")))
        figs["heatmap"]     = json.loads(pio.to_json(g_heatmap_horario(mauriti.df)))
        if not cf_dow.empty:
            figs["heatmap_dow"] = json.loads(pio.to_json(
                g_heatmap_dow_hora(cf_dow)))
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

    # Charts REN 1.030
    if not met_ren.get("vazio"):
        figs["ren_mensal"] = json.loads(pio.to_json(g_ren_mensal(eventos)))
        figs["ren_origem"] = json.loads(pio.to_json(g_ren_origem(eventos)))

    # Charts solar resource
    if not met_irr.get("vazio"):
        figs["irr_scatter"] = json.loads(pio.to_json(g_irrad_scatter(cruz_irr)))
        figs["irr_perfil"]  = json.loads(pio.to_json(g_irrad_perfil(cruz_irr)))

    # ===== ONDA 2A.3: Donut chart curtailment por razao =====
    if not df_razao.empty:
        figs["razao_donut"] = json.loads(pio.to_json(
            g_donut_curtailment_razao(df_razao)))

    grupos_str = ", ".join([g.label for g in grupos]) if grupos else "—"
    pld_fallback = bool(pld.attrs.get("fallback", False))

    # ===== ONDA 1: Freshness data =====
    # Calcula datas dos ultimos dados disponiveis vs hoje
    # Status: ok (< 2 dias) / stale (2-5 dias) / old (> 5 dias)
    def _freshness_status(last_date, today_date):
        if last_date is None:
            return "old"
        delta_days = (today_date - last_date).days
        if delta_days <= 2:
            return "ok"
        elif delta_days <= 5:
            return "stale"
        return "old"

    # Ultimo dia ONS curtailment (do mauriti.df)
    ons_last_date = None
    try:
        if not mauriti.df.empty:
            ons_last_date = mauriti.df["din_instante"].max().date()
    except Exception:
        pass
    # Ultimo dia PLD CCEE
    pld_last_date = None
    try:
        if not pld.empty:
            pld_last_date = pld["hora"].max().date()
    except Exception:
        pass
    # Proxima execucao: 9:15 BRT do dia seguinte (= 12:15 UTC)
    proxima = today + timedelta(days=1)
    freshness_data = dict(
        ons_last=ons_last_date.strftime("%d/%m") if ons_last_date else "—",
        ons_status=_freshness_status(ons_last_date, today),
        pld_last=pld_last_date.strftime("%d/%m") if pld_last_date else "—",
        pld_status=_freshness_status(pld_last_date, today),
        next_run=proxima.strftime("%d/%m") + " 09:15 BRT",
    )

    html = Template(HTML_TEMPLATE).render(
        met_m=met_m, met_b=met_b,
        n_grupos=len(grupos), grupos_str=grupos_str,
        submercado=pld_sub, periodo=periodo,
        gerado_em=today.strftime("%d/%m/%Y"),
        dash_version=DASH_VERSION,
        dash_version_date=DASH_VERSION_DATE,
        dash_changes=DASH_CHANGES,
        freshness=freshness_data,
        yoy_modulation=yoy_data,
        itm_otm=itm_data,
        razao_breakdown=razao_data,
        figs_json=json.dumps(figs),
        insights_m=_gera_insights_mauriti(mauriti.df, met_m),
        insights_c=_gera_insights_comp(met_m, met_b),
        pld_fallback=pld_fallback,
        mauriti_lat=cfg["mauriti_lat"],
        mauriti_lon=cfg["mauriti_lon"],
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
        mod_summary=mod_summary,
        mod_tabela_dias=mod_tabela_dias,
        mod_mensal_json=json.dumps(mod_mensal_data, default=str),
        forecast_data=forecast_data,
        forecast_json=json.dumps(forecast_data, default=str),
        bench_kpis_json=json.dumps(bench_kpis_data, default=str),
        trend=trend,
        met_ren=met_ren,
        eventos_top=eventos_top,
        met_irr=met_irr,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    # ========== EXPORT JSON DE RESUMO DIARIO (pra envio por email) ==========
    # Este JSON eh consumido pelo enviar_email.py no workflow do GitHub Actions.
    # Salva em ./public/daily_summary.json (publicado junto com o dashboard).
    try:
        df_d = mauriti.df.copy()
        df_d["dia"] = df_d["din_instante"].dt.date
        df_d = df_d.dropna(subset=["dia"])
        most_recent_day = max(df_d["dia"]) if not df_d.empty else None
        if most_recent_day is not None:
            sub = df_d[df_d["dia"] == most_recent_day]
            mwh_curt = float(sub["curtailment_mwh"].sum())
            mwh_estim = float(sub["estimada_mwh"].sum())
            mwh_gen = float(sub["geracao_mwh"].sum())
            cf_pct = (100 * mwh_curt / mwh_estim) if mwh_estim > 0 else 0.0
            # Modulacao do dia (via diario_m)
            dia_ts = pd.Timestamp(most_recent_day)
            row_m = diario_m[diario_m["dia"] == dia_ts] if not diario_m.empty else pd.DataFrame()
            if not row_m.empty:
                r = row_m.iloc[0]
                pld_avg_day = float(r["pld_avg"])
                pld_eff_day = (float(r["preco_efetivo"])
                                 if pd.notna(r["preco_efetivo"]) else 0.0)
                mod_pct_day = (float(r["desconto_pct"])
                                if pd.notna(r["desconto_pct"]) else 0.0)
                mod_rs_day = (float(r["receita_real"] - r["receita_flat"])
                                if pd.notna(r["receita_real"]) else 0.0)
            else:
                pld_avg_day = pld_eff_day = mod_pct_day = mod_rs_day = 0.0
            # Pior hora do dia
            sub2 = sub.copy()
            sub2["hora_dia"] = sub2["din_instante"].dt.hour
            by_hour = (sub2.groupby("hora_dia")
                         .agg(mwh=("curtailment_mwh", "sum"),
                               pld=("pld", "mean"))
                         .reset_index())
            if not by_hour.empty and by_hour["mwh"].max() > 0.01:
                w = by_hour.loc[by_hour["mwh"].idxmax()]
                worst_hour = {"hour": int(w["hora_dia"]),
                                "mwh": round(float(w["mwh"]), 1),
                                "pld": round(float(w["pld"]), 2)}
            else:
                worst_hour = None
            # REN events do dia
            if not eventos.empty:
                eventos_dia = eventos[
                    pd.to_datetime(eventos["data_inicio_str"]).dt.date == most_recent_day
                ]
                n_ren = int(len(eventos_dia))
                ren_mwh_day = float(eventos_dia["mwh_cortado"].sum())
            else:
                n_ren = 0
                ren_mwh_day = 0.0

            daily_summary = {
                "generated_at": today.isoformat(),
                "period": periodo,
                "submarket": pld_sub,
                "dashboard_url": "https://rtbarbosa3.github.io/mauriti-curtailment/",
                "csv_url": ("https://rtbarbosa3.github.io/"
                             "mauriti-curtailment/eventos_elegiveis_ren1030.csv"),
                "most_recent_day": {
                    "date": most_recent_day.isoformat(),
                    "date_br": most_recent_day.strftime("%d/%m/%Y"),
                    "cf_pct": round(cf_pct, 2),
                    "mwh_curtailed": round(mwh_curt, 1),
                    "mwh_estimated": round(mwh_estim, 1),
                    "mwh_generated": round(mwh_gen, 1),
                    "pld_avg_rs_mwh": round(pld_avg_day, 1),
                    "pld_effective_rs_mwh": round(pld_eff_day, 1),
                    "modulation_disc_pct": round(mod_pct_day, 2),
                    "modulation_disc_rs": round(mod_rs_day, 0),
                    "ren_events": n_ren,
                    "ren_mwh_eligible": round(ren_mwh_day, 1),
                    "worst_hour": worst_hour,
                },
                "trend_30_90_365": {
                    "direction": trend.get("tendencia", "indefinida")
                        if not trend.get("vazio") else "n/a",
                    "cf_30d_pct": round(trend.get("cf_d30", 0), 2),
                    "cf_90d_pct": round(trend.get("cf_d90", 0), 2),
                    "cf_365d_pct": round(trend.get("cf_d365", 0), 2),
                    "delta_30_vs_365_pp": round(trend.get("delta_30_vs_365", 0), 2),
                },
                "month_to_date": mod_summary if not mod_summary.get("vazio") else None,
                "vs_peers": {
                    "n_peer_groups": len(grupos),
                    "mauriti_cf_pct": round(met_m.get("curtailment_factor", 0), 2),
                    "peers_cf_pct": round(met_b.get("curtailment_factor", 0), 2)
                        if not met_b.get("vazio") else None,
                    "delta_pp": round(
                        met_m.get("curtailment_factor", 0)
                        - met_b.get("curtailment_factor", 0), 2)
                        if not met_b.get("vazio") else None,
                },
                "vs_ne_fleet_modulation": {
                    "mauriti_disc_pct": round(met_mod_m.get("desconto_pct", 0), 2)
                        if not met_mod_m.get("vazio") else None,
                    "ne_fleet_disc_pct": round(met_mod_ne.get("desconto_pct", 0), 2)
                        if not met_mod_ne.get("vazio") else None,
                },
                "totals_period": {
                    "mwh_curtailed": round(met_m.get("total_curt_mwh", 0), 0),
                    "lost_revenue_rs": round(met_m.get("receita_perdida", 0), 0),
                    "cf_pct": round(met_m.get("curtailment_factor", 0), 2),
                    "ren_eligible_events": int(met_ren.get("n_eventos", 0))
                        if not met_ren.get("vazio") else 0,
                    "ren_eligible_mwh": round(met_ren.get("mwh_total", 0), 0)
                        if not met_ren.get("vazio") else 0,
                },
            }
            json_path = output.parent / "daily_summary.json"
            json_path.write_text(
                json.dumps(daily_summary, indent=2, ensure_ascii=False,
                            default=str),
                encoding="utf-8")
            print(f"\n[OK] Resumo diario salvo em: {json_path}")
            print(f"     Ultimo dia com dados: {most_recent_day} "
                  f"(CF {cf_pct:.2f}%, {mwh_curt:.0f} MWh cortados)")
    except Exception as e:
        print(f"\n[!] Erro ao gerar daily_summary.json: "
              f"{e.__class__.__name__}: {e}")


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    print("=" * 78)
    print(" CURTAILMENT MAURITI - dashboard ONS+CCEE+NASA  (v5) ")
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
    # NASA POWER irradiancia - falhas nao quebram o pipeline
    try:
        irradiancia = carregar_irradiancia_nasa(cfg, dt_ini, dt_fim)
    except Exception as e:
        print(f"  [!] NASA POWER falhou: {e.__class__.__name__}: {e}")
        irradiancia = pd.DataFrame(columns=["hora", "ghi", "temp"])

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
    gerar_html(mauriti, grupos, pld, ne_horario, irradiancia,
                cfg["submercado"], periodo, out, date.today(), cfg)
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
