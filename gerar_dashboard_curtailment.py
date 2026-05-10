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

    # Se cache existe e foi atualizado hoje, le dele
    if cache_file.exists():
        try:
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime).date()
            if mtime == date.today():
                df = pd.read_csv(cache_file)
                df["hora"] = pd.to_datetime(df["hora"])
                df = df[(df["hora"] >= pd.Timestamp(dt_ini)) &
                        (df["hora"] <= pd.Timestamp(dt_fim) + pd.Timedelta(days=1))]
                print(f"  [cache hoje] {len(df):,} horas")
                return df
        except Exception:
            pass

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
                ghi_v = float(ghi_val) if ghi_val not in (None, -999, "-999") else None
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
    df = df.dropna(subset=["ghi"])
    df = df.drop_duplicates("hora").sort_values("hora")
    # Salva cache
    try:
        df.to_csv(cache_file, index=False)
    except Exception:
        pass
    print(f"  -> {len(df):,} horas, GHI medio diurno "
          f"{df[df['ghi'] > 50]['ghi'].mean():.0f} W/m2")
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
#  REN 1.030/2022 - eventos elegiveis
# =============================================================================

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
    # Curtailment "classificado" = so o que tem razao ONS oficial
    razoes_oficiais = ["REL", "CNF", "ENE", "PAR"]
    df["curt_classificado_mwh"] = df["curtailment_mwh"].where(
        df["cod_razaorestricao"].isin(razoes_oficiais), 0.0)
    agg = df.groupby("hora").agg(
        mwh_gen=("geracao_mwh", "sum"),
        mwh_curt=("curt_classificado_mwh", "sum"),
        mwh_estim=("estimada_mwh", "sum"),
    ).reset_index()
    out = agg.merge(irradiancia[["hora", "ghi", "temp"]], on="hora", how="inner")
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
    fig = go.Figure(go.Bar(
        y=by_origem.index, x=by_origem["mwh"], orientation="h",
        marker=dict(color=EL["accent_2"]),
        text=[f"{v:,.0f} MWh · {n} events" for v, n in
              zip(by_origem["mwh"], by_origem["n"])],
        textposition="outside",
        textfont=dict(color=EL["ink_2"], size=10,
                       family="'IBM Plex Mono',monospace"),
        hovertemplate="<b>%{y}</b><br>%{x:,.1f} MWh<extra></extra>",
    ))
    lay = dict(LAY); lay.update(
        title=ed_title("Top sources of eligible curtailment", 20),
        xaxis=dict(title="MWh", gridcolor=EL["border"]),
        yaxis=dict(gridcolor=EL["border"],
                    tickfont=dict(family="'IBM Plex Mono',monospace",
                                   size=11, color=EL["ink"])),
        height=max(300, 38 * len(by_origem) + 80),
        margin=dict(l=260, r=180, t=60, b=50),  # label mais larga
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
    x_lab = [dow_labels[i] for i in cf_pivot.columns]
    fig = go.Figure(go.Heatmap(
        z=cf_pivot.values, x=x_lab, y=cf_pivot.index,
        colorscale=[[0.0, EL["bg"]], [0.05, "#f1e8d8"], [0.20, "#e6c7a8"],
                     [0.45, EL["accent_light"]], [0.75, EL["accent"]],
                     [1.0, "#5e1d10"]],
        colorbar=dict(title=dict(text="CF%", font=dict(size=11)),
                       outlinewidth=0, ticksuffix="%",
                       tickfont=dict(family="'IBM Plex Mono'", size=10)),
        zmax=min(80, float(cf_pivot.values.max()) if cf_pivot.size > 0 else 80),
        zmin=0,
        hovertemplate="<b>%{x} %{y}h</b><br>CF: %{z:.1f}%<extra></extra>",
    ))
    lay = dict(LAY); lay.update(
        title=ed_title("Weekday × hour heatmap — weekly pattern", 20),
        yaxis=dict(title="Hour of day", dtick=2, autorange="reversed",
                    gridcolor=EL["border"]),
        xaxis=dict(title="", gridcolor=EL["border"]),
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

/* Trend sparkline strip */
.trend-strip{display:grid;grid-template-columns:repeat(3,1fr);
  gap:24px;margin:0 0 48px;padding:24px 28px;
  background:var(--bg-alt);border:1px solid var(--rule);border-radius:2px}
.trend-cell{display:flex;flex-direction:column;gap:6px}
.trend-cell .lbl{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);letter-spacing:0.18em;text-transform:uppercase}
.trend-cell .val{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:32px;line-height:1;letter-spacing:-0.01em;
  font-variation-settings:"opsz" 72}
.trend-cell .val .unit{font-family:'IBM Plex Sans',sans-serif;font-size:13px;
  color:var(--muted);font-weight:400;margin-left:4px}
.trend-cell .delta{font-family:'IBM Plex Mono',monospace;font-size:11px;
  letter-spacing:0.04em}
.trend-cell .delta.up{color:var(--accent-today);font-weight:500}
.trend-cell .delta.down{color:var(--ok);font-weight:500}
.trend-cell .delta.flat{color:var(--muted)}
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

<!-- Language toggle -->
<div class="lang-toggle">
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
  </div>

  <div class="tabs">
    <button class="tab active" data-tab="curt" data-i18n="tab_curt">Curtailment</button>
    <button class="tab" data-tab="mod" data-i18n="tab_mod">Modulation effect</button>
    <button class="tab" data-tab="ren" data-i18n="tab_ren">REN 1.030 tracker</button>
    <button class="tab" data-tab="solar" data-i18n="tab_solar">Solar resource</button>
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
      <div class="trend-cell">
        <div class="lbl" data-i18n="trend_30d">Last 30 days</div>
        <div class="val">{{ "%.2f"|format(trend.cf_d30) }}<span class="unit">%</span></div>
        <div class="delta">CF · {{ "%.0f"|format(trend.mwh_d30/1000) }} GWh <span data-i18n="trend_curtailed">curtailed</span></div>
      </div>
      <div class="trend-cell">
        <div class="lbl" data-i18n="trend_90d">Last 90 days</div>
        <div class="val">{{ "%.2f"|format(trend.cf_d90) }}<span class="unit">%</span></div>
        <div class="delta">CF · {{ "%.0f"|format(trend.mwh_d90/1000) }} GWh <span data-i18n="trend_curtailed">curtailed</span></div>
      </div>
      <div class="trend-cell">
        <div class="lbl" data-i18n="trend_365d">Last 365 days</div>
        <div class="val">{{ "%.2f"|format(trend.cf_d365) }}<span class="unit">%</span></div>
        <div class="delta">CF · {{ "%.0f"|format(trend.mwh_d365/1000) }} GWh <span data-i18n="trend_curtailed">curtailed</span></div>
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
          <div class="lbl" data-i18n="expected_month">Expected (month)</div>
          <div class="val">{{ "%.1f"|format(tracker.esperada_mes) }}<span class="unit">MWh</span></div>
          <div class="delta">{{ tracker.dias_decorridos }} <span data-i18n="days_short">days</span></div>
        </div>
        <div class="t-stat">
          <div class="lbl" data-i18n="cut_month">Curtailed (month)</div>
          <div class="val">{{ "%.1f"|format(tracker.cortada_mes) }}<span class="unit">MWh</span></div>
          <div class="delta"><span data-i18n="of_lower">of</span> {{ "%.1f"|format(tracker.esperada_mes) }} <span data-i18n="expected_lower">expected</span></div>
        </div>
        <div class="t-stat alt">
          <div class="lbl" data-i18n="cf_month">CF% of month</div>
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
    <div class="chart"><div id="heatmap_dow" style="height:440px"></div></div>

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

      <div style="background:transparent;padding:0">
        <div id="mod_tracker" style="height:460px"></div>
      </div>
    </div>

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
  </footer>

</div>

<script>
const FIGS = {{ figs_json|safe }};

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
    footer_built: "Gerado em"
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
        trend=trend,
        met_ren=met_ren,
        eventos_top=eventos_top,
        met_irr=met_irr,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


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
