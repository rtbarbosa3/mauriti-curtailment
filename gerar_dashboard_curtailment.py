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


def _download(url: str, dest: Path, timeout: int, retries: int,
               force: bool = False) -> bool:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return True
    if force and dest.exists():
        dest.unlink()
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
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
    """Mantem apenas linhas onde nom_usina contem algum dos patterns.
    Reduz drasticamente uso de memoria descartando usinas irrelevantes
    logo na entrada (antes de concatenar varios meses)."""
    if df.empty or not patterns or "nom_usina" not in df.columns:
        return df
    nom_norm = df["nom_usina"].astype(str).map(_normalize)
    mask = pd.Series(False, index=df.index)
    for pat in patterns:
        mask |= nom_norm.str.contains(pat, na=False, regex=False)
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
    df["cod_razaorestricao"] = df["cod_razaorestricao"].fillna("DESCONHECIDA")
    df["cod_origemrestricao"] = df["cod_origemrestricao"].fillna("DESCONHECIDA")
    print(f"  -> {len(df):,} linhas com razao/origem")
    return df


def carregar_pld(cfg: dict, dt_ini: date, dt_fim: date,
                  submercado: str) -> pd.DataFrame:
    cache = _ensure_dir(Path(cfg["cache_dir"]) / "ccee")
    today = date.today()
    anos = sorted({d.year for d in pd.date_range(dt_ini, dt_fim, freq="D")})
    print(f"\n[3/4] PLD horario CCEE ({anos})...")
    frames = []
    for ano in anos:
        url = CCEE_PLD_URLS.get(ano)
        if not url: continue
        dest = cache / f"pld_horario_{ano}.csv"
        # Sempre re-baixa o ano corrente
        force = (ano == today.year)
        if not _download(url, dest, cfg["request_timeout"], cfg["max_retries"],
                          force=force):
            continue
        try:
            frames.append(_read_csv_robust(dest))
        except Exception as e:
            print(f"  [!] {e}")
    if not frames:
        rng = pd.date_range(dt_ini, dt_fim + pd.Timedelta(days=1), freq="h")
        return pd.DataFrame({"hora": rng, "pld": 200.0})
    pld = pd.concat(frames, ignore_index=True)
    pld.columns = [c.strip().lower() for c in pld.columns]
    col_data = next((c for c in pld.columns if "din_inicio" in c
                     or "din_referencia" in c or "din_instante" in c), None)
    col_sub = next((c for c in pld.columns if "submercado" in c), None)
    col_pld = next((c for c in pld.columns
                    if c == "val_pld" or ("pld" in c and "val" in c)), None)
    if col_pld is None:
        col_pld = next((c for c in pld.columns if "preco" in c or "valor" in c),
                        None)
    pld = pld.rename(columns={col_data: "hora", col_sub: "sub", col_pld: "pld"})
    pld["hora"] = pd.to_datetime(pld["hora"], errors="coerce")
    pld["pld"] = pd.to_numeric(pld["pld"].astype(str).str.replace(",", "."),
                                errors="coerce")
    pld = pld.dropna(subset=["hora", "pld"])
    pld = pld[pld["sub"].astype(str).str.upper().str.contains(submercado.upper())]
    pld = pld[(pld["hora"] >= pd.Timestamp(dt_ini)) &
              (pld["hora"] <= pd.Timestamp(dt_fim) + pd.Timedelta(days=1))]
    pld = pld[["hora", "pld"]].drop_duplicates("hora").sort_values("hora")
    print(f"  -> {len(pld):,} horas em {submercado}")
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

    if not cons.empty and "cod_razaorestricao" in cons.columns:
        cons_keys = ["din_instante", "ceg", "cod_razaorestricao",
                     "cod_origemrestricao"]
        prio = {"CNF": 0, "REL": 1, "PAR": 2, "ENE": 3, "DESCONHECIDA": 9}
        c2 = cons[cons_keys].copy()
        c2["prio"] = c2["cod_razaorestricao"].map(prio).fillna(99)
        c2 = (c2.sort_values(["din_instante", "ceg", "prio"])
                .drop_duplicates(["din_instante", "ceg"], keep="first")
                .drop(columns=["prio"]))
        df = df.merge(c2, on=["din_instante", "ceg"], how="left")
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
    for g in groups_cfg:
        pat = _normalize(g["match"])
        sub = df[df["nom_usina_norm"].str.contains(pat, na=False)].copy()
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
    razao_top = cortes.groupby("cod_razaorestricao")["curtailment_mwh"].sum().idxmax()
    pct_top = (cortes[cortes["cod_razaorestricao"] == razao_top]["curtailment_mwh"]
                .sum() / cortes["curtailment_mwh"].sum() * 100)
    origem_top = cortes.groupby("cod_origemrestricao")["curtailment_mwh"].sum().idxmax()
    horas = cortes.groupby(cortes["din_instante"].dt.hour)["curtailment_mwh"].sum()
    hora_pico = int(horas.idxmax())
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
<title>Mauriti — Estudo de Curtailment</title>
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
  border-bottom:1px solid var(--ink);padding-bottom:16px;margin-bottom:64px;
  font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--ink-2);letter-spacing:0.18em;text-transform:uppercase}
.masthead .vol{font-weight:600}
.hero{margin-bottom:80px}
.hero .kicker{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--accent);letter-spacing:0.25em;text-transform:uppercase;
  margin-bottom:18px}
.hero h1{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:clamp(44px,8vw,86px);line-height:0.96;
  letter-spacing:-0.02em;margin:0 0 28px;font-variation-settings:"opsz" 144}
.hero h1 em{font-style:italic;font-weight:400;color:var(--accent)}
.hero .lede{font-family:'Fraunces',Georgia,serif;font-weight:400;font-size:22px;
  line-height:1.45;color:var(--ink-2);max-width:720px;
  font-variation-settings:"opsz" 36}
.hero .lede strong{color:var(--ink);font-weight:500}
.hero .byline{margin-top:36px;font-family:'IBM Plex Mono',monospace;
  font-size:11px;color:var(--muted);letter-spacing:0.1em;
  text-transform:uppercase;line-height:1.8}
.hero .byline span{color:var(--ink)}
.tracker{margin:60px 0 80px;padding:32px 36px;background:var(--bg-alt);
  border:1px solid var(--rule);border-radius:2px;position:relative}
.tracker .liveflag{position:absolute;top:-12px;left:32px;
  background:var(--accent-today);color:#fff;
  font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;
  letter-spacing:0.2em;padding:5px 10px;text-transform:uppercase}
.tracker h2{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:30px;line-height:1.1;letter-spacing:-0.01em;margin:0 0 6px}
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
  align-items:end;margin:80px 0 60px;padding-top:40px;
  border-top:3px double var(--rule)}
.bignum .figure{font-family:'Fraunces',Georgia,serif;font-weight:300;
  font-size:clamp(96px,18vw,180px);line-height:0.9;letter-spacing:-0.04em;
  color:var(--accent);font-variation-settings:"opsz" 144}
.bignum .figure span{font-size:0.32em;color:var(--ink);font-weight:500;
  margin-left:14px;letter-spacing:0;display:inline-block}
.bignum .copy h2{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:32px;line-height:1.15;letter-spacing:-0.01em;margin:0 0 16px}
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
  font-size:38px;line-height:1;letter-spacing:-0.02em;color:var(--ink);
  font-variation-settings:"opsz" 72}
.stat .val .unit{font-family:'IBM Plex Sans',sans-serif;font-size:14px;
  color:var(--muted);font-weight:400;margin-left:4px}
.stat .delta{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--muted);margin-top:8px;letter-spacing:0.05em}
.stat.alt .val{color:var(--accent)}
.section-head{margin:80px 0 32px;padding-bottom:14px;
  border-bottom:1px solid var(--ink);
  display:flex;align-items:baseline;gap:18px}
.section-head .num{font-family:'IBM Plex Mono',monospace;font-size:11px;
  color:var(--muted);letter-spacing:0.2em}
.section-head h3{font-family:'Fraunces',Georgia,serif;font-weight:500;
  font-size:28px;line-height:1.1;letter-spacing:-0.01em;margin:0;flex:1;
  font-variation-settings:"opsz" 36}
.section-head .tag{font-family:'IBM Plex Mono',monospace;font-size:10px;
  color:var(--muted);text-transform:uppercase;letter-spacing:0.15em}
.section-desc{font-family:'Fraunces',Georgia,serif;font-size:18px;
  line-height:1.55;color:var(--ink-2);max-width:680px;margin:0 0 32px;
  font-variation-settings:"opsz" 28}
.pullquote{margin:48px 0;padding:32px 40px;background:var(--bg-alt);
  border-left:3px solid var(--accent);
  font-family:'Fraunces',Georgia,serif;font-size:21px;line-height:1.5;
  color:var(--ink);font-variation-settings:"opsz" 36}
.pullquote strong{color:var(--accent);font-weight:500;font-style:italic}
.pullquote cite{display:block;margin-top:14px;font-family:'IBM Plex Mono',monospace;
  font-size:10px;color:var(--muted);font-style:normal;
  letter-spacing:0.15em;text-transform:uppercase}
.chart{background:var(--panel);border:1px solid var(--border);
  border-radius:2px;margin:24px 0;padding:8px 4px 4px}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:24px 0}
@media (max-width:880px){.chart-row{grid-template-columns:1fr}}
footer{margin-top:120px;padding-top:32px;border-top:1px solid var(--ink);
  font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);
  line-height:1.8;letter-spacing:0.04em}
footer p{margin:0 0 10px;max-width:680px}
footer .colofao{font-family:'Fraunces',Georgia,serif;font-style:italic;
  font-size:13px;color:var(--ink-2);margin-top:24px}
</style>
</head>
<body>

<div class="wrap">

  <div class="masthead">
    <div class="vol">Constrained-off Report — N&deg; 01</div>
    <div>{{ periodo }} &nbsp;&middot;&nbsp; atualizado {{ gerado_em }}</div>
  </div>

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

  <!-- TRACKER MES CORRENTE -->
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
        <strong>{{ "%.2f"|format(met_m.curtailment_factor) }}%</strong> sobre a
        gera&ccedil;&atilde;o de refer&ecirc;ncia.</p>
      <p>Desse total, estima-se que <strong>{{ "%.1f"|format(met_m.pct_ressarcivel) }}%
        sejam potencialmente ressarciveis</strong> sob a REN ANEEL 1.030/2022
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
    Compara&ccedil;&atilde;o direta com {{ n_grupos }} grupos de ativos no Cear&aacute;:
    <strong>{{ grupos_str }}</strong>. Cada grupo agrega todas as usinas que
    casarem com seu padr&atilde;o de nome.
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
    Heatmap do CF% por hora do dia ao longo do per&iacute;odo. Cortes concentrados em
    11h&ndash;14h indicam restri&ccedil;&atilde;o sist&ecirc;mica do submercado NE
    (pico solar coincide com vale de demanda); cortes pulverizados sugerem
    limita&ccedil;&atilde;o local da SE/LT que escoa o complexo.
  </p>
  <div class="chart"><div id="heatmap" style="height:440px"></div></div>

  <footer>
    <p><strong>Fontes</strong> Restri&ccedil;&atilde;o de Opera&ccedil;&atilde;o por Constrained-off
      (ONS, base semi-hor&aacute;ria, detalhamento por usina + consolidado com raz&otilde;es).
      PLD hor&aacute;rio (CCEE).</p>
    <p><strong>Defini&ccedil;&otilde;es</strong> Curtailment = max(0,
      val_geracaoestimada &minus; val_geracaoverificada). Receita perdida =
      curtailment_MWh &times; PLD_horario. CF = curtailment / esperada (%).</p>
    <p><strong>Ressarcimento</strong> A estimativa de potencial ressarciv&eacute;l
      considera apenas as raz&otilde;es REL e CNF sob a REN ANEEL 1.030/2022,
      sujeita &agrave; modalidade da usina e aos termos do contrato.</p>
    <p class="colofao">Mauriti — Estudo de Curtailment, gerado em
      {{ gerado_em }}.</p>
  </footer>

</div>

<script>
const FIGS = {{ figs_json|safe }};
for (const [k, fig] of Object.entries(FIGS)) {
  if (document.getElementById(k)) {
    Plotly.newPlot(k, fig.data, fig.layout, {
      responsive:true, displaylogo:false,
      modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d','zoomIn2d','zoomOut2d']
    });
  }
}
</script>
</body>
</html>
"""


# =============================================================================
#  RENDER
# =============================================================================

def gerar_html(mauriti: Selecao, grupos: list[Grupo], pld_sub: str,
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

    grupos_str = ", ".join([g.label for g in grupos]) if grupos else "—"
    html = Template(HTML_TEMPLATE).render(
        met_m=met_m, met_b=met_b,
        n_grupos=len(grupos), grupos_str=grupos_str,
        submercado=pld_sub, periodo=periodo,
        gerado_em=today.strftime("%d/%m/%Y"),
        figs_json=json.dumps(figs),
        insights_m=_gera_insights_mauriti(mauriti.df, met_m),
        insights_c=_gera_insights_comp(met_m, met_b),
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
    gerar_html(mauriti, grupos, cfg["submercado"], periodo, out, date.today())
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
