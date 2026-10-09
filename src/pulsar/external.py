"""Внешние данные Росстата (БДМО через каталог «Если быть точным»): население, работники и зарплаты по ОКВЭД2.

Читаем прямо из zip (parquet-член или CSV-часть за нужный год), берём только МО верхнего уровня, укрупняем
разделы ОКВЭД2 в секторы (таблица в configs/external.yaml) и присоединяем к territory_id через справочник
СберИндекса (версия, актуальная в году; ключ — ОКТМО 8 знаков; запасные ключи — oktmo_stable Росстата и
предшественники по change_id). Функции разбора принимают DataFrame, поэтому тестируются без архивов.

Запуск: `python -m pulsar.external` → outputs/external/rosstat_{year}.parquet (одна строка на territory_id)
и rosstat_{year}_by_oktmo.parquet (одна строка на ОКТМО Росстата).
"""
from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd

from pulsar import data

ROOT = data.ROOT
OUT = ROOT / "outputs" / "external"
TOTAL = "_total"  # служебный код строки «итого» в длинных таблицах разделов
META = ["municipality", "region_name", "oktmo_stable"]


def load_config() -> dict:
    return data.load_config("external")


# ---- чтение из архива --------------------------------------------------------

def read_bdmo_csv(buf, cfg: dict) -> pd.DataFrame:
    """CSV БДМО: sep=';', всё строками, ND/UD → NaN."""
    c = cfg["csv"]
    return pd.read_csv(buf, sep=c["sep"], dtype=str, na_values=c["na_values"], keep_default_na=False, encoding="utf-8")


def read_indicator(key: str, year: int, cfg: dict) -> pd.DataFrame:
    """Строки показателя `key` за `year`: parquet-член архива (если есть), иначе CSV-часть за год. Без распаковки."""
    path = ROOT / cfg["dir"] / cfg["files"][key]
    code = re.search(r"data_(Y\d+)_", path.name).group(1)
    m = cfg["members"]
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        pq = m["parquet"].format(code=code)
        if cfg.get("prefer_parquet", True) and pq in names:
            df = pd.read_parquet(io.BytesIO(z.read(pq)))
            df = df[df["year"].astype(int) == year]
        else:
            csv = m["csv_year"].format(code=code, year=year)
            if csv not in names:
                raise FileNotFoundError(f"{path.name}: нет члена {csv}")
            with z.open(csv) as f:
                df = read_bdmo_csv(f, cfg)
    if df.empty:
        raise FileNotFoundError(f"{path.name}: нет данных за {year}")
    return df.reset_index(drop=True)


# ---- разбор таблиц (чистые функции над DataFrame) ----------------------------

def oktmo8(s: pd.Series) -> pd.Series:
    """ОКТМО → 8 знаков: только цифры, ведущие нули (теряются в Excel/CSV), обрезка до 8; без цифр → NaN."""
    digits = s.astype(str).str.replace(r"\D", "", regex=True)
    return digits.str.zfill(8).str[:8].where(digits != "")


def top_level(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Только МО верхнего уровня; oktmo8, числовое value (ND/UD → NaN), без полных дублей строк (в БДМО встречаются)."""
    d = df[df["mun_level"] == cfg["top_level"]].copy()
    d["oktmo8"] = oktmo8(d["oktmo"])
    d = d.dropna(subset=["oktmo8"])
    d["value"] = pd.to_numeric(d["indicator_value"], errors="coerce")
    d["oktmo_stable"] = oktmo8(d["oktmo_stable"]) if "oktmo_stable" in d else np.nan
    for c in ("municipality", "region_name"):
        if c not in d:
            d[c] = np.nan
    return d.drop_duplicates()


def _meta(d: pd.DataFrame) -> pd.DataFrame:
    return d.drop_duplicates("oktmo8").set_index("oktmo8")[META]


def parse_population(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Одна строка на oktmo8: population (Все население), population_urban. Период «На 1 января» в приоритете."""
    p = cfg["population"]
    d = top_level(df, cfg)
    d["_rank"] = (d["indicator_period"] != p["period"]).astype(int)
    d = d.sort_values("_rank", kind="stable").drop_duplicates(["oktmo8", "mest"], keep="first")
    w = d.pivot(index="oktmo8", columns="mest", values="value")
    out = pd.DataFrame(index=w.index)
    out["population"] = w[p["total"]] if p["total"] in w else np.nan
    out["population_urban"] = w[p["urban"]] if p["urban"] in w else np.nan
    if p.get("rural") in w:  # у сельских районов строки «Городское» нет: городское = всё − сельское
        out["population_urban"] = out["population_urban"].fillna((out["population"] - w[p["rural"]]).clip(lower=0))
    out.loc[out["population"] <= 0, "population"] = np.nan  # нулевая численность — заглушка источника
    return out.join(_meta(d))


def section_code(label, cfg: dict) -> str | None:
    """«Раздел Н Транспортировка…» → 'H' (кириллические двойники → латиница); строка «итого» → TOTAL; иначе None."""
    s = cfg["sectors"]
    if label == s["total_label"]:
        return TOTAL
    m = re.match(r"Раздел\s+(\S)", str(label))
    if not m:
        return None
    letter = m.group(1).upper()
    return s["homoglyphs"].get(letter, letter)


def parse_sections(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Длинная таблица oktmo8 × раздел ОКВЭД2 за один период на МО: полный год, иначе последнее нарастающим итогом.

    Период выбирается по строке «итого» (чтобы итог и разделы были из одного периода), если её нет — по любой строке.
    """
    s = cfg["sectors"]
    d = top_level(df, cfg)
    d["section"] = d["okved2"].map(lambda x: section_code(x, cfg))
    d = d.dropna(subset=["section"])
    prio = {p: i for i, p in enumerate(s["period_priority"])}
    d["_rank"] = d["indicator_period"].map(prio)
    d = d.dropna(subset=["_rank"])
    r_any = d.groupby("oktmo8")["_rank"].min()
    r_tot = d[d["section"] == TOTAL].groupby("oktmo8")["_rank"].min()
    best = r_tot.reindex(r_any.index).fillna(r_any)
    d = d[d["_rank"].values == d["oktmo8"].map(best).values]
    d = d.drop_duplicates(["oktmo8", "section"])
    out = d[["oktmo8", "section", "value", "indicator_period"]].rename(columns={"indicator_period": "period"})
    out["full_year"] = out["period"] == s["full_year"]
    return out.merge(_meta(d), left_on="oktmo8", right_index=True, how="left").reset_index(drop=True)


def _wmean(v: pd.Series, w: pd.Series) -> float:
    """Среднее, взвешенное по w; если весов нет — простое среднее; если значений нет — NaN."""
    m = v.notna() & w.notna() & (w > 0)
    if m.any():
        return float((v[m] * w[m]).sum() / w[m].sum())
    return float(v.mean()) if v.notna().any() else np.nan


def sections_to_sectors(emp: pd.DataFrame, wage: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Разделы → укрупнённые секторы: работники суммируются, зарплата взвешивается по работникам раздела.

    Остаток «итого − сумма разделов» (конфиденциальные/пропущенные разделы) добавляется к сектору other,
    чтобы доли давали 1. Если строки «итого» нет — итог = сумма разделов.
    """
    s = cfg["sectors"]
    order, other = s["order"], s["other"]
    e = emp[["oktmo8", "section", "value"]].rename(columns={"value": "emp"})
    w = wage[["oktmo8", "section", "value"]].rename(columns={"value": "wage"})
    m = e.merge(w, on=["oktmo8", "section"], how="outer")
    tot = (m[m["section"] == TOTAL].set_index("oktmo8")[["emp", "wage"]]
           .rename(columns={"emp": "employees_total", "wage": "wage_total"}))
    m = m[m["section"] != TOTAL].copy()
    m["sector"] = m["section"].map(s["map"]).fillna(other)
    g = m.groupby(["oktmo8", "sector"])
    e_sec = g["emp"].sum(min_count=1).unstack("sector")
    both = m["emp"].notna() & m["wage"].notna()
    ww = (m.assign(we=(m["emp"] * m["wage"]).where(both), ew=m["emp"].where(both))
          .groupby(["oktmo8", "sector"])[["we", "ew"]].sum(min_count=1))
    w_sec = (ww["we"] / ww["ew"]).unstack("sector")
    w_sec = w_sec.where(w_sec.notna(), g["wage"].mean().unstack("sector"))
    idx = e_sec.index.union(w_sec.index).union(tot.index)
    base = pd.DataFrame(index=idx)
    for sec in order:
        base[f"emp_{sec}"] = e_sec[sec] if sec in e_sec else np.nan
    for sec in order:
        base[f"wage_{sec}"] = w_sec[sec] if sec in w_sec else np.nan
    base = base.join(tot)
    emp_cols = [f"emp_{sec}" for sec in order]
    known = base[emp_cols].sum(axis=1, min_count=1)
    has_tot = base["employees_total"].notna()
    # остаток = разделы, которые Росстат не опубликовал (мало организаций → конфиденциально); нулей в данных нет
    base["emp_residual"] = (base["employees_total"] - known.fillna(0)).clip(lower=0)
    base["n_sectors_reported"] = base[[f"emp_{sec}" for sec in order if sec != other]].notna().sum(axis=1)
    base.loc[has_tot, f"emp_{other}"] = base.loc[has_tot, f"emp_{other}"].fillna(0) + base.loc[has_tot, "emp_residual"]
    base["employees_total"] = base["employees_total"].fillna(base[emp_cols].sum(axis=1, min_count=1))
    for name, t in (("emp", emp), ("wage", wage)):
        per = t.drop_duplicates("oktmo8").set_index("oktmo8")
        base[f"{name}_period"] = per["period"]
        base[f"{name}_full_year"] = per["full_year"]
    meta = pd.concat([_meta_from_long(emp), _meta_from_long(wage)])
    meta = meta[~meta.index.duplicated()]
    return base.join(meta)


def _meta_from_long(t: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in META if c in t]
    m = t.drop_duplicates("oktmo8").set_index("oktmo8")[cols]
    return m.reindex(columns=META)


def _align_by_stable(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Строки `a`, чей oktmo8 отсутствует в `b`, но oktmo_stable там есть, переносятся под код из `b`.

    Таблицы БДМО обновляют код ОКТМО не синхронно: население за год может идти под старым кодом, работники — под новым.
    """
    idx = a.index.to_series()
    move = (~idx.isin(b.index)) & a["oktmo_stable"].isin(b.index) & (a["oktmo_stable"] != idx)
    move &= ~a["oktmo_stable"].isin(idx[~move])  # не склеивать, если под новым кодом в `a` уже есть своя строка
    new = idx.where(~move, a["oktmo_stable"])
    out = a.set_axis(pd.Index(new.tolist(), name="oktmo8"))
    return out[~out.index.duplicated()]


def assemble(pop: pd.DataFrame, emp: pd.DataFrame, wage: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Население + секторы в одну таблицу (индекс oktmo8); метаданные — из первого источника, где они есть."""
    sec = sections_to_sectors(emp, wage, cfg)
    pop = _align_by_stable(pop, sec)
    base = pop.drop(columns=META).join(sec.drop(columns=META), how="outer")
    meta = pop[META].combine_first(sec[META]).reindex(base.index)
    return base.join(meta)


def base_columns(cfg: dict) -> list[str]:
    order = cfg["sectors"]["order"]
    return (["population", "population_urban", "employees_total", "wage_total", "emp_residual", "n_sectors_reported"]
            + [f"emp_{s}" for s in order] + [f"wage_{s}" for s in order]
            + ["emp_period", "wage_period", "emp_full_year", "wage_full_year"] + META)


def derive(base: pd.DataFrame, cfg: dict, wage_median: float | None = None) -> pd.DataFrame:
    """Производные: доли секторов, HHI, ведущий сектор, добывающая/бюджетная доли, работники на жителя, wage_rel.

    Доли считаются при известном «итого»: неопубликованный сектор = 0 (его занятые, если есть, уже в остатке → other).
    Качество — `n_sectors_reported` и `residual_share` (доля итога, не разнесённая по опубликованным разделам);
    верхняя граница доли любого неопубликованного сектора = residual_share.
    """
    order, other = cfg["sectors"]["order"], cfg["sectors"]["other"]
    out = base.reindex(columns=list(dict.fromkeys(list(base.columns) + base_columns(cfg)))).copy()
    out["population_urban_share"] = out["population_urban"] / out["population"]
    tot = out["employees_total"]
    ok = tot.notna() & (tot > 0)
    shares = pd.DataFrame({sec: (out[f"emp_{sec}"].fillna(0) / tot).where(ok) for sec in order}, index=out.index)
    for sec in order:
        out[f"emp_share_{sec}"] = shares[sec]
    out["residual_share"] = (out["emp_residual"] / tot).where(ok)
    named = [sec for sec in order if sec != other]
    out["hhi_employment"] = (shares ** 2).sum(axis=1, min_count=1)
    out["top_sector"] = shares[named].fillna(-1.0).idxmax(axis=1).where(ok & (out["n_sectors_reported"] > 0))
    out["extractive_share"] = shares[cfg["derived"]["extractive"]].sum(axis=1, min_count=1)
    out["public_share"] = shares[cfg["derived"]["public"]].sum(axis=1, min_count=1)
    out["emp_per_capita"] = tot / out["population"]
    med = out["wage_total"].median() if wage_median is None else wage_median
    out["wage_rel"] = out["wage_total"] / med
    return out


def from_frames(pop_raw: pd.DataFrame, emp_raw: pd.DataFrame, wage_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Сырые таблицы БДМО (как в CSV) → одна строка на oktmo8 со всеми признаками."""
    base = assemble(parse_population(pop_raw, cfg), parse_sections(emp_raw, cfg), parse_sections(wage_raw, cfg), cfg)
    return derive(base, cfg).rename_axis("oktmo8").reset_index()


def load_rosstat(year: int = 2023, cfg: dict | None = None) -> pd.DataFrame:
    """Одна строка на oktmo8 (МО верхнего уровня) за `year`: население, работники и зарплаты по секторам, производные."""
    cfg = cfg or load_config()
    out = from_frames(read_indicator("population", year, cfg), read_indicator("employees", year, cfg),
                      read_indicator("wages", year, cfg), cfg)
    out["year"] = year
    return out


# ---- присоединение к territory_id -------------------------------------------

def aggregate_rows(rows: pd.DataFrame, cfg: dict) -> dict:
    """Несколько МО Росстата → одно: численности суммируются, зарплаты взвешиваются по работникам (сектора — по своим)."""
    order = cfg["sectors"]["order"]
    out = {}
    for c in ["population", "population_urban", "employees_total", "emp_residual"] + [f"emp_{s}" for s in order]:
        out[c] = rows[c].sum(min_count=1)
    out["n_sectors_reported"] = rows["n_sectors_reported"].min() if rows["n_sectors_reported"].notna().any() else np.nan
    out["wage_total"] = _wmean(rows["wage_total"], rows["employees_total"])
    for s in order:
        out[f"wage_{s}"] = _wmean(rows[f"wage_{s}"], rows[f"emp_{s}"])
    for c in ("emp_period", "wage_period"):
        out[c] = ", ".join(sorted(set(rows[c].dropna()))) or np.nan
    for c in ("emp_full_year", "wage_full_year"):
        out[c] = bool(rows[c].fillna(False).all()) if rows[c].notna().any() else np.nan
    out["municipality"] = " + ".join(rows["municipality"].dropna().astype(str)) or np.nan
    out["region_name"] = rows["region_name"].dropna().iloc[0] if rows["region_name"].notna().any() else np.nan
    out["oktmo_stable"] = rows["oktmo_stable"].dropna().iloc[0] if rows["oktmo_stable"].notna().any() else np.nan
    return out


def attach_to_territory(df: pd.DataFrame, year: int = 2023, mdict: pd.DataFrame | None = None,
                        cfg: dict | None = None) -> pd.DataFrame:
    """Одна строка на territory_id справочника СберИндекса (версия, актуальная в `year`).

    Ключи по порядку: oktmo8 == oktmo Росстата ('exact'); oktmo8 == oktmo_stable Росстата (последний действующий код,
    'exact' если один источник, иначе 'aggregated'); предшественники по change_id (union/transfer) — 'aggregated';
    иначе 'missing'. При агрегировании численности суммируются, зарплаты взвешиваются по работникам.
    """
    cfg = cfg or load_config()
    if mdict is None:
        mdict = data.read_municipal_dict(year=year)
    cols = base_columns(cfg)
    R = df.drop_duplicates("oktmo8").set_index("oktmo8").reindex(columns=cols)
    wage_median = R["wage_total"].median()
    by_stable = R[R["oktmo_stable"].notna()].groupby("oktmo_stable").groups
    rows = []
    for t in mdict.itertuples(index=False):
        key, via = t.oktmo8, None
        cid = getattr(t, "change_id_from", np.nan)
        if key in R.index:
            src, via = [key], "oktmo"
        elif key in by_stable:
            src, via = list(by_stable[key]), "oktmo_stable"
        elif pd.notna(cid) and str(cid).startswith("union") and "change_id_to" in mdict:  # transfer — не сумма
            preds = mdict.loc[(mdict["change_id_to"] == cid) & (mdict["territory_id"] != t.territory_id), "oktmo8"]
            src, via = [k for k in dict.fromkeys(preds) if k in R.index], "union"
        else:
            src = []
        if not src:
            rec, quality, via = {}, "missing", None
        elif len(src) == 1 and via != "union":
            rec, quality = R.loc[src[0]].to_dict(), "exact"
        else:
            rec, quality = aggregate_rows(R.loc[src], cfg), "aggregated"
        rec.update(territory_id=t.territory_id, oktmo8=key, match_quality=quality, match_via=via, n_sources=len(src))
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("territory_id")
    out = derive(out, cfg, wage_median=wage_median).reset_index()
    out["year"] = year
    front = ["territory_id", "oktmo8", "match_quality", "match_via", "n_sources"]
    return out[front + [c for c in out.columns if c not in front]]


# ---- сборка -------------------------------------------------------------------

def build(write: bool = True) -> pd.DataFrame:
    """outputs/external/rosstat_{year}.parquet для основного и запасных годов; возвращает таблицу основного года."""
    cfg = load_config()
    main = None
    for year in [cfg["year"]] + list(cfg.get("fallback_years", [])):
        try:
            ros = load_rosstat(year, cfg)
        except FileNotFoundError as e:
            print(f"! {year}: {e}")
            continue
        out = attach_to_territory(ros, year, cfg=cfg)
        if write:
            OUT.mkdir(parents=True, exist_ok=True)
            out.to_parquet(OUT / f"rosstat_{year}.parquet", index=False)
            ros.to_parquet(OUT / f"rosstat_{year}_by_oktmo.parquet", index=False)
        print(f"{year}: ОКТМО Росстата {len(ros)}, territory_id {len(out)}, "
              f"match: {out['match_quality'].value_counts().to_dict()}")
        if main is None:
            main = out
    return main


def coverage(out: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """Доля МО панели (in_panel) с заполненными ключевыми колонками."""
    ids = panel.loc[panel["in_panel"], "territory_id"]
    sub = out[out["territory_id"].isin(ids)]
    cols = ["population", "wage_total", "employees_total", "emp_share_manufacturing"]
    s = {f"n_{c}": int(sub[c].notna().sum()) for c in cols}
    s.update(n_panel=int(len(ids)), n_matched=int(len(sub)), **{f"q_{k}": int(v) for k, v in sub["match_quality"].value_counts().items()})
    return pd.Series(s)


if __name__ == "__main__":
    res = build()
    panel_path = ROOT / "outputs" / "features" / "panel.parquet"
    if res is not None and panel_path.exists():
        print(coverage(res, pd.read_parquet(panel_path)).to_string())
