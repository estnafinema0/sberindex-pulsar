"""Аудит исходных данных СберИндекса: покрытие, пропуски, аномалии. Пишет docs/data_audit.md."""
from __future__ import annotations
import sys, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hackathonlicence"
DICT = ROOT / "data/external/t_dict_municipal/t_dict_municipal_districts.xlsx"
OUT = ROOT / "docs/data_audit.md"
L: list[str] = []
def p(s=""): L.append(str(s)); print(s)

c = pd.read_parquet(RAW / "consumption.parquet")
m = pd.read_parquet(RAW / "market_access.parquet")
k = pd.read_parquet(RAW / "connection.parquet")
d = pd.read_excel(DICT)

p("# Аудит данных\n")
p("## consumption.parquet")
p(f"- строк: {len(c):,}; МО: {c.territory_id.nunique():,}; месяцев: {c.date.nunique()} ({c.date.min()}–{c.date.max()}); категорий: {c.category.nunique()}")
p(f"- дубликатов (territory_id, date, category): {c.duplicated(['territory_id','date','category']).sum()}")
p(f"- value: min={c.value.min()}, медиана={c.value.median():.0f}, max={c.value.max():,}; нулей: {(c.value<=0).sum()}")
tot = c[c.category == "Все категории"]
cov = tot.groupby("territory_id").date.nunique()
p(f"- МО с полной историей (24 мес. в «Все категории»): {(cov==24).sum():,}; с ≥18 мес.: {(cov>=18).sum():,}; ≤6 мес.: {(cov<=6).sum():,}")
p("- МО по месяцам (Все категории): " + ", ".join(f"{dt[2:]}:{n}" for dt, n in tot.groupby("date").territory_id.nunique().items()))
# по категориям: сколько МО
p("- МО по категориям: " + ", ".join(f"{cat}: {n}" for cat, n in c.groupby("category").territory_id.nunique().items()))
# полнота категорий внутри МО-месяца
w = c.pivot_table(index=["territory_id","date"], columns="category", values="value")
p(f"- МО-месяцев всего: {len(w):,}; с полными 6 категориями: {w.notna().all(axis=1).sum():,}; без «Все категории»: {w['Все категории'].isna().sum():,}")
cats = [x for x in w.columns if x != "Все категории"]
share = w[cats].div(w["Все категории"], axis=0)
p("- медианные доли категорий в итоге: " + ", ".join(f"{x}: {share[x].median():.3f}" for x in cats) + f"; сумма пяти: {share.sum(axis=1).median():.3f}; доля >1: {(share.sum(axis=1)>1).sum()}")
# сезонность и апрель 2023
nat = w.groupby("date").median()
mom = nat.pct_change()
p("- медианный уровень «Все категории» по месяцам: " + ", ".join(f"{i[2:]}:{v:.0f}" for i, v in nat["Все категории"].items()))
p("- прирост медианы м/м (Все категории): " + ", ".join(f"{i[2:]}:{v:+.1%}" for i, v in mom["Все категории"].dropna().items()))
for cat in cats:
    p(f"  - {cat}: " + ", ".join(f"{i[2:]}:{v:+.1%}" for i, v in mom[cat].dropna().items()))
# доля МО с резким скачком в апреле 2023 по категориям
lvl = w.unstack("date")
for cat in ["Все категории"] + cats:
    a, b = lvl[cat].get("2023-03"), lvl[cat].get("2023-04")
    if a is not None and b is not None:
        r = (b / a).dropna()
        p(f"  - 2023-04/2023-03, {cat}: медиана {r.median():.3f}, доля МО с изменением >+20%: {(r>1.2).mean():.1%}, < -20%: {(r<0.8).mean():.1%}")

p("\n## market_access.parquet")
p(f"- строк: {len(m):,}; уникальных МО: {m.territory_id.nunique():,}; NaN: {m.market_access.isna().sum()}; min/med/max: {m.market_access.min()}/{m.market_access.median()}/{m.market_access.max()}")
p(f"- МО из consumption без market_access: {len(set(c.territory_id)-set(m.territory_id))}")

p("\n## connection.parquet")
p(f"- строк: {len(k):,}; типы: {k.type.value_counts().to_dict()}")
p(f"- уникальных МО (x∪y): {len(set(k.territory_id_x)|set(k.territory_id_y)):,}")
same = k[k.territory_id_x == k.territory_id_y]
p(f"- пар с x==y: {len(same):,}")
zero = k[(k.distance == 0) & (k.territory_id_x != k.territory_id_y)]
p(f"- нулевых расстояний между разными МО: {len(zero):,} (по типам: {zero.type.value_counts().to_dict()})")
if len(zero):
    nm = d.drop_duplicates("territory_id").set_index("territory_id")
    lab = lambda t: f"{nm.municipal_district_name_short.get(t, '?')} ({nm.region_name.get(t, '?')})"
    zz = zero.assign(nx=zero.territory_id_x.map(lab), ny=zero.territory_id_y.map(lab))
    p(f"  нулевые пары: уникальных МО в них {len(set(zero.territory_id_x)|set(zero.territory_id_y))}; есть ли среди них МО consumption: {len((set(zero.territory_id_x)|set(zero.territory_id_y)) & set(c.territory_id))}")
    p("  примеры: " + "; ".join(f"{r.nx} — {r.ny}" for r in zz.head(8).itertuples()))
hw = k[k.type == "highway"]
p(f"- highway: пар {len(hw):,}; дубликатов (x,y): {hw.duplicated(['territory_id_x','territory_id_y']).sum()}; есть ли обратные пары (симметрия хранится дважды?): {len(hw.merge(hw, left_on=['territory_id_x','territory_id_y'], right_on=['territory_id_y','territory_id_x'])):,}")
p(f"- distance: min/med/max = {k.distance.min()}/{k.distance.median()}/{k.distance.max()}; NaN: {k.distance.isna().sum()}")
cons_ids = set(c.territory_id)
hw_ids = set(hw.territory_id_x) | set(hw.territory_id_y)
p(f"- МО из consumption без highway-расстояний: {len(cons_ids - hw_ids)}")
# покрытие пар среди МО consumption
n = len(cons_ids); 
sub = hw[hw.territory_id_x.isin(cons_ids) & hw.territory_id_y.isin(cons_ids)]
p(f"- highway-пар среди МО consumption: {len(sub):,} из {n*(n-1)//2:,} возможных ({len(sub)/(n*(n-1)//2):.1%})")

p("\n## Справочник МО (t_dict_municipal_districts.xlsx)")
p(f"- строк: {len(d):,}; уникальных territory_id: {d.territory_id.nunique():,}; колонок: {list(d.columns)}")
cur = d[(d.year_from <= 2024) & (d.year_to > 2024)]
p(f"- версий, актуальных в 2024: {len(cur):,}; territory_id без актуальной версии в 2024 среди consumption: {len(cons_ids - set(cur.territory_id))}")
p(f"- МО consumption, отсутствующих в справочнике вовсе: {len(cons_ids - set(d.territory_id))}")
dd = d.drop_duplicates("territory_id")
j = pd.Series(sorted(cons_ids), name="territory_id").to_frame().merge(dd, on="territory_id", how="left")
p(f"- типы МО в consumption: {j.municipal_district_type.value_counts(dropna=False).to_dict()}")
p(f"- статусы: {j.municipal_district_status.value_counts(dropna=False).to_dict()}")
regs_all = set(d.region_name); regs_c = set(j.region_name.dropna())
p(f"- регионов в справочнике: {len(regs_all)}; в consumption: {len(regs_c)}; отсутствуют: {sorted(regs_all - regs_c)}")
p("- регионов с ≤3 МО в consumption: " + str(j.region_name.value_counts().pipe(lambda s: s[s<=3]).to_dict()))
ch = d[d.change_id_to.notna()][["territory_id","municipal_district_name_short","region_name","year_to","change_id_to"]]
p(f"- записей с change_id_to (объединения/изменения): {len(ch)}; за 2023–2025: " + "; ".join(f"{r.municipal_district_name_short} ({r.region_name}, до {r.year_to}, {r.change_id_to})" for r in ch[ch.year_to>=2023].itertuples()))
# Орск
o = d[d.municipal_district_name_short.str.contains("Орск", na=False)]
p(f"- Орск: {o[['territory_id','municipal_district_name','region_name','year_from','year_to']].to_dict('records')}")
if len(o):
    tid = int(o.territory_id.iloc[0]); s = w.loc[tid] if tid in w.index.get_level_values(0) else None
    if s is not None:
        p("  Орск, «Все категории» по месяцам: " + ", ".join(f"{i[2:]}:{v:.0f}" for i, v in s["Все категории"].items()))
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"\n-> {OUT}")
