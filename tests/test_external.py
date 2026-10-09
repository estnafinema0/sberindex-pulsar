"""Тесты внешних данных Росстата: верхний уровень, zfill ОКТМО, ND → NaN, доли секторов, периоды, union."""
from __future__ import annotations

import io
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pulsar import external  # noqa: E402

CFG = external.load_config()
TOP = CFG["top_level"]
LOW = "Муниципальное образование нижнего уровня"
TOTAL = CFG["sectors"]["total_label"]
COLS = ["mun_level", "municipality", "region_name", "oktmo", "oktmo_stable", "year", "indicator_value", "indicator_period"]


def raw(attr: str, rows: list[tuple]) -> pd.DataFrame:
    """Строки как в CSV БДМО: (oktmo, attr, period, value[, mun_level[, oktmo_stable]])."""
    recs = []
    for r in rows:
        oktmo, a, period, value = r[:4]
        level = r[4] if len(r) > 4 else TOP
        stable = r[5] if len(r) > 5 else oktmo
        recs.append(dict(mun_level=level, municipality=f"МО {oktmo}", region_name="Регион", oktmo=oktmo,
                         oktmo_stable=stable, year="2023", indicator_value=value, indicator_period=period, **{attr: a}))
    return pd.DataFrame(recs)


def pop_raw() -> pd.DataFrame:
    return raw("mest", [
        ("1512000", "Все население", "На 1 января", "1000"),      # 7 знаков — потерян ведущий ноль
        ("1512000", "Все население", "На 1 января", "1000"),      # полный дубль строки
        ("1512000", "Все население", "Значение показателя за год", "999"),  # второстепенный период
        ("1512000", "Городское население", "На 1 января", "400"),
        ("01512411", "Все население", "На 1 января", "300", LOW),  # поселение — не берём
        ("02000000", "Все население", "На 1 января", "ND"),
        ("02000000", "Городское население", "На 1 января", "UD"),
        ("03000000", "Все население", "На 1 января", "500"),       # сельский район: строки «Городское» нет
        ("03000000", "Сельское население", "На 1 января", "500"),
    ])


def emp_raw() -> pd.DataFrame:
    return raw("okved2", [
        ("01512000", TOTAL, "Январь-декабрь", "1000"),
        ("01512000", "Раздел А Сельское хозяйство", "Январь-декабрь", "100"),   # кириллическая А
        ("01512000", "Раздел C Обрабатывающие производства", "Январь-декабрь", "200"),
        ("01512000", "Раздел D Обеспечение электроэнергией", "Январь-декабрь", "75"),
        ("01512000", "Раздел Е Водоснабжение", "Январь-декабрь", "25"),       # кириллическая Е
        ("01512000", "Раздел Н Транспортировка и хранение", "Январь-декабрь", "100"),  # кириллическая Н
        ("01512000", "Раздел P Образование", "Январь-декабрь", "250"),
        ("01512000", "Раздел R Культура", "Январь-декабрь", "50"),
        ("01512000", "Раздел B Добыча", "Январь-декабрь", "ND"),
        ("01512000", TOTAL, "Январь-сентябрь", "900"),               # менее полный период — игнорируем
        ("02000000", TOTAL, "Январь-сентябрь", "500"),               # полного года нет
        ("02000000", "Раздел C Обрабатывающие производства", "Январь-сентябрь", "500"),
        ("02000000", "Раздел C Обрабатывающие производства", "Январь-март", "450"),
    ])


def wage_raw() -> pd.DataFrame:
    return raw("okved2", [
        ("01512000", TOTAL, "Январь-декабрь", "50000"),
        ("01512000", "Раздел D Обеспечение электроэнергией", "Январь-декабрь", "60000"),
        ("01512000", "Раздел Е Водоснабжение", "Январь-декабрь", "40000"),
        ("01512000", "Раздел L Недвижимость", "Январь-декабрь", "30000"),   # зарплата есть, работников нет
        ("01512000", "Раздел M Наука", "Январь-декабрь", "50000"),
        ("02000000", TOTAL, "Январь-сентябрь", "25000"),
    ])


def test_read_csv_nd_to_nan_and_zfill():
    csv = "mun_level;oktmo;mest;indicator_value;indicator_period\n" + \
          f"{TOP};1512000;Все население;ND;На 1 января\n{TOP};1512000;Городское население;5;На 1 января\n"
    df = external.read_bdmo_csv(io.StringIO(csv), CFG)
    assert df["indicator_value"].isna().iloc[0] and df["oktmo"].iloc[0] == "1512000"
    d = external.top_level(df.assign(oktmo_stable="1512000"), CFG)
    assert set(d["oktmo8"]) == {"01512000"}
    assert d["value"].isna().iloc[0] and d["value"].iloc[1] == 5


def test_population_top_level_dedup_and_period():
    p = external.parse_population(pop_raw(), CFG)
    assert list(p.index) == ["01512000", "02000000", "03000000"]  # поселение отброшено, ОКТМО дополнен нулём
    assert p.loc["01512000", "population"] == 1000              # «На 1 января» важнее годового, дубль схлопнут
    assert p.loc["01512000", "population_urban"] == 400
    assert np.isnan(p.loc["02000000", "population"])            # ND → NaN
    assert p.loc["03000000", "population_urban"] == 0           # городское = всё − сельское


def test_population_aligned_by_stable_code():
    pop = raw("mest", [("44702000", "Все население", "На 1 января", "9000", TOP, "44502000")])   # старый код
    emp = raw("okved2", [("44502000", TOTAL, "Январь-декабрь", "100")])                          # уже новый код
    wage = raw("okved2", [("44502000", TOTAL, "Январь-декабрь", "1000")])
    out = external.from_frames(pop, emp, wage, CFG).set_index("oktmo8")
    assert list(out.index) == ["44502000"] and out.loc["44502000", "population"] == 9000


def test_section_code_homoglyphs():
    assert external.section_code("Раздел Н Транспортировка и хранение", CFG) == "H"
    assert external.section_code("Раздел А Сельское хозяйство", CFG) == "A"
    assert external.section_code("Раздел E Водоснабжение", CFG) == "E"
    assert external.section_code(TOTAL, CFG) == external.TOTAL
    assert external.section_code("что-то другое", CFG) is None


def test_sector_shares_sum_to_one_and_derived():
    out = external.from_frames(pop_raw(), emp_raw(), wage_raw(), CFG).set_index("oktmo8")
    r = out.loc["01512000"]
    order = CFG["sectors"]["order"]
    shares = np.array([r[f"emp_share_{s}"] for s in order], dtype=float)
    assert np.nansum(shares) == pytest.approx(1.0)
    assert r["emp_share_energy_utilities"] == pytest.approx(0.1)          # D + E
    assert r["emp_share_other"] == pytest.approx(0.25)                    # R (50) + остаток итога (200)
    assert r["employees_total"] == 1000 and r["emp_full_year"]
    assert r["wage_energy_utilities"] == pytest.approx(55000)             # взвешено по работникам 75/25
    assert r["wage_realestate_prof"] == pytest.approx(40000)              # нет весов → простое среднее
    assert r["top_sector"] == "education"
    assert r["extractive_share"] == pytest.approx(0.1)                    # mining NaN + energy 0.1
    assert r["public_share"] == pytest.approx(0.25)
    assert r["hhi_employment"] == pytest.approx(0.1**2 + 0.2**2 + 0.1**2 + 0.1**2 + 0.25**2 + 0.25**2)
    assert r["population_urban_share"] == pytest.approx(0.4)
    assert r["emp_per_capita"] == pytest.approx(1.0)
    assert np.isnan(r["emp_mining"]) and r["emp_share_mining"] == 0       # ND → NaN в абсолюте, 0 в доле (есть итог)
    assert r["n_sectors_reported"] == 5 and r["residual_share"] == pytest.approx(0.2)


def test_period_fallback_when_no_full_year():
    out = external.from_frames(pop_raw(), emp_raw(), wage_raw(), CFG).set_index("oktmo8")
    r = out.loc["02000000"]
    assert r["emp_period"] == "Январь-сентябрь" and not r["emp_full_year"]
    assert r["employees_total"] == 500 and r["emp_share_manufacturing"] == pytest.approx(1.0)
    assert r["wage_total"] == 25000 and r["wage_rel"] == pytest.approx(25000 / 37500)  # медиана по двум МО


def test_attach_exact_stable_union_missing():
    pop = raw("mest", [
        ("11111000", "Все население", "На 1 января", "100"),
        ("11112000", "Все население", "На 1 января", "300"),
        ("22222000", "Все население", "На 1 января", "50"),
        ("33333000", "Все население", "На 1 января", "70", TOP, "33334000"),   # код сменился, stable — новый
    ])
    emp = raw("okved2", [
        ("11111000", TOTAL, "Январь-декабрь", "10"), ("11111000", "Раздел C Обработка", "Январь-декабрь", "10"),
        ("11112000", TOTAL, "Январь-декабрь", "30"), ("11112000", "Раздел A Сельское", "Январь-декабрь", "30"),
        ("22222000", TOTAL, "Январь-декабрь", "5"),
        ("33333000", TOTAL, "Январь-декабрь", "7", TOP, "33334000"),
    ])
    wage = raw("okved2", [
        ("11111000", TOTAL, "Январь-декабрь", "100"), ("11112000", TOTAL, "Январь-декабрь", "200"),
        ("22222000", TOTAL, "Январь-декабрь", "300"), ("33333000", TOTAL, "Январь-декабрь", "400", TOP, "33334000"),
    ])
    ros = external.from_frames(pop, emp, wage, CFG)
    mdict = pd.DataFrame({
        "territory_id": [1, 2, 3, 4, 11, 12],
        "oktmo8": ["22222000", "11110000", "33334000", "99999000", "11111000", "11112000"],
        "change_id_from": [np.nan, "union_1", np.nan, np.nan, np.nan, np.nan],
        "change_id_to": [np.nan, np.nan, np.nan, np.nan, "union_1", "union_1"],
    })
    out = external.attach_to_territory(ros, 2023, mdict=mdict, cfg=CFG).set_index("territory_id")
    assert out.loc[1, "match_quality"] == "exact" and out.loc[1, "population"] == 50
    assert out.loc[3, "match_quality"] == "exact" and out.loc[3, "match_via"] == "oktmo_stable" and out.loc[3, "population"] == 70
    assert out.loc[4, "match_quality"] == "missing" and np.isnan(out.loc[4, "population"])
    u = out.loc[2]
    assert u["match_quality"] == "aggregated" and u["match_via"] == "union" and u["n_sources"] == 2
    assert u["population"] == 400 and u["employees_total"] == 40
    assert u["wage_total"] == pytest.approx((100 * 10 + 200 * 30) / 40)      # взвешено по работникам
    assert u["emp_share_manufacturing"] == pytest.approx(0.25) and u["emp_share_agriculture"] == pytest.approx(0.75)
    assert u["wage_rel"] == pytest.approx(u["wage_total"] / ros["wage_total"].median())  # медиана по всем ОКТМО
    assert len(out) == len(mdict)
