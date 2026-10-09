"""Тесты интерпретации типов на синтетике: 4 гауссовых кластера с известными сдвигами по признакам."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pulsar import interpret  # noqa: E402

NAMES = ["clr_Продовольствие", "clr_Здоровье", "clr_Маркетплейсы", "clr_Общественное питание",
         "clr_Транспорт", "clr_Прочее", "log_level_rel"]
RAW_NAMES = ["Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание", "Транспорт", "Прочее", "total_per_capita"]
# Сдвиги центров (в σ): тип 0 — города (уровень + общепит), 1 — село/продовольствие, 2 — село/онлайн, 3 — север/прочее.
SHIFTS = np.array([
    [-0.5, 0.0, 0.0, 1.5, 0.0, 0.0, 2.0],
    [2.0, 0.0, -0.5, -0.5, 0.0, 0.0, -1.5],
    [-0.5, 0.0, 2.0, -0.5, 0.0, 0.0, -1.2],
    [-0.5, 0.0, -1.5, 0.0, 0.0, 2.0, 1.2],
])
SIZES = [120, 150, 100, 80]
CFG = interpret.load_config()


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(SHIFTS[k], 0.5, size=(n, len(NAMES))) for k, n in enumerate(SIZES)])
    labels = np.repeat(np.arange(len(SIZES)), SIZES)
    # исходные единицы: доли по softmax от CLR-подобных значений, итог в ₽ — экспонента уровня
    sh = np.exp(X[:, :6]); sh = sh / sh.sum(axis=1, keepdims=True)
    tot = 20000 * np.exp(0.6 * X[:, 6])
    raw = np.column_stack([sh, tot])
    N = len(labels)
    meta = pd.DataFrame({"territory_id": np.arange(1000, 1000 + N), "name": [f"МО {i}" for i in range(N)],
                         "region_name": rng.choice(["Регион А", "Регион Б", "Регион В"], N),
                         "municipal_district_type": np.where(labels == 0, "Городской округ", "Муниципальный район"),
                         "municipal_district_status": rng.choice(["обычный", "столица"], N),
                         "lat": rng.uniform(50, 70, N), "lon": rng.uniform(30, 150, N),
                         "market_access": rng.normal(size=N)})
    return X, raw, labels, meta


# --------------------------------------------------------------------------- 1. Миркин
def test_mirkin_profile_decomposition(synth):
    X, _, labels, _ = synth
    prof = interpret.mirkin_profile(X, labels, NAMES)
    assert set(["type", "feature", "n", "mean", "rel_dev", "contrib", "contrib_pct_T", "contrib_pct_B"]) <= set(prof.columns)
    assert len(prof) == len(SIZES) * len(NAMES)
    T, B, W = prof.attrs["T"], prof.attrs["B"], prof.attrs["W"]
    assert prof["contrib"].sum() == pytest.approx(B)
    assert T == pytest.approx(B + W, rel=1e-9)   # пифагорово разложение
    assert 0.0 <= prof.attrs["B_over_T"] <= 1.0
    assert prof.attrs["B_over_T"] > 0.7        # кластеры хорошо разделены
    assert prof["contrib_pct_T"].sum() == pytest.approx(100 * prof.attrs["B_over_T"])
    assert prof["contrib_pct_B"].sum() == pytest.approx(100.0)


def test_mirkin_top_feature_matches_shift(synth):
    X, _, labels, _ = synth
    prof = interpret.mirkin_profile(X, labels, NAMES)
    expected = {0: "log_level_rel", 1: "clr_Продовольствие", 2: "clr_Маркетплейсы", 3: "clr_Прочее"}
    for k, feat in expected.items():
        tf = interpret.top_features(prof, k, n=3)
        assert tf.iloc[0]["feature"] == feat
        assert tf.iloc[0]["sign"] == "+"
    # для стандартизованных признаков rel_dev — в единицах СКО и совпадает по знаку со сдвигом
    wide = interpret.profile_wide(prof, "rel_dev")
    assert wide.loc[1, "log_level_rel"] < 0 < wide.loc[0, "log_level_rel"]


def test_mirkin_rel_dev_with_nonzero_mean():
    X = np.array([[1.0, 10.0]] * 5 + [[3.0, 10.0]] * 5)
    prof = interpret.mirkin_profile(X, [0] * 5 + [1] * 5, ["a", "b"])
    r = prof.set_index(["type", "feature"])["rel_dev"]
    assert r[(0, "a")] == pytest.approx(-0.5) and r[(1, "a")] == pytest.approx(0.5)   # (c − m)/|m|, m = 2
    assert r[(0, "b")] == pytest.approx(0.0)


# --------------------------------------------------------------------------- 2. паспорт типа
def test_readable_profile_units(synth):
    _, raw, labels, _ = synth
    rd = interpret.readable_profile(raw, labels, RAW_NAMES, cfg=CFG)
    assert interpret.ALL_LABEL in rd.index and set(range(4)) <= set(rd.index)
    assert "Продовольствие, %" in rd.columns and "total_per_capita, ₽" in rd.columns
    assert rd.loc[interpret.ALL_LABEL, "total_per_capita, % к медиане"] == pytest.approx(100.0)
    assert rd.loc[0, "total_per_capita, % к медиане"] > 100 > rd.loc[1, "total_per_capita, % к медиане"]
    assert rd.loc[1, "Продовольствие, %"] > rd.loc[interpret.ALL_LABEL, "Продовольствие, %"]
    assert rd["n"].iloc[:-1].sum() == len(labels)


# --------------------------------------------------------------------------- 3. внешняя проверка
def test_kruskal_eta_separated_vs_noise():
    rng = np.random.default_rng(1)
    labels = np.repeat([0, 1, 2], 100)
    perfect = labels * 10.0 + rng.normal(0, 0.01, 300)
    noise = rng.normal(size=300)
    r1 = interpret.kruskal_eta(perfect, labels)
    r0 = interpret.kruskal_eta(noise, labels)
    ceiling = 1 - 1 / 3**2                            # потолок η²_H для k=3 равных групп (ранговая статистика)
    assert r1["eta2_H"] > 0.98 * ceiling and r1["p"] < 1e-10
    assert r0["eta2_H"] < 0.05 and r0["p"] > 0.01
    assert set(r1) >= {"H", "p", "eta2_H"}
    # NaN отбрасываются
    with_nan = perfect.copy(); with_nan[:5] = np.nan
    assert interpret.kruskal_eta(with_nan, labels)["n"] == 295


def test_external_validation_sorted_with_expected_columns(synth):
    X, _, labels, meta = synth
    rng = np.random.default_rng(2)
    N = len(labels)
    ext = pd.DataFrame({
        "territory_id": meta["territory_id"],
        "wage_total": 30000 + 15000 * (labels == 0) + rng.normal(0, 2000, N),   # сильно связана с типом
        "population": rng.lognormal(10, 1, N),                                  # шум
        "top_sector": np.where(labels == 3, "добыча", rng.choice(["торговля", "сельское хозяйство"], N)),
    })
    lab = pd.Series(labels, index=meta["territory_id"])
    res = interpret.external_validation(ext, lab, meta=meta, cfg=CFG)
    for c in ["variable", "kind", "stat", "strength", "p", "nmi", "median_0", "median_all"]:
        assert c in res.columns
    assert res["strength"].is_monotonic_decreasing
    assert set(res["variable"]) >= {"wage_total", "population", "top_sector", "municipal_district_type", "region_name"}
    row = res.set_index("variable")
    assert row.loc["wage_total", "kind"] == "numeric" and row.loc["wage_total", "strength"] > row.loc["population", "strength"]
    assert row.loc["top_sector", "kind"] == "categorical" and row.loc["top_sector", "strength"] > row.loc["region_name", "strength"]
    assert row.loc["municipal_district_type", "strength"] == pytest.approx(1.0)   # тип МО идеально отделяет тип 0
    assert "top_sector" in res.attrs["tables"] and res.attrs["tables"]["top_sector"].shape[1] == 4
    assert row.loc["wage_total", "median_0"] > row.loc["wage_total", "median_1"]


# --------------------------------------------------------------------------- 4. дерево правил
def test_rule_tree_accuracy_and_russian_names(synth):
    _, raw, labels, _ = synth
    res = interpret.rule_tree(raw, labels, RAW_NAMES, max_depth=3, min_leaf=20, cfg=CFG)
    assert res["accuracy"] > 0.9 and res["cv_accuracy"] > 0.9
    assert "Доля «Продовольствие», %" in res["text"] and "Итог на жителя, ₽" in res["text"]
    assert "share_" not in res["text"] and "feature_" not in res["text"]
    assert res["tree"].get_depth() <= 3
    assert all(0 < p <= 1 for p in res["leaf_purity"].values())


# --------------------------------------------------------------------------- 5. типичные и пограничные
def test_typical_closer_than_border(synth):
    X, _, labels, meta = synth
    tb = interpret.typical_and_border(X, labels, meta, n=3)
    assert set(tb["role"]) == {"typical", "border"} and len(tb) == 4 * 6
    for k in range(4):
        typ = tb[(tb["type"] == k) & (tb["role"] == "typical")]
        bor = tb[(tb["type"] == k) & (tb["role"] == "border")]
        assert typ["dist_own"].max() <= bor["dist_own"].min()
        assert typ["margin"].min() >= bor["margin"].max()
    assert {"territory_id", "name", "region_name", "second_type"} <= set(tb.columns)
    m = interpret.membership_margin(X, labels)
    assert m.shape == (len(labels),) and np.all(m >= -1) and np.all(m <= 1)
    assert np.median(m) > 0.3                      # большинство МО уверенно внутри своего типа
    # точка ровно посередине двух центров → зазор 0
    C = np.vstack([X[labels == j].mean(axis=0) for j in range(4)])
    mid = ((C[0] + C[1]) / 2)[None, :]
    assert abs(interpret.membership_margin(mid, [0], centers=C)[0]) < 1e-9


# --------------------------------------------------------------------------- 6. имена
def test_name_types_and_check_names(synth):
    X, raw, labels, _ = synth
    prof = interpret.mirkin_profile(X, labels, NAMES)
    rd = interpret.readable_profile(raw, labels, RAW_NAMES, cfg=CFG)
    names = interpret.name_types(prof, rd, None, CFG)
    assert set(names) == {0, 1, 2, 3}
    assert names[0]["name"].startswith("Крупные города")
    assert names[1]["name"].startswith("Сельская периферия: продовольственная")
    assert names[2]["name"].startswith("Сельская периферия: онлайн")
    assert names[3]["name"].startswith("Северные и ресурсные")
    assert all(v["description"] for v in names.values())
    assert interpret.check_names(prof, names, CFG) == []
    # заведомо неверное имя: тип 1 (село, низкий уровень) назван «городами»
    wrong = {**names, 1: {**names[1], "name": "Крупные города и столичные центры", "rule": None}}
    problems = interpret.check_names(prof, wrong, CFG)
    assert problems and any("тип 1" in p and "уровень трат" in p for p in problems)
    # имя вне правил не проверяется
    assert interpret.check_names(prof, {0: {"name": "Произвольное имя"}}, CFG) == []


# --------------------------------------------------------------------------- 7. SHAP
def test_shap_importance_top_feature(synth):
    pytest.importorskip("lightgbm"); pytest.importorskip("shap")
    X, _, labels, _ = synth
    imp = interpret.shap_importance(X, labels, NAMES, seed=0)
    assert imp is not None and imp.shape == (4, len(NAMES)) and (imp.to_numpy() >= 0).all()
    assert imp.loc[1].idxmax() == "clr_Продовольствие"
    assert imp.loc[2].idxmax() == "clr_Маркетплейсы"


# --------------------------------------------------------------------------- 8. сводка
def test_type_summary_is_json_serializable(synth):
    X, raw, labels, meta = synth
    prof = interpret.mirkin_profile(X, labels, NAMES)
    rd = interpret.readable_profile(raw, labels, RAW_NAMES, cfg=CFG)
    tb = interpret.typical_and_border(X, labels, meta, n=3)
    names = interpret.name_types(prof, rd, None, CFG)
    rules = interpret.rule_tree(raw, labels, RAW_NAMES, min_leaf=20, cfg=CFG)
    ext = pd.DataFrame({"territory_id": meta["territory_id"], "wage_total": 1000.0 * labels})
    ev = interpret.external_validation(ext, pd.Series(labels, index=meta["territory_id"]), meta=meta, cfg=CFG)
    s = interpret.type_summary(labels, meta, prof, rd, tb, names, external_val=ev, rules=rules)
    txt = json.dumps(s, ensure_ascii=False)
    assert s["k"] == 4 and s["n_total"] == len(labels) and len(s["types"]) == 4
    t0 = s["types"][0]
    assert t0["name"] and t0["n"] == SIZES[0] and len(t0["typical"]) == 3 and len(t0["border"]) == 3
    assert "log_level_rel" in t0["profile"] and "external_medians" in t0 and "wage_total" in t0["external_medians"]
    assert "Доля" in s["rule_tree"]["text"] and "Крупные города" in txt
