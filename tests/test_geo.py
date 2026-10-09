"""Тесты географии МО на синтетических полигонах (без чтения GPKG): антимеридиан, упрощение, экспорт."""
from __future__ import annotations

import json
import pathlib
import sys

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from pulsar import geo  # noqa: E402


def _lon_range(geom) -> tuple[float, float]:
    xs = shapely.get_coordinates(geom)[:, 0]
    return float(xs.min()), float(xs.max())


def _noisy_blob(cx: float, cy: float, r: float, n: int = 400, seed: int = 0) -> Polygon:
    """«Изрезанный» полигон с n вершинами: окружность радиуса r (в градусах) с мелким шумом."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = r * (1 + 0.02 * rng.standard_normal(n))
    return Polygon(np.column_stack([cx + rr * np.cos(t) * 1.8, cy + rr * np.sin(t)]))


# ----------------------------------------------------------------------------------------------------------- антимеридиан


def test_fix_antimeridian_splits_crossing_polygon():
    # «Чукотка»: четырёхугольник 170°…−170° в.д., записанный со скачком долготы через 180°
    poly = Polygon([(170, 62), (-170, 62), (-170, 68), (170, 68)])
    gdf = gpd.GeoDataFrame({"territory_id": [1]}, geometry=[poly], crs="EPSG:4326")
    out = geo.fix_antimeridian(gdf)
    geom = out.geometry.iloc[0]
    assert geom.geom_type == "MultiPolygon" and len(geom.geoms) == 2
    lo, hi = _lon_range(geom)
    assert lo >= -180 and hi <= 180
    for part in geom.geoms:
        plo, phi = _lon_range(part)
        assert phi - plo <= 10 + 1e-9  # каждая часть лежит с одной стороны от 180°
    # площадь (в градусах²) сохранена: 20° × 6°
    assert geom.area == pytest.approx(20 * 6, rel=1e-6)


def test_fix_antimeridian_keeps_ordinary_polygon_and_fixes_noise():
    ordinary = Polygon([(37, 55), (38, 55), (38, 56), (37, 56)])
    # западная половина, у которой одна вершина «уехала» на +180 из-за численного шума обратной проекции
    noisy = Polygon([(180.0, 64), (-179.5, 64), (-179.5, 65), (-180.0, 65)])
    gdf = gpd.GeoDataFrame({"territory_id": [1, 2]}, geometry=[ordinary, noisy], crs="EPSG:4326")
    out = geo.fix_antimeridian(gdf)
    assert out.geometry.iloc[0].equals(ordinary)
    g2 = out.geometry.iloc[1]
    lo, hi = _lon_range(g2)
    assert lo >= -180 and hi <= -179.5 + 1e-9  # всё в западной копии мира, ширина 0.5°, а не 359.5°
    assert g2.area == pytest.approx(0.5, rel=1e-6)


# -------------------------------------------------------------------------------------------------------------- упрощение


def test_simplify_reduces_vertices_and_keeps_area():
    # два соседних «района» под Москвой с общей границей + остров-крошка
    a = _noisy_blob(37.0, 55.5, 0.4, seed=1)
    b = shapely.affinity.translate(a, xoff=1.2).difference(a)
    island = Polygon([(36.0, 55.0), (36.01, 55.0), (36.01, 55.01), (36.0, 55.01)])  # ~0.7 км²
    gdf = gpd.GeoDataFrame({"territory_id": [1, 2]}, geometry=[a, shapely.MultiPolygon([b, island])], crs="EPSG:4326")
    out = geo.simplify(gdf, tolerance_m=2000, min_island_km2=5)
    n0 = shapely.get_num_coordinates(gdf.geometry.values)
    n1 = shapely.get_num_coordinates(out.geometry.values)
    assert (n1 < n0).all() and n1.sum() < 0.5 * n0.sum()
    assert out.crs == gdf.crs
    assert out.geometry.is_valid.all()
    # площади в метрической проекции — в пределах 5 %
    a0 = gdf.to_crs(geo.AEA_RUSSIA).area.values
    a1 = out.to_crs(geo.AEA_RUSSIA).area.values
    assert np.all(np.abs(a1 - a0) / a0 < 0.05)
    # остров < 5 км² убран, крупная часть осталась
    assert out.geometry.iloc[1].geom_type == "Polygon"
    # общая граница упрощена согласованно: соседи по-прежнему не накладываются и не расходятся
    inter = out.geometry.iloc[0].intersection(out.geometry.iloc[1])
    assert inter.area < 1e-9


# ---------------------------------------------------------------------------------------------------------------- экспорт


def test_export_writes_valid_geojson_with_props_and_rounding(tmp_path):
    poly = Polygon([(37.123456, 55.654321), (37.98765, 55.654321), (37.98765, 56.111111), (37.123456, 56.111111)])
    gdf = gpd.GeoDataFrame(
        {"territory_id": np.array([42], dtype=np.int64), "name": ["Тестовый"], "region": ["Регион"],
         "type": ["муниципальный район"], "in_panel": np.array([True]), "extra": [1.5]},
        geometry=[poly], crs="EPSG:4326")
    path = tmp_path / "mo.geojson"
    size = geo.export(gdf, path, props=["territory_id", "name", "region", "type", "in_panel"], precision=3)
    assert size == path.stat().st_size > 0
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["type"] == "FeatureCollection" and len(data["features"]) == 1
    feat = data["features"][0]
    assert feat["properties"] == {"territory_id": 42, "name": "Тестовый", "region": "Регион",
                                  "type": "муниципальный район", "in_panel": True}
    assert "extra" not in feat["properties"]
    coords = np.array(feat["geometry"]["coordinates"][0])
    assert np.allclose(coords, np.round(coords, 3))  # не больше 3 знаков
    assert coords[0].tolist() == [37.123, 55.654]
    # файл читается geopandas и геометрия совпадает с округлённой исходной
    back = gpd.read_file(path)
    assert back.geometry.iloc[0].is_valid
    assert back.geometry.iloc[0].area == pytest.approx(poly.area, rel=1e-2)


def test_export_topojson(tmp_path):
    pytest.importorskip("topojson")
    a = Polygon([(30, 50), (31, 50), (31, 51), (30, 51)])
    b = Polygon([(31, 50), (32, 50), (32, 51), (31, 51)])  # общая граница с a → один общий arc
    gdf = gpd.GeoDataFrame({"territory_id": [1, 2], "name": ["A", "B"]}, geometry=[a, b], crs="EPSG:4326")
    path = tmp_path / "mo.topojson"
    geo.export(gdf, path, props=["territory_id", "name"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["type"] == "Topology" and "mo" in data["objects"]
    assert len(data["objects"]["mo"]["geometries"]) == 2
    assert len(data["arcs"]) == 3  # a-только, общая, b-только
    props = [g["properties"]["territory_id"] for g in data["objects"]["mo"]["geometries"]]
    assert sorted(props) == [1, 2]


def test_centroids_inside_polygons_wgs84():
    a = _noisy_blob(60.0, 57.0, 0.5)
    gdf = gpd.GeoDataFrame({"territory_id": [7], "name": ["X"], "region": ["R"], "type": ["t"], "in_panel": [False]},
                           geometry=[a], crs="EPSG:4326")
    c = geo.centroids(gdf)
    assert list(c.columns) == ["territory_id", "name", "region", "type", "in_panel", "lat", "lon"]
    assert a.contains(shapely.Point(c.lon.iloc[0], c.lat.iloc[0]))
