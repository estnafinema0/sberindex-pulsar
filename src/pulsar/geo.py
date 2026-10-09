"""Лёгкая география МО для статичного лендинга (GitHub Pages, рендер в браузере: Plotly.js / D3).

Что делает:
  * берёт полигоны МО из справочника СберИндекса (GPKG, EPSG:4326), версию границ на заданный год;
  * оставляет только МО из панели признаков (outputs/features/panel.parquet);
  * упрощает геометрию как единое покрытие (coverage) в равноплощадной проекции Альберса для России —
    общие границы соседей упрощаются одинаково, щелей и наложений не появляется;
  * убирает мелкие острова, режет полигоны по 180-му меридиану (Чукотка), округляет координаты;
  * пишет GeoJSON / TopoJSON МО, контур регионов, репрезентативные точки и PNG-превью.

Запуск: `python -m pulsar.geo` (параметры — configs/geo.yaml). Опции: `--sizes` — только таблица размеров
для нескольких допусков, `--tolerance N` — переопределить допуск, `--no-preview` — без PNG.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml
from shapely.geometry import MultiPolygon, Polygon

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Равноплощадная коническая Альберса для России: стандартные параллели 52/64 внутри основного широтного пояса,
# осевой меридиан 100° в.д. — страна лежит симметрично, Чукотка (за 180°) не рвётся (см. docs/research/benchmarks.md §6.1).
AEA_RUSSIA = "+proj=aea +lat_1=52 +lat_2=64 +lat_0=0 +lon_0=100 +datum=WGS84 +units=m +no_defs"
WGS84 = "EPSG:4326"

# Минимальный набор свойств в GeoJSON: ключ для join с метками кластеров + подписи для тултипа.
DEFAULT_PROPS = ["territory_id", "name", "region", "type", "in_panel"]


def load_config(name: str = "geo") -> dict:
    with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------------------------------------------------
# Загрузка


def _pick_version(poly: pd.DataFrame, year: int) -> pd.DataFrame:
    """Одна строка на territory_id: версия границ, действующая в `year` (year_from ≤ year < year_to), иначе последняя."""
    poly = poly.copy()
    poly["_current"] = (poly["year_from"] <= year) & (year < poly["year_to"])
    poly = poly.sort_values(["territory_id", "_current", "year_to", "year_from"], ascending=[True, False, False, False])
    return poly.drop_duplicates("territory_id", keep="first").drop(columns="_current")


def load_polygons(year: int = 2024, gpkg: pathlib.Path | None = None, panel: pathlib.Path | None = None,
                  cfg: dict | None = None) -> gpd.GeoDataFrame:
    """Полигоны МО из панели: territory_id (int), name, region, type, in_panel, geometry (EPSG:4326); одна строка на МО."""
    cfg = cfg or load_config()
    gpkg = pathlib.Path(gpkg or ROOT / cfg["inputs"]["gpkg"])
    panel = pathlib.Path(panel or ROOT / cfg["inputs"]["panel"])
    poly = gpd.read_file(gpkg, layer=cfg["inputs"].get("layer"), engine="pyogrio")
    poly["territory_id"] = poly["territory_id"].astype(int)  # в GPKG ключ хранится строкой
    poly = _pick_version(poly, year)
    attrs = pd.read_parquet(panel, columns=["territory_id", "municipal_district_name_short", "region_name",
                                            "municipal_district_type", "in_panel"])
    attrs = attrs.rename(columns={"municipal_district_name_short": "name", "region_name": "region",
                                  "municipal_district_type": "type"})
    gdf = attrs.merge(poly[["territory_id", "year_from", "year_to", "geometry"]], on="territory_id", how="inner")
    missing = set(attrs["territory_id"]) - set(gdf["territory_id"])
    if missing:
        warnings.warn(f"без геометрии {len(missing)} МО: {sorted(missing)[:10]}…")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=poly.crs or WGS84).to_crs(WGS84)
    gdf["in_panel"] = gdf["in_panel"].astype(bool)
    return gdf.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------------
# Упрощение


def _drop_small_parts(geom, min_area: float):
    """Убирает части мультиполигона площадью < min_area (в единицах CRS²); самая большая часть остаётся всегда."""
    if geom is None or geom.is_empty or geom.geom_type != "MultiPolygon":
        return geom
    parts = list(geom.geoms)
    areas = np.array([p.area for p in parts])
    keep = areas >= min_area
    keep[int(areas.argmax())] = True
    kept = [p for p, k in zip(parts, keep) if k]
    return kept[0] if len(kept) == 1 else MultiPolygon(kept)


def simplify(gdf: gpd.GeoDataFrame, tolerance_m: float, min_island_km2: float = 0.0, clean: bool = True,
             crs: str = AEA_RUSSIA) -> gpd.GeoDataFrame:
    """Упрощение покрытия МО в метрической равноплощадной проекции с сохранением общих границ соседей.

    Алгоритм: проекция в `crs` → удаление мелких островов → (опц.) `coverage_clean` — выравнивание микронесовпадений
    границ между соседями → `coverage_simplify` (GEOS CoverageSimplifier, аналог mapshaper `-simplify keep-shapes`:
    общие рёбра упрощаются один раз, кольца не исчезают) → обратно в исходную CRS.
    МО, которые при очистке покрытия исчезли (полностью перекрыты преемником — упразднённые в отчётном году),
    получают собственную геометрию, упрощённую отдельно (`preserve_topology=True`), чтобы не потерять строки.
    """
    src_crs = gdf.crs or WGS84
    proj = gdf.set_crs(src_crs, allow_override=True).to_crs(crs)
    geoms = proj.geometry.values
    if min_island_km2 > 0:
        geoms = np.array([_drop_small_parts(g, min_island_km2 * 1e6) for g in geoms], dtype=object)
    original = geoms
    if clean:
        try:
            geoms = shapely.coverage_clean(geoms, snapping_distance=1.0)  # 1 м: только склейка численного шума на стыках
        except Exception as e:  # noqa: BLE001 — старый GEOS без CoverageCleaner: продолжаем без очистки
            warnings.warn(f"coverage_clean недоступен ({e}); упрощаем без очистки покрытия")
    simp = shapely.coverage_simplify(geoms, tolerance_m, simplify_boundary=True)
    # пропавшие при очистке (перекрытые) МО — упрощаем индивидуально по исходной геометрии
    lost = shapely.is_empty(simp) | shapely.is_missing(simp)
    if lost.any():
        simp = simp.copy()
        simp[lost] = shapely.simplify(original[lost], tolerance_m, preserve_topology=True)
    bad = ~shapely.is_valid(simp)
    if bad.any():
        simp = simp.copy()
        simp[bad] = shapely.make_valid(simp[bad])
    out = proj.copy()
    out["geometry"] = gpd.GeoSeries(simp, index=proj.index, crs=crs)
    return out.to_crs(src_crs)


# ----------------------------------------------------------------------------------------------------------------------
# Антимеридиан


def _unwrap_ring(coords: np.ndarray, ref: float | None = None) -> np.ndarray:
    """Долготы кольца делаем непрерывными (скачки > 180° убираем прибавкой ±360°); `ref` — целевое среднее
    по долготе (для дырок — чтобы они легли в ту же «копию мира», что и внешнее кольцо)."""
    c = np.asarray(coords, dtype=float)
    lon = np.degrees(np.unwrap(np.radians(c[:, 0])))
    if ref is not None:
        lon = lon + 360.0 * np.round((ref - lon.mean()) / 360.0)
    return np.column_stack([lon, c[:, 1]])


def _split_polygon(poly: Polygon) -> list[Polygon]:
    """Полигон, пересекающий 180° — на части внутри [-180, 180] через сдвиг долготы и разрез рамками."""
    ext = _unwrap_ring(np.asarray(poly.exterior.coords))
    ref = float(ext[:, 0].mean())
    holes = [_unwrap_ring(np.asarray(r.coords), ref) for r in poly.interiors]
    lo, hi = ext[:, 0].min(), ext[:, 0].max()
    if lo >= -180.0 and hi <= 180.0:
        return [poly]  # не пересекает — исходная геометрия как есть
    unwrapped = Polygon(ext, holes)
    if not unwrapped.is_valid:
        unwrapped = shapely.make_valid(unwrapped)
    parts: list[Polygon] = []
    for k in (-1, 0, 1):  # окна [-180, 180] со сдвигом k·360°
        box = shapely.box(-180.0 + 360 * k, -90.0, 180.0 + 360 * k, 90.0)
        piece = unwrapped.intersection(box)
        if piece.is_empty:
            continue
        piece = shapely.transform(piece, lambda xy, k=k: xy - np.array([360.0 * k, 0.0]))
        for g in getattr(piece, "geoms", [piece]):
            if g.geom_type == "Polygon" and g.area > 0:
                parts.append(g)
    return parts


def fix_antimeridian(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Разрезает полигоны, пересекающие 180° (Чукотка), чтобы в GeoJSON не было «полос через весь мир».

    Работает в WGS84. Полигоны, уже разрезанные по 180° в источнике, не трогает; вершины, уехавшие из-за
    численного шума на другую сторону (−180 вместо 180), возвращает на место. Все координаты результата в [-180, 180].
    """
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(WGS84)
    out = gdf.copy()
    fixed = []
    for geom in gdf.geometry.values:
        if geom is None or geom.is_empty:
            fixed.append(geom)
            continue
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        parts = [p for poly in polys for p in _split_polygon(poly)]
        fixed.append(parts[0] if len(parts) == 1 else MultiPolygon(parts))
    out["geometry"] = gpd.GeoSeries(fixed, index=gdf.index, crs=WGS84)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Экспорт


def _round_ring(coords, precision: int) -> np.ndarray | None:
    """Округляет кольцо, убирает возникшие повторы подряд; None, если кольцо выродилось (< 3 различных точек)."""
    c = np.round(np.asarray(coords, dtype=float), precision)
    keep = np.ones(len(c), dtype=bool)
    keep[1:] = np.any(c[1:] != c[:-1], axis=1)
    c = c[keep]
    if len(c) and not np.array_equal(c[0], c[-1]):
        c = np.vstack([c, c[0]])
    return c if len(c) >= 4 else None


def _round_geometry(geom, precision: int):
    """Округление координат + удаление вырожденных колец и частей (острова, схлопнувшиеся в точку/отрезок)."""
    polys = []
    for p in getattr(geom, "geoms", [geom]):
        if p.geom_type != "Polygon":
            continue
        ext = _round_ring(p.exterior.coords, precision)
        if ext is None:
            continue
        holes = [h for h in (_round_ring(r.coords, precision) for r in p.interiors) if h is not None]
        q = Polygon(ext, holes)
        if q.area > 0:
            polys.append(q)
    if not polys:
        return shapely.transform(geom, lambda xy: np.round(xy, precision))  # лучше вырожденный, чем пустой
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def _py(v):
    """numpy-скаляры → типы Python для json."""
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def export(gdf: gpd.GeoDataFrame, path: str | pathlib.Path, props: list[str] | None = None, precision: int = 3,
           topo_quantize: float = 1e5, object_name: str = "mo") -> int:
    """GeoJSON (или TopoJSON, если суффикс .topojson) с минимальными свойствами и округлёнными координатами.

    Возвращает размер файла в байтах. Координаты — WGS84, округление до `precision` знаков
    (3 знака ≈ 55 м по долготе на 60° с.ш. — достаточно при допуске упрощения ≥ 500 м).
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    props = list(props) if props is not None else [c for c in DEFAULT_PROPS if c in gdf.columns]
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(WGS84)
    geoms = [_round_geometry(g, precision) for g in gdf.geometry.values]
    if path.suffix.lower() == ".topojson":
        import topojson as tp  # опциональная зависимость (BSD-3)

        data = gpd.GeoDataFrame(gdf[props].reset_index(drop=True), geometry=gpd.GeoSeries(geoms, crs=WGS84))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            topo = tp.Topology(data, prequantize=topo_quantize, topology=True, object_name=object_name)
        txt = topo.to_json()
    else:
        feats = []
        for geom, (_, row) in zip(geoms, gdf[props].iterrows()):
            p = {k: _py(row[k]) for k in props}
            feats.append('{"type":"Feature","properties":' + json.dumps(p, ensure_ascii=False, separators=(",", ":"))
                         + ',"geometry":' + shapely.to_geojson(geom) + "}")
        txt = '{"type":"FeatureCollection","features":[' + ",".join(feats) + "]}"
    path.write_text(txt, encoding="utf-8")
    return path.stat().st_size


# ----------------------------------------------------------------------------------------------------------------------
# Производные слои


def regions_outline(gdf: gpd.GeoDataFrame, tolerance_m: float | None = None, by: str = "region",
                    crs: str = AEA_RUSSIA) -> gpd.GeoDataFrame:
    """Контур регионов: dissolve полигонов МО по `by` (coverage union в проекции), опц. упрощение, WGS84.

    Если на вход подать уже упрощённые МО, контур совпадёт с их внешними рёбрами (без двойных линий на карте).
    """
    proj = gdf[[by, "geometry"]].to_crs(crs)
    try:
        reg = proj.dissolve(by=by, method="coverage")
    except Exception:  # noqa: BLE001 — если покрытие невалидно, обычный union
        reg = proj.dissolve(by=by)
    reg = reg.reset_index()
    if tolerance_m:
        reg["geometry"] = gpd.GeoSeries(shapely.coverage_simplify(reg.geometry.values, tolerance_m), crs=crs)
    reg["geometry"] = reg.geometry.apply(lambda g: g if g.geom_type in ("Polygon", "MultiPolygon")
                                         else shapely.unary_union([p for p in g.geoms if p.geom_type == "Polygon"]))
    return fix_antimeridian(reg.to_crs(WGS84))


def centroids(gdf: gpd.GeoDataFrame, crs: str = AEA_RUSSIA) -> pd.DataFrame:
    """Репрезентативные точки МО (гарантированно внутри полигона) в WGS84 — для маркеров и подписей."""
    proj = gdf.to_crs(crs)
    # для МО, разрезанных по 180°, точка берётся в самой большой части — иначе может лечь на линию разреза
    def _main_part(g):
        return max(g.geoms, key=lambda p: p.area) if g.geom_type == "MultiPolygon" else g
    pts = gpd.GeoSeries([_main_part(g).representative_point() for g in proj.geometry.values], crs=crs).to_crs(WGS84)
    cols = [c for c in DEFAULT_PROPS if c in gdf.columns]
    out = gdf[cols].copy().reset_index(drop=True)
    out["lat"] = np.round(pts.y.values, 4)
    out["lon"] = np.round(pts.x.values, 4)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Превью


def preview_png(mo: gpd.GeoDataFrame, regions: gpd.GeoDataFrame, path: pathlib.Path, crs: str = AEA_RUSSIA,
                region_filter: list[str] | None = None, title: str = "", color_by_index: bool = False,
                separate_panels: bool = False) -> None:
    """PNG через matplotlib в проекции Альберса: вся Россия или зум на выбранные регионы."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = mo if region_filter is None else mo[mo["region"].isin(region_filter)]
    r = regions if region_filter is None else regions[regions["region"].isin(region_filter)]
    m, r = m.to_crs(crs), r.to_crs(crs)
    # панели: одна на всю страну / один регион; при нескольких регионах (напр. Москва и СПб) — по панели на регион,
    # иначе далёкие друг от друга города превращаются в точки на общем холсте
    panels = [None] if region_filter is None or not separate_panels else list(region_filter)
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 9) if region_filter is None else (9 * len(panels), 9), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, reg in zip(axes, panels):
        mm = m if reg is None else m[m["region"] == reg]
        rr = r if reg is None else r[r["region"] == reg]
        if color_by_index:
            mm = mm.assign(_c=np.arange(len(mm)) % 12)
            mm.plot(ax=ax, column="_c", cmap="Set3", edgecolor="#333333", linewidth=0.3)
        else:
            mm.plot(ax=ax, color=np.where(mm["in_panel"], "#cfe3f2", "#eeeeee"), edgecolor="#6b8aa6", linewidth=0.15)
        rr.boundary.plot(ax=ax, color="#1f2a44", linewidth=0.6 if region_filter is None else 1.0)
        ax.set_axis_off()
        ax.set_title(f"{reg}: {len(mm)} МО" if reg else (title or f"{len(m)} МО · проекция Альберса (lon0=100, параллели 52/64)"))
    if title and len(panels) > 1:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------------
# Отчёт о размерах и CLI


def size_report(gdf: gpd.GeoDataFrame, tolerances: list[float], min_island_km2: float, precision: int,
                topo_quantize: float, tmp_dir: pathlib.Path) -> pd.DataFrame:
    """Размеры GeoJSON/TopoJSON и число вершин для нескольких допусков (файлы пишутся во временный каталог)."""
    rows = []
    for tol in tolerances:
        s = fix_antimeridian(simplify(gdf, tol, min_island_km2))
        gj = export(s, tmp_dir / f"mo_{int(tol)}.geojson", precision=precision)
        tj = export(s, tmp_dir / f"mo_{int(tol)}.topojson", precision=precision, topo_quantize=topo_quantize)
        rows.append({"tolerance_m": tol, "vertices": int(shapely.get_num_coordinates(s.geometry.values).sum()),
                     "geojson_mb": round(gj / 1e6, 2), "topojson_mb": round(tj / 1e6, 2)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="География МО для лендинга")
    ap.add_argument("--tolerance", type=float, default=None, help="допуск упрощения, м (по умолчанию из configs/geo.yaml)")
    ap.add_argument("--sizes", action="store_true", help="только таблица размеров для simplify.size_report_tolerances")
    ap.add_argument("--no-preview", action="store_true", help="не рисовать PNG")
    args = ap.parse_args(argv)

    cfg = load_config()
    out_dir = ROOT / cfg["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    scfg, ecfg = cfg["simplify"], cfg["export"]
    tol = args.tolerance or scfg["tolerance_m"]

    t0 = time.time()
    gdf = load_polygons(cfg["year"], cfg=cfg)
    print(f"полигонов: {len(gdf)} (вершин {shapely.get_num_coordinates(gdf.geometry.values).sum():,}), "
          f"{time.time() - t0:.1f} с")

    if args.sizes:
        rep = size_report(gdf, scfg["size_report_tolerances"], scfg["min_island_km2"], ecfg["precision"],
                          ecfg["topo_quantize"], out_dir / "_sizes")
        print(rep.to_string(index=False))
        return

    mo = fix_antimeridian(simplify(gdf, tol, scfg["min_island_km2"]))
    b = mo.bounds
    assert b.minx.min() >= -180 and b.maxx.max() <= 180, "координаты вне [-180, 180]"
    pb = mo.explode(index_parts=False).bounds  # ширину проверяем по частям: у Чукотки части законно лежат по обе стороны 180°
    wide = pb[(pb.maxx - pb.minx) >= 180]
    assert wide.empty, f"части шириной ≥ 180° (не разрезаны по антимеридиану): {mo.loc[wide.index.unique(), 'name'].tolist()}"
    print(f"упрощение {tol:.0f} м, острова < {scfg['min_island_km2']} км² убраны: "
          f"вершин {shapely.get_num_coordinates(mo.geometry.values).sum():,}")

    files = cfg["outputs"]["files"]
    sizes = {"mo.geojson": export(mo, out_dir / files["mo_geojson"], precision=ecfg["precision"])}
    try:
        sizes["mo.topojson"] = export(mo, out_dir / files["mo_topojson"], precision=ecfg["precision"],
                                      topo_quantize=ecfg["topo_quantize"])
    except ImportError:
        print("пакет topojson не установлен — TopoJSON пропущен")

    regions = regions_outline(mo, tolerance_m=scfg.get("regions_tolerance_m"))
    sizes["regions.geojson"] = export(regions, out_dir / files["regions_geojson"], props=["region"],
                                      precision=ecfg["precision"])
    cen = centroids(mo)
    cen.to_csv(out_dir / files["centroids_csv"], index=False)
    sizes["centroids.csv"] = (out_dir / files["centroids_csv"]).stat().st_size

    # детальный слой городов федерального значения для врезки (внутригородские территории на общей карте слишком мелкие)
    det = cfg.get("cities_detail")
    if det:
        cities = gdf[gdf["region"].isin(det["regions"])]
        cities = fix_antimeridian(simplify(cities, det["tolerance_m"], 0.0))
        sizes[files["cities_geojson"]] = export(cities, out_dir / files["cities_geojson"], precision=ecfg["precision"])

    for k, v in sizes.items():
        print(f"  {k:>20}: {v / 1e6:6.2f} МБ")

    if not args.no_preview:
        preview_png(mo, regions, out_dir / files["preview_map"])
        preview_png(mo, regions, out_dir / files["preview_moscow"], region_filter=["Москва", "Московская область"],
                    title="Москва и Московская область: МО после упрощения", color_by_index=True)
        if det:
            preview_png(cities, regions, out_dir / files["preview_cities"], region_filter=det["regions"],
                        title=f"Города федерального значения, допуск {det['tolerance_m']} м", color_by_index=True,
                        separate_panels=True)
    print(f"готово за {time.time() - t0:.1f} с → {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
