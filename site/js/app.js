/* Pulsar landing — весь интерактив. Без сборки; Plotly.js подключён с CDN.
 * Данные: site/data/*.json|geojson по контракту docs/site_data_contract.md.
 * Модули (функции): load → header → map (+legend, controls, passport) → types → dynamics → methods → cases → toc.
 */
(() => {
  "use strict";

  // ---------------------------------------------------------------- constants
  const DATA_DIR = "data/";
  const FALLBACK_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#8C6D31", "#332288", "#999933", "#882255"];
  const NODATA = "#c9ccd1";
  const DIMMED = "#e1e4e8";
  const CONF_THRESHOLD = 0.7;
  const CAT_RAMP = ["#1f4e5f", "#2f7183", "#4f94a6", "#7db5c3", "#acd2dc", "#dbe9ed"];
  const MONTH_NAMES = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"];
  const MONTH_SHORT = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];
  const METHOD_NAMES = { kefrin: "KEFRiN (ρ=ξ=1)", kefrin_balanced: "KEFRiN сбалансированный", temporal_leiden: "temporal Leiden", leiden: "Leiden", spectral: "Spectral", kmeans: "k-means", ward: "Ward" };
  const METRIC_NAMES = { SW: "SW", CH: "CH", S_Dbw: "S_Dbw", AVI: "AVI", AVU: "AVU", MQ: "MQ", ANUI: "ANUI", borda: "Борда", threshold_rank: "Порог. ранг", win_rate: "Доля побед", n1: "плохо", n2: "средне", n3: "хорошо", mean_win_rate: "доля лидерства" };
  const DEFAULT_DIRECTIONS = { SW: "max", CH: "max", S_Dbw: "min", AVI: "max", AVU: "min", MQ: "max", ANUI: "max", borda: "max", threshold_rank: "min", win_rate: "max", Q: "max", DBI: "min" };
  const LAYER_NAMES = { fused: "Слияние SNF (итог)", behaviour: "Поведение", gravity: "География", behavior: "Поведение", comovement: "Со-движение", geo: "География", "behavior+comovement": "Поведение + со-движение", "behavior+geo": "Поведение + география", "comovement+geo": "Со-движение + география" };
  const EVENT_NAMES = { split: "Расщепление", merge: "Слияние", continue: "Продолжение", grow: "Рост", shrink: "Сжатие", birth: "Рождение", death: "Исчезновение", form: "Рождение", dissolve: "Исчезновение" };
  const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] };
  const STATIC_CONFIG = { displayModeBar: false, responsive: true };
  const FONT = { family: '"SB Sans Text", "SBSansText", Onest, system-ui, -apple-system, Segoe UI, Roboto, sans-serif', size: 12, color: "#262626" };
  const ACCENT = "#08A652";

  const fmtInt = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 });
  const fmt1 = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
  const fmtPct = (x, d = 0) => (x == null || Number.isNaN(x) ? "–" : new Intl.NumberFormat("ru-RU", { maximumFractionDigits: d, minimumFractionDigits: d }).format(100 * x) + " %");

  // ---------------------------------------------------------------- state
  const S = {
    meta: null, types: [], labels: {}, trans: null, methods: null, cases: [], mo: null, regions: null,
    K: 0, months: [], T: 0,
    ids: [],               // territory_id (строки) в порядке признаков mo.geojson
    featById: new Map(),
    t: 0,                  // индекс выбранного месяца
    mode: "confirmed",     // confirmed | raw
    unc: false,            // слой неопределённости
    focus: null,           // null | {kind:"type", type} | {kind:"flow", from, to} | {kind:"ids", ids:Set, label}
    selected: null,        // territory_id выбранного МО
    timer: null,
    mapReady: false,
    lastRenderMs: null,
  };

  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, attrs = {}, ...children) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "style") n.style.cssText = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (k === "html") n.innerHTML = v;
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) if (c != null) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
    return n;
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ---------------------------------------------------------------- helpers
  const typeColor = (k) => (k == null || k < 0 ? NODATA : (S.types[k] && S.types[k].color) || FALLBACK_PALETTE[k % FALLBACK_PALETTE.length]);
  const typeName = (k) => (k == null || k < 0 ? "нет данных" : (S.types[k] && S.types[k].name) || `Тип ${k}`);
  const typeShort = (k) => (k == null || k < 0 ? "нет данных" : (S.types[k] && (S.types[k].short || S.types[k].name)) || `Тип ${k}`);
  const monthLabel = (i) => { const [y, m] = S.months[i].split("-"); return `${MONTH_NAMES[+m - 1]} ${y}`; };
  const monthShort = (m) => { const [y, mm] = m.split("-"); return `${MONTH_SHORT[+mm - 1]} ${y.slice(2)}`; };
  const hexA = (hex, a) => { const h = hex.replace("#", ""); const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16); return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`; };
  const seqOf = (L) => (L ? (S.mode === "raw" ? L.type : L.confirmed || L.type) : null);
  const typeAt = (id, t = S.t) => { const s = seqOf(S.labels[id]); return s ? (s[t] ?? -1) : -1; };
  const confAt = (id, t = S.t) => { const L = S.labels[id]; return L && L.confidence ? L.confidence[t] : null; };
  const idxOfMonth = (m, dflt) => { const i = S.months.indexOf(m); return i >= 0 ? i : dflt; };
  const flowIdx = () => [idxOfMonth(S.trans && S.trans.from_month, 11), idxOfMonth(S.trans && S.trans.to_month, S.T - 1)];
  const opacityFromConf = (c) => (c == null ? 1 : c < CONF_THRESHOLD ? 0.12 + 0.3 * Math.max(0, c) : 0.45 + 0.55 * ((c - CONF_THRESHOLD) / (1 - CONF_THRESHOLD)));
  const baseLayout = (extra = {}) => ({ font: FONT, separators: ", ", paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { l: 40, r: 10, t: 10, b: 30 }, hoverlabel: { font: { family: FONT.family, size: 12 } }, ...extra });
  const axis = (extra = {}) => ({ gridcolor: "#E4E8EB", zerolinecolor: "#D0D5D9", linecolor: "#E4E8EB", tickfont: { size: 11, color: "#808080" }, ...extra });

  // ---------------------------------------------------------------- load
  async function fetchJSON(name, optional = false) {
    try {
      const r = await fetch(DATA_DIR + name, { cache: "no-cache" });
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
      return await r.json();
    } catch (e) {
      if (optional) { console.warn("[pulsar] нет файла", name, e); return null; }
      throw e;
    }
  }

  // d3-geo (внутри Plotly geo) считает внешним кольцом то, что обходится ПО часовой стрелке; GeoJSON RFC 7946 —
  // наоборот. Без перемотки полигон заливает «весь глобус минус МО». Перематываем при загрузке, чтобы контракт
  // не зависел от того, чем выгружена геометрия (geopandas, mapshaper, …).
  function ringArea(ring) {
    let a = 0;
    for (let i = 0, n = ring.length - 1; i < n; i++) a += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
    return a / 2;
  }
  function rewindPolygon(rings) {
    return rings.map((r, i) => { const cw = ringArea(r) < 0; return (i === 0 ? cw : !cw) ? r : r.slice().reverse(); });
  }
  function rewindGeoJSON(fc) {
    for (const f of fc.features || []) {
      const g = f.geometry; if (!g) continue;
      if (g.type === "Polygon") g.coordinates = rewindPolygon(g.coordinates);
      else if (g.type === "MultiPolygon") g.coordinates = g.coordinates.map(rewindPolygon);
    }
    return fc;
  }

  async function loadData() {
    const [meta, types, labels, trans, methods, cases, mo, regions] = await Promise.all([
      fetchJSON("meta.json"), fetchJSON("types.json"), fetchJSON("labels.json"), fetchJSON("transitions.json"),
      fetchJSON("methods.json", true), fetchJSON("cases.json", true), fetchJSON("mo.geojson"), fetchJSON("regions.geojson", true),
    ]);
    rewindGeoJSON(mo);
    if (regions) rewindGeoJSON(regions);
    Object.assign(S, { meta, types: [...types].sort((a, b) => a.id - b.id), labels, trans, methods, cases: cases || [], mo, regions });
    S.months = meta.months;
    S.T = S.months.length;
    S.K = meta.k ?? types.length;
    S.t = S.T - 1;
    S.ids = mo.features.map((f) => String(f.properties.territory_id));
    mo.features.forEach((f) => S.featById.set(String(f.properties.territory_id), f));
  }

  // ---------------------------------------------------------------- 1. header
  function renderHeader() {
    const m = S.meta;
    if (m.mock) $("#mock-banner").hidden = false;
    const kpi = (v, l) => el("div", { class: "kpi" }, el("div", { class: "v" }, v), el("div", { class: "l" }, l));
    $("#kpis").replaceChildren(
      kpi(fmtInt.format(m.n_all ?? S.ids.length), `муниципальных образований, из них ${fmtInt.format(m.n_panel ?? 0)} с полной историей`),
      kpi(String(S.T), `месяцев: ${monthShort(S.months[0])} – ${monthShort(S.months[S.T - 1])}`),
      kpi(String(S.K), `типов локальных экономик (метод ${METHOD_NAMES[m.method] || m.method})`),
      kpi(S.trans && S.trans.stable_share != null ? fmtPct(S.trans.stable_share) : "–", "МО ни разу не сменили подтверждённый тип"),
    );
    if (m.attribution) $("#attribution").textContent = m.attribution;
    if (m.generated) $("#generated").textContent = `Данные собраны: ${new Date(m.generated).toLocaleString("ru-RU")}`;
  }

  // ---------------------------------------------------------------- 2. map
  function colorscale() {
    // дискретная шкала: 0..K-1 — типы, K — нет данных, K+1 — приглушённые
    const cols = [...Array(S.K).keys()].map(typeColor).concat([NODATA, DIMMED]);
    const N = cols.length, cs = [];
    cols.forEach((c, i) => { cs.push([i / N, c], [(i + 1) / N, c]); });
    return cs;
  }

  function matchesFocus(id, ty) {
    const f = S.focus;
    if (!f) return true;
    if (f.kind === "type") return ty === f.type;
    if (f.kind === "flow") {
      const L = S.labels[id]; if (!L) return false;
      const c = L.confirmed || L.type; const [i0, i1] = flowIdx();
      return c[i0] === f.from && c[i1] === f.to;
    }
    if (f.kind === "ids") return f.ids.has(id);
    return true;
  }

  function mapArrays() {
    const n = S.ids.length, z = new Array(n), op = new Array(n);
    for (let i = 0; i < n; i++) {
      const id = S.ids[i], ty = typeAt(id);
      z[i] = ty < 0 ? S.K : matchesFocus(id, ty) ? ty : S.K + 1;
      const c = confAt(id);
      op[i] = S.unc && ty >= 0 ? opacityFromConf(c) : 1;
    }
    return { z, op };
  }

  function ringsOf(geom) {
    if (!geom) return [];
    if (geom.type === "Polygon") return geom.coordinates;
    if (geom.type === "MultiPolygon") return geom.coordinates.flat();
    return [];
  }
  function linesFromFeatures(features) {
    const lon = [], lat = [];
    for (const f of features) for (const ring of ringsOf(f.geometry)) {
      for (const [x, y] of ring) { lon.push(x); lat.push(y); }
      lon.push(null); lat.push(null);
    }
    return { lon, lat };
  }

  // Охват карты в «повёрнутых» долготах: всё, что западнее lon0−180, переносим на +360 (Чукотка: −175 → 185).
  // Центр кадра ищем в проекции: середина спроецированного охвата → обратная проекция. Иначе на конической
  // проекции (дуги параллелей поднимаются к краям) Plotly центрирует по средней широте и срезает верх карты.
  function albers(lon0 = 100, p1 = 52, p2 = 64) {
    const R = Math.PI / 180, sy0 = Math.sin(p1 * R), n = (sy0 + Math.sin(p2 * R)) / 2, c = 1 + sy0 * (2 * n - sy0), r0 = Math.sqrt(c) / n;
    const fwd = (lon, lat) => { const x = (lon - lon0) * R, r = Math.sqrt(Math.max(0, c - 2 * n * Math.sin(lat * R))) / n; return [r * Math.sin(x * n), r0 - r * Math.cos(x * n)]; };
    const inv = (x, y) => { const r0y = r0 - y; let l = Math.atan2(x, Math.abs(r0y)) * Math.sign(r0y); if (r0y * n < 0) l -= Math.PI * Math.sign(x) * Math.sign(r0y); return [lon0 + l / n / R, Math.asin(Math.max(-1, Math.min(1, (c - (x * x + r0y * r0y) * n * n) / (2 * n)))) / R]; };
    return { fwd, inv };
  }
  function mapBounds(lon0 = 100) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, px0 = Infinity, px1 = -Infinity, py0 = Infinity, py1 = -Infinity;
    const P = albers(lon0);
    for (const f of S.mo.features) for (const ring of ringsOf(f.geometry)) for (const [x, y] of ring) {
      const xx = x < lon0 - 180 ? x + 360 : x;
      if (xx < x0) x0 = xx; if (xx > x1) x1 = xx; if (y < y0) y0 = y; if (y > y1) y1 = y;
      const [u, v] = P.fwd(xx, y);
      if (u < px0) px0 = u; if (u > px1) px1 = u; if (v < py0) py0 = v; if (v > py1) py1 = v;
    }
    if (!Number.isFinite(x0)) return { lon: [19, 191], lat: [41, 82], center: [lon0, 66] };
    const px = Math.max(1, (x1 - x0) * 0.02), py = Math.max(0.5, (y1 - y0) * 0.03);
    const center = P.inv((px0 + px1) / 2, (py0 + py1) / 2);
    return { lon: [x0 - px, x1 + px], lat: [Math.max(-89, y0 - py), Math.min(89, y1 + py)], center };
  }

  function initMap() {
    const gd = $("#map");
    const { z, op } = mapArrays();
    const choropleth = {
      type: "choropleth", geojson: S.mo, featureidkey: "properties.territory_id",
      locations: S.ids.map((s) => S.featById.get(s).properties.territory_id),
      z, zmin: -0.5, zmax: S.K + 1.5, colorscale: colorscale(), showscale: false,
      marker: { line: { color: "#ffffff", width: 0.4 }, opacity: op },
      hoverinfo: "none", name: "МО",
    };
    const reg = S.regions ? linesFromFeatures(S.regions.features) : { lon: [], lat: [] };
    const regions = { type: "scattergeo", mode: "lines", lon: reg.lon, lat: reg.lat, line: { color: "#7b8494", width: 0.9 }, hoverinfo: "skip", name: "регионы" };
    const sel = { type: "scattergeo", mode: "lines", lon: [], lat: [], line: { color: "#111827", width: 2.6 }, hoverinfo: "skip", name: "выбранное МО" };
    const b = mapBounds();
    const layout = baseLayout({
      margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false,
      // на сенсорных экранах перетаскивание карты мешает прокрутке страницы — отключаем
      dragmode: window.matchMedia && window.matchMedia("(pointer: coarse)").matches ? false : "pan",
      geo: {
        projection: { type: "conic equal area", rotation: { lon: 100 }, parallels: [52, 64] },
        // fitbounds:"locations" здесь не годится: при геометрии, разрезанной по 180°, Plotly берёт диапазон
        // долгот −180…180 и сбрасывает rotation.lon — карта уезжает в Европу. Границы считаем сами (см. mapBounds).
        fitbounds: false, center: { lon: b.center[0], lat: b.center[1] },
        lonaxis: { range: b.lon }, lataxis: { range: b.lat }, showframe: false, showcoastlines: false, showland: true, landcolor: "#f1f2ef",
        showcountries: true, countrycolor: "#d8dbe0", showocean: false, showlakes: false, bgcolor: "rgba(0,0,0,0)",
        resolution: 110,
      },
    });
    const t0 = performance.now();
    const pin = { type: "scattergeo", mode: "markers", lon: [], lat: [], marker: { symbol: "circle-open", size: 22, color: "#111827", line: { width: 2.5 } }, hoverinfo: "skip", name: "метка МО" };
    return Plotly.newPlot(gd, [choropleth, regions, sel, pin], layout, { ...PLOT_CONFIG, scrollZoom: false }).then(() => {
      console.info(`[pulsar] карта построена за ${Math.round(performance.now() - t0)} мс (${S.ids.length} МО)`);
      S.mapReady = true;
      gd.on("plotly_click", (ev) => { const p = ev.points && ev.points[0]; if (p && p.curveNumber === 0) selectMO(String(p.location)); });
      gd.on("plotly_hover", (ev) => showTip(ev));
      gd.on("plotly_unhover", () => { $("#map-tip").hidden = true; });
    });
  }

  // Быстрый путь: перекрашиваем уже нарисованные Plotly пути (fill/opacity) и синхронно правим модель Plotly
  // (data, _fullData, calcdata), чтобы любая последующая перерисовка (ресайз, зум) дала те же цвета.
  // Plotly.restyle по z у choropleth — это полный пересчёт геометрии: ≈ 0,5 с на 2190 МО × 100 вершин;
  // быстрый путь — ≈ 10–20 мс. Если структура DOM не совпала (другая версия Plotly) — честный restyle.
  function fastRecolor(gd, z, op) {
    const cd = gd.calcdata && gd.calcdata[0];
    const paths = gd.querySelectorAll(".choroplethlocation");
    if (!cd || !paths.length || paths.length !== cd.length) return false;
    const cols = [...Array(S.K).keys()].map(typeColor).concat([NODATA, DIMMED]);
    for (const p of paths) {
      const pt = p.__data__;
      if (!pt || pt.i == null || z[pt.i] == null) return false;
      const i = pt.i, zi = z[i];
      pt.z = zi; pt.mo = op[i];
      p.setAttribute("fill", cols[zi]);
      p.style.opacity = op[i] === 1 ? "" : String(op[i]);
    }
    gd.data[0].z = z; gd.data[0].marker.opacity = op;
    if (gd._fullData && gd._fullData[0]) { gd._fullData[0].z = z; gd._fullData[0].marker.opacity = op; }
    return true;
  }

  function updateMap() {
    if (!S.mapReady) return Promise.resolve();
    const gd = $("#map");
    const { z, op } = mapArrays();
    const t0 = performance.now();
    const done = () => { S.lastRenderMs = performance.now() - t0; window.__pulsarLastRenderMs = S.lastRenderMs; };
    if (fastRecolor(gd, z, op)) { done(); return Promise.resolve(); }
    return Plotly.restyle(gd, { z: [z], "marker.opacity": [op] }, [0]).then(done);
  }

  function updateSelectionOutline() {
    if (!S.mapReady) return;
    const f = S.selected ? S.featById.get(S.selected) : null;
    const { lon, lat } = f ? linesFromFeatures([f]) : { lon: [], lat: [] };
    // кольцо-метка в центре охвата МО: мелкие МО (внутригородские) иначе не найти глазами
    let c = [[], []];
    if (f) {
      const xs = lon.filter((v) => v != null).map((x) => (x < -80 ? x + 360 : x)), ys = lat.filter((v) => v != null);
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
      c = [[cx > 180 ? cx - 360 : cx], [(Math.min(...ys) + Math.max(...ys)) / 2]];
    }
    Plotly.restyle($("#map"), { lon: [lon, c[0]], lat: [lat, c[1]] }, [2, 3]);
  }

  function showTip(ev) {
    const p = ev.points && ev.points[0];
    if (!p || p.curveNumber !== 0) return;
    const id = String(p.location), L = S.labels[id], f = S.featById.get(id);
    const ty = typeAt(id), c = confAt(id);
    const lvl = L && L.level ? L.level[S.t] : null;
    const tip = $("#map-tip");
    tip.innerHTML =
      `<b>${esc((L && L.name) || f.properties.name)}</b><br><span style="opacity:.75">${esc((L && L.region) || f.properties.region || "")}</span><br>` +
      `<span class="sw" style="background:${typeColor(ty)}"></span>${esc(typeName(ty))}` +
      (c != null && ty >= 0 ? `<br>уверенность: ${fmtPct(c)}${c < CONF_THRESHOLD ? " – ненадёжно" : ""}` : "") +
      (lvl != null ? `<br>траты на жителя: ${fmtInt.format(lvl)} ₽` : "") +
      (f && f.properties.in_panel === false ? `<br><i>вне обучающей панели</i>` : "");
    const wrap = $(".map-col").getBoundingClientRect();
    const e = ev.event || {};
    let x = (e.clientX ?? 0) - wrap.left + 14, y = (e.clientY ?? 0) - wrap.top + 14;
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    if (x + w > wrap.width - 8) x = x - w - 28;
    if (y + h > wrap.height - 8) y = y - h - 28;
    tip.style.left = `${Math.max(4, x)}px`;
    tip.style.top = `${Math.max(4, y)}px`;
  }

  // ---- legend
  function renderLegend() {
    const counts = new Array(S.K + 1).fill(0);
    for (const id of S.ids) { const ty = typeAt(id); counts[ty < 0 ? S.K : ty]++; }
    const items = [...Array(S.K).keys()].map((k) => {
      const on = S.focus && S.focus.kind === "type" && S.focus.type === k;
      const dim = S.focus && !on;
      return el("button", {
        type: "button", class: `legend-item${on ? " on" : ""}${dim ? " dim" : ""}`, style: `--c:${typeColor(k)}`, title: `${typeName(k)} – клик, чтобы оставить только этот тип`,
        "aria-pressed": on ? "true" : "false", onclick: () => setFocus(on ? null : { kind: "type", type: k }),
      }, el("span", { class: "sw", style: `background:${typeColor(k)}` }), typeShort(k), el("span", { class: "cnt" }, fmtInt.format(counts[k])));
    });
    if (counts[S.K]) items.push(el("span", { class: "legend-item nodata", title: "МО нет в данных этого месяца или вне панели" }, el("span", { class: "sw" }), "нет данных", el("span", { class: "cnt" }, fmtInt.format(counts[S.K]))));
    $("#legend").replaceChildren(...items);

    const note = $("#highlight-note");
    if (!S.focus) { note.hidden = true; return; }
    let text = "";
    if (S.focus.kind === "type") text = `Показан только тип «${typeName(S.focus.type)}»: ${fmtInt.format(counts[S.focus.type])} МО в выбранном месяце.`;
    if (S.focus.kind === "flow") {
      const n = S.ids.filter((id) => matchesFocus(id, 0)).length;
      text = `Переход «${typeShort(S.focus.from)}» → «${typeShort(S.focus.to)}» (${S.trans.from_month || "2023-12"} → ${S.trans.to_month || "2024-12"}, подтверждённые типы): ${fmtInt.format(n)} МО. Цвет – тип в выбранном месяце.`;
    }
    if (S.focus.kind === "ids") text = `${S.focus.label}: ${fmtInt.format(S.focus.ids.size)} МО.`;
    note.replaceChildren(el("span", {}, text), el("button", { type: "button", class: "btn-link", onclick: () => setFocus(null) }, "Сбросить"));
    note.hidden = false;
  }

  function setFocus(f) {
    S.focus = f;
    renderLegend();
    updateMap();
  }

  // ---- controls
  function setMonth(i) {
    S.t = Math.max(0, Math.min(S.T - 1, i));
    $("#month-slider").value = S.t;
    $("#month-label").textContent = monthLabel(S.t);
    renderLegend();
    updateMap();
    if (S.selected) updatePassportMonth();
    markSizesMonth();
  }

  function togglePlay(force) {
    const btn = $("#play");
    const stop = force === false || (force == null && S.timer);
    if (stop) {
      clearInterval(S.timer); S.timer = null;
      btn.textContent = "▶"; btn.setAttribute("aria-label", "Воспроизвести");
      return;
    }
    if (S.t >= S.T - 1) setMonth(0);
    btn.textContent = "❚❚"; btn.setAttribute("aria-label", "Пауза");
    S.timer = setInterval(() => {
      if (S.t >= S.T - 1) { togglePlay(false); return; }
      setMonth(S.t + 1);
    }, 750);
  }

  function initControls() {
    const slider = $("#month-slider");
    slider.max = S.T - 1;
    slider.value = S.t;
    $("#month-label").textContent = monthLabel(S.t);
    slider.addEventListener("input", () => { togglePlay(false); setMonth(+slider.value); });
    $("#play").addEventListener("click", () => togglePlay());
    document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", () => {
      S.mode = r.value; renderLegend(); updateMap(); if (S.selected) updatePassportMonth();
    }));
    $("#uncertainty").addEventListener("change", (e) => { S.unc = e.target.checked; updateMap(); updateMapNote(); });

    // поиск
    const dl = $("#mo-list"), byLabel = new Map();
    const frag = document.createDocumentFragment();
    const rows = S.ids.map((id) => {
      const L = S.labels[id], p = S.featById.get(id).properties;
      return [`${(L && L.name) || p.name} – ${(L && L.region) || p.region || ""}`, id];
    }).sort((a, b) => a[0].localeCompare(b[0], "ru"));
    for (const [label, id] of rows) { byLabel.set(label.toLowerCase(), id); frag.append(el("option", { value: label })); }
    dl.replaceChildren(frag);
    const input = $("#mo-search");
    const tryPick = (loose) => {
      const v = input.value.trim().toLowerCase();
      if (!v) return;
      let id = byLabel.get(v);
      if (!id && loose) {
        const low = rows.map(([l, i]) => [l.toLowerCase(), i]);
        const hit = low.find(([l]) => l.startsWith(v)) || low.find(([l]) => l.includes(" " + v) || l.includes("(" + v)) || low.find(([l]) => l.includes(v));
        if (hit) id = hit[1];
      }
      if (id) { selectMO(id); input.blur(); }
    };
    input.addEventListener("change", () => tryPick(false));
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") tryPick(true); });
    updateMapNote();
  }

  function updateMapNote() {
    $("#map-note").textContent = S.unc
      ? `Слой неопределённости: чем бледнее МО, тем ниже бутстрэп-вероятность его типа; почти прозрачные – ниже ${fmtPct(CONF_THRESHOLD)}. Серые – нет данных в этом месяце.`
      : "";
  }

  // ---- passport МО
  function selectMO(id, opts = {}) {
    if (!S.featById.has(id) && !S.labels[id]) return;
    S.selected = id;
    updateSelectionOutline();
    renderPassport();
    if (opts.scroll) $("#map-section").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderPassport() {
    const id = S.selected, L = S.labels[id] || {}, f = S.featById.get(id), p = f ? f.properties : {};
    const box = $("#passport");
    box.replaceChildren(
      el("button", { type: "button", class: "btn pp-close", "aria-label": "Закрыть паспорт", onclick: closePassport }, "✕"),
      el("p", { class: "muted", style: "margin:0 0 4px;font-size:12px" }, "Паспорт МО"),
      el("h3", {}, L.name || p.name || `МО ${id}`),
      el("div", { class: "region" }, [L.region || p.region, p.type].filter(Boolean).join(", ")),
      el("div", { class: "pp-badges", id: "pp-badges" }),
      el("div", { class: "pp-label" }, el("span", {}, "Подтверждённый тип, 24 мес."), el("span", {}, "клик – выбрать месяц")),
      el("div", { class: "ribbon", id: "pp-ribbon" }),
      el("div", { class: "pp-label" }, el("span", {}, "Сырой тип")),
      el("div", { class: "ribbon raw", id: "pp-ribbon-raw" }),
      el("div", { class: "ribbon-axis" }, el("span", {}, monthShort(S.months[0])), el("span", {}, monthShort(S.months[Math.floor(S.T / 2)])), el("span", {}, monthShort(S.months[S.T - 1]))),
      el("div", { class: "pp-label" }, el("span", {}, "Траты на жителя, ₽")),
      el("div", { class: "pp-chart", id: "pp-level" }),
      el("div", { class: "pp-label" }, el("span", {}, "Структура трат, доля")),
      el("div", { class: "pp-chart", id: "pp-shares" }),
      el("div", { class: "pp-legend", id: "pp-shares-legend" }),
      el("div", { class: "pp-label" }, el("span", {}, "Уверенность типа (бутстрэп)"), el("span", {}, `порог ${fmtPct(CONF_THRESHOLD)}`)),
      el("div", { class: "pp-chart small", id: "pp-conf" }),
    );
    const mkRibbon = (seq, target) => {
      const cells = S.months.map((m, i) => {
        const ty = seq ? seq[i] : -1, c = L.confidence ? L.confidence[i] : null;
        return el("div", {
          class: ty < 0 ? "nd" : "", style: `background:${typeColor(ty)}`,
          title: `${monthLabel(i)}: ${typeName(ty)}${c != null && ty >= 0 ? `, уверенность ${fmtPct(c)}` : ""}`,
          onclick: () => { togglePlay(false); setMonth(i); },
        });
      });
      $(target).replaceChildren(...cells);
    };
    mkRibbon(L.confirmed || L.type, "#pp-ribbon");
    mkRibbon(L.type, "#pp-ribbon-raw");
    drawPassportCharts(L);
    updatePassportMonth();
  }

  function closePassport() {
    S.selected = null;
    updateSelectionOutline();
    $("#passport").replaceChildren(el("div", { class: "passport-empty" }, el("p", { class: "muted" }, "Паспорт МО"), el("p", {}, "Кликните по муниципальному образованию на карте или найдите его через поиск.")));
  }

  function monthLine(i, yref = "paper") {
    return { type: "line", xref: "x", yref, x0: S.months[i], x1: S.months[i], y0: 0, y1: 1, line: { color: "#1b1f24", width: 1, dash: "dash" } };
  }

  function drawPassportCharts(L) {
    const x = S.months;
    const conf = L.confirmed || L.type || [];
    if (L.level) {
      Plotly.react("pp-level", [{
        type: "scatter", mode: "lines", x, y: L.level, connectgaps: false,
        line: { color: "#4a5360", width: 2 }, marker: { size: 7, color: x.map((_, i) => typeColor(conf[i])), line: { color: "#fff", width: 1 } },
        hovertemplate: "%{x}<br>%{y:,.0f} ₽<extra></extra>",
      }], baseLayout({ margin: { l: 48, r: 6, t: 6, b: 24 }, xaxis: axis({ type: "category", tickvals: [x[0], x[12], x[S.T - 1]].filter(Boolean), ticktext: [x[0], x[12], x[S.T - 1]].filter(Boolean).map(monthShort) }), yaxis: axis({ tickformat: ",.0f" }), shapes: [monthLine(S.t)] }), STATIC_CONFIG);
    } else $("#pp-level").replaceChildren(el("p", { class: "muted" }, "нет ряда трат"));

    if (L.shares) {
      const cats = Object.keys(L.shares);
      const other = x.map((_, i) => { const v = cats.map((c) => L.shares[c][i]); return v.some((a) => a == null) ? null : Math.max(0, 1 - v.reduce((a, b) => a + b, 0)); });
      const series = cats.map((c) => [c, L.shares[c]]).concat([["Прочее", other]]);
      $("#pp-shares-legend").replaceChildren(...series.map(([name], j) => el("span", {}, el("i", { style: `background:${CAT_RAMP[j % CAT_RAMP.length]}` }), name)));
      Plotly.react("pp-shares", series.map(([name, y], j) => ({
        type: "scatter", mode: "lines", stackgroup: "s", name, x, y, line: { width: 0.5, color: "#fff" }, fillcolor: CAT_RAMP[j % CAT_RAMP.length],
        hovertemplate: `${name}: %{y:.1%}<extra></extra>`,
      })), baseLayout({ margin: { l: 36, r: 6, t: 6, b: 24 }, showlegend: false, hovermode: "x unified", xaxis: axis({ type: "category", tickvals: [x[0], x[12], x[S.T - 1]].filter(Boolean), ticktext: [x[0], x[12], x[S.T - 1]].filter(Boolean).map(monthShort) }), yaxis: axis({ tickformat: ".0%", range: [0, 1] }), shapes: [monthLine(S.t)] }), STATIC_CONFIG);
    } else $("#pp-shares").replaceChildren(el("p", { class: "muted" }, "нет структуры трат"));

    if (L.confidence) {
      Plotly.react("pp-conf", [{
        type: "bar", x, y: L.confidence, marker: { color: L.confidence.map((c) => (c < CONF_THRESHOLD ? "#c9ccd1" : "#6b7280")) },
        hovertemplate: "%{x}: %{y:.0%}<extra></extra>",
      }], baseLayout({ margin: { l: 36, r: 6, t: 4, b: 20 }, bargap: 0.25, xaxis: axis({ type: "category", showticklabels: false }), yaxis: axis({ tickformat: ".0%", range: [0, 1], dtick: 0.5 }),
        shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: CONF_THRESHOLD, y1: CONF_THRESHOLD, line: { color: "#D55E00", width: 1, dash: "dash" } }, monthLine(S.t)] }), STATIC_CONFIG);
    }
  }

  function updatePassportMonth() {
    const id = S.selected; if (!id) return;
    const L = S.labels[id] || {}, f = S.featById.get(id), ty = typeAt(id), c = confAt(id);
    const badges = [
      el("span", { class: "badge" }, el("span", { class: "sw", style: `background:${typeColor(ty)}` }), `${monthLabel(S.t)}: ${typeName(ty)}`),
    ];
    if (c != null && ty >= 0) badges.push(el("span", { class: `badge${c < CONF_THRESHOLD ? " warn" : ""}` }, `уверенность ${fmtPct(c)}`));
    if (S.mode === "raw") badges.push(el("span", { class: "badge" }, "сырые типы"));
    if (f && f.properties.in_panel === false) badges.push(el("span", { class: "badge warn" }, "вне обучающей панели"));
    const lvl = L.level ? L.level[S.t] : null;
    if (lvl != null) badges.push(el("span", { class: "badge" }, `${fmtInt.format(lvl)} ₽ на жителя`));
    const b = $("#pp-badges"); if (b) b.replaceChildren(...badges);
    ["#pp-ribbon", "#pp-ribbon-raw"].forEach((s) => { const r = $(s); if (r) [...r.children].forEach((d, i) => d.classList.toggle("cur", i === S.t && s === "#pp-ribbon")); });
    for (const gid of ["pp-level", "pp-shares", "pp-conf"]) {
      const gd = document.getElementById(gid);
      if (gd && gd.layout && gd.layout.shapes) {
        const shapes = gd.layout.shapes.slice(0, -1).concat([monthLine(S.t)]);
        Plotly.relayout(gd, { shapes });
      }
    }
  }

  // ---------------------------------------------------------------- 3. types
  function renderTypes() {
    const feats = S.types.length ? Object.keys(S.types[0].profile || {}) : [];
    const maxAbs = Math.max(1, ...S.types.flatMap((t) => Object.values(t.profile || {}).map((v) => Math.abs(v))));
    const range = [-maxAbs * 1.1, maxAbs * 1.1];
    const wrap = $("#type-cards");
    wrap.replaceChildren();
    for (const t of S.types) {
      const color = typeColor(t.id);
      const chip = (m, cls) => el("button", { type: "button", class: `chip ${cls}`, title: "Открыть паспорт МО на карте", onclick: () => { selectMO(String(m.territory_id), { scroll: true }); } }, m.name, " ", el("small", {}, m.region || ""));
      const kv = (obj, fmtFn) => el("dl", { class: "kv" }, ...Object.entries(obj || {}).map(([k, v]) => el("div", {}, el("dt", {}, k), el("dd", {}, fmtFn(k, v)))));
      const fmtMed = (k, v) => (v == null ? "–" : /₽|медиане/.test(k) ? fmtInt.format(v) : /%/.test(k) ? fmt1.format(v) : typeof v === "number" ? fmtInt.format(v) : String(v));
      const fmtExt = (k, v) => (v == null ? "–" : typeof v === "number" ? (Math.abs(v) < 10 && !Number.isInteger(v) ? new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 }).format(v) : fmtInt.format(v)) : String(v));
      const profId = `tc-prof-${t.id}`;
      const card = el("article", { class: "type-card", id: `type-${t.id}` },
        el("div", { class: "tc-img", style: `--c:${color}` },
          el("img", { src: `img/type${t.id}.webp`, loading: "lazy", alt: `Иллюстрация типа «${t.name}»`, onerror: (e) => { e.target.style.visibility = "hidden"; } })),
        el("div", { class: "body" },
          el("div", { class: "tc-head" },
            el("div", {}, el("h3", {}, t.name)),
            el("div", { class: "tc-n", title: `доля от ${fmtInt.format(S.meta.n_panel || 2016)} МО с полной историей` }, el("b", {}, fmtInt.format(t.n ?? 0)), el("div", {}, `МО, ${fmtPct(t.share, 1)} панели`))),
          el("p", { class: "desc" }, t.description || ""),
          el("p", { class: "sub-h" }, "Профиль, σ от средней по стране"),
          el("div", { class: "tc-profile", id: profId }),
          el("p", { class: "sub-h" }, "Медианы"),
          kv(t.medians, fmtMed),
          t.external && Object.keys(t.external).length ? el("p", { class: "sub-h" }, "Внешние показатели (не участвовали в кластеризации)") : null,
          t.external && Object.keys(t.external).length ? kv(t.external, fmtExt) : null,
          el("p", { class: "sub-h" }, "Типичные МО"),
          el("div", { class: "chips" }, ...(t.typical || []).map((m) => chip(m, "typical"))),
          el("p", { class: "sub-h" }, "Пограничные МО"),
          el("div", { class: "chips" }, ...(t.border || []).map((m) => chip(m, "border"))),
          el("button", { type: "button", class: "btn", style: "align-self:flex-start;margin-top:4px", onclick: () => { setFocus({ kind: "type", type: t.id }); $("#map-section").scrollIntoView({ behavior: "smooth" }); } }, "Показать тип на карте"),
        ));
      wrap.append(card);
      const vals = feats.map((f) => (t.profile || {})[f] ?? 0);
      Plotly.newPlot(profId, [{
        type: "bar", orientation: "h", y: feats, x: vals,
        marker: { color: vals.map((v) => (Math.abs(v) >= 0.5 ? color : hexA(color, 0.4))) },
        text: vals.map((v) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt1.format(Math.abs(v))), textposition: "outside", cliponaxis: false,
        textfont: { size: 11, color: "#4a5360" },
        hovertemplate: "%{y}: %{x:+.2f} σ<extra></extra>",
      }], baseLayout({ margin: { l: 150, r: 30, t: 4, b: 22 }, bargap: 0.3,
        xaxis: axis({ range, zeroline: true, zerolinecolor: "#9aa1ab", zerolinewidth: 1, ticksuffix: " σ", dtick: 1 }),
        yaxis: axis({ autorange: "reversed", tickfont: { size: 12, color: "#1b1f24" }, gridcolor: "rgba(0,0,0,0)" }) }), STATIC_CONFIG);
    }
  }

  // ---------------------------------------------------------------- 4. dynamics
  function renderDynamics() {
    const tr = S.trans, K = S.K;
    const [i0, i1] = flowIdx();
    const fromM = tr.from_month || S.months[i0], toM = tr.to_month || S.months[i1];
    const counts = tr.counts || [];
    const total = counts.flat().reduce((a, b) => a + b, 0);
    const moved = counts.reduce((a, row, i) => a + row.reduce((s, v, j) => s + (i === j ? 0 : v), 0), 0);
    $("#dyn-lede").textContent = `Сравниваем подтверждённые типы ${monthLabel(i0)} и ${monthLabel(i1)}: смена засчитывается, только если новое поведение держится не менее трёх месяцев подряд и устойчиво по бутстрэпу. Сырые месячные метки меняются заметно чаще, это видно по индексу мобильности Шоррокса.`;
    $("#sankey-sub").textContent = `${fromM} → ${toM}, подтверждённые типы, число МО`;
    const kpi = (v, l) => el("div", { class: "kpi" }, el("div", { class: "v" }, v), el("div", { class: "l" }, l));
    $("#dyn-kpis").replaceChildren(
      kpi(total ? fmtPct(moved / total, 1) : "–", `МО, у которых подтверждённый тип в декабре 2024 другой, чем в декабре 2023 (${fmtInt.format(moved)} из ${fmtInt.format(total)})`),
      kpi(tr.shorrocks ? fmt2(tr.shorrocks.confirmed) : "–", "индекс Шоррокса, подтверждённые типы"),
      kpi(tr.shorrocks ? fmt2(tr.shorrocks.raw) : "–", "индекс Шоррокса, сырые месячные метки"),
      kpi(tr.stable_share != null ? fmtPct(tr.stable_share) : "–", "МО без смен подтверждённого типа за 24 месяца"),
    );

    // Sankey
    const flows = (tr.flows && tr.flows.length ? tr.flows : counts.flatMap((row, i) => row.map((n, j) => ({ from: i, to: j, n })))).filter((f) => f.n > 0);
    const outTot = new Array(K).fill(0); flows.forEach((f) => { outTot[f.from] += f.n; });
    const inTot = new Array(K).fill(0); flows.forEach((f) => { inTot[f.to] += f.n; });
    const order = [...Array(K).keys()];
    const ys = (tot) => { const s = tot.reduce((a, b) => a + b, 0) || 1; let acc = 0; return tot.map((v) => { const y = (acc + v / 2) / s; acc += v; return 0.02 + 0.96 * y; }); };
    const yL = ys(outTot), yR = ys(inTot);
    Plotly.newPlot("sankey", [{
      type: "sankey", arrangement: "fixed", valueformat: ",.0f",
      node: {
        label: order.map((k) => `${typeShort(k)}`).concat(order.map((k) => `${typeShort(k)}`)),
        color: order.map(typeColor).concat(order.map(typeColor)),
        x: order.map(() => 0.001).concat(order.map(() => 0.999)), y: yL.concat(yR),
        pad: 14, thickness: 16, line: { color: "#fff", width: 1 },
        customdata: order.map((k) => [typeName(k), fromM]).concat(order.map((k) => [typeName(k), toM])),
        hovertemplate: "%{customdata[0]}<br>%{customdata[1]}: %{value:,.0f} МО<extra></extra>",
      },
      link: {
        source: flows.map((f) => f.from), target: flows.map((f) => K + f.to), value: flows.map((f) => f.n),
        color: flows.map((f) => hexA(typeColor(f.from), f.from === f.to ? 0.22 : 0.5)),
        customdata: flows.map((f) => [typeShort(f.from), typeShort(f.to), outTot[f.from] ? f.n / outTot[f.from] : 0]),
        hovertemplate: "%{customdata[0]} → %{customdata[1]}<br>%{value:,.0f} МО (%{customdata[2]:.0%} типа)<extra>клик – показать на карте</extra>",
      },
    }], baseLayout({ margin: { l: 10, r: 10, t: 24, b: 10 }, font: { ...FONT, size: 12 },
      annotations: [
        { x: 0, y: 1.05, xref: "paper", yref: "paper", text: `<b>${fromM}</b>`, showarrow: false, xanchor: "left", font: { size: 12, color: "#4a5360" } },
        { x: 1, y: 1.05, xref: "paper", yref: "paper", text: `<b>${toM}</b>`, showarrow: false, xanchor: "right", font: { size: 12, color: "#4a5360" } },
      ] }), STATIC_CONFIG).then((gd) => {
      gd.on("plotly_click", (ev) => {
        const p = ev.points && ev.points[0]; if (!p) return;
        if (p.sourceLinks !== undefined || p.targetLinks !== undefined) {   // узел
          const k = p.pointNumber % K;
          setFocus({ kind: "type", type: k });
        } else {                                                            // поток
          const f = flows[p.pointNumber]; if (!f) return;
          S.mode = "confirmed"; $('input[name="mode"][value="confirmed"]').checked = true;
          S.focus = { kind: "flow", from: f.from, to: f.to };
          setMonth(i1);
        }
        $("#map-section").scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });

    // Матрица переходов
    const mat = (tr.matrix_2023_2024 || tr.matrix || []).map((r) => r.map((v) => 100 * v));
    Plotly.newPlot("matrix", [{
      type: "heatmap", z: mat, x: order.map(typeShort), y: order.map(typeShort),
      colorscale: [[0, "#F5F6F7"], [0.15, "#CDEFD9"], [0.5, "#4FC283"], [1, "#148F2B"]], zmin: 0, zmax: 100, xgap: 2, ygap: 2,
      customdata: counts, text: mat.map((r) => r.map((v) => (v >= 0.5 ? fmtInt.format(v) : ""))), texttemplate: "%{text}", textfont: { size: 12 },
      hovertemplate: `из «%{y}» (${fromM})<br>в «%{x}» (${toM})<br>%{z:.1f} % строки, %{customdata} МО<extra></extra>`,
      colorbar: { thickness: 10, ticksuffix: " %", len: 0.9, outlinewidth: 0, tickfont: { size: 10 } },
    }], baseLayout({ margin: { l: 110, r: 10, t: 10, b: 80 },
      xaxis: axis({ title: { text: `тип в ${toM}`, font: { size: 12 } }, tickangle: -30, side: "bottom", gridcolor: "rgba(0,0,0,0)" }),
      yaxis: axis({ title: { text: `тип в ${fromM}`, font: { size: 12 }, standoff: 12 }, autorange: "reversed", gridcolor: "rgba(0,0,0,0)", automargin: true }) }), STATIC_CONFIG).then((gd) => {
      gd.on("plotly_click", (ev) => {
        const p = ev.points && ev.points[0]; if (!p) return;
        const from = order.find((k) => typeShort(k) === p.y), to = order.find((k) => typeShort(k) === p.x);
        if (from == null || to == null) return;
        S.mode = "confirmed"; $('input[name="mode"][value="confirmed"]').checked = true;
        S.focus = { kind: "flow", from, to };
        setMonth(i1);
        $("#map-section").scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });

    // Размеры типов по месяцам
    const sizes = tr.sizes || {};
    const xs = S.months.filter((m) => sizes[m]);
    Plotly.newPlot("sizes", order.map((k) => ({
      type: "scatter", mode: "lines", stackgroup: "one", name: typeShort(k), x: xs, y: xs.map((m) => sizes[m][k] ?? 0),
      line: { width: 0.6, color: "#fff" }, fillcolor: typeColor(k),
      hovertemplate: `${typeShort(k)}: %{y:,.0f}<extra></extra>`,
    })), baseLayout({ margin: { l: 48, r: 10, t: 10, b: 40 }, hovermode: "x unified",
      legend: { orientation: "h", y: -0.18, font: { size: 11 }, traceorder: "normal" },
      xaxis: axis({ type: "category", tickvals: xs.filter((_, i) => i % 3 === 0), ticktext: xs.filter((_, i) => i % 3 === 0).map(monthShort) }),
      yaxis: axis({ tickformat: ",.0f" }), shapes: [monthLine(S.t)] }), STATIC_CONFIG).then((gd) => {
      gd.on("plotly_click", (ev) => { const p = ev.points && ev.points[0]; if (p) { const i = S.months.indexOf(p.x); if (i >= 0) setMonth(i); } });
    });

    // События
    const pill = (k) => el("button", { type: "button", class: "tpill", title: `${typeName(k)} – показать на карте`, onclick: () => { setFocus({ kind: "type", type: k }); $("#map-section").scrollIntoView({ behavior: "smooth" }); } }, el("span", { class: "sw", style: `background:${typeColor(k)}` }), typeShort(k));
    const evs = tr.events || [];
    const cnt = (...ks) => evs.filter((e) => ks.includes(e.event)).length;
    const other = cnt("merge", "split", "birth", "form", "death", "dissolve");
    $("#events").replaceChildren(el("li", { class: "ev-summary" }, el("p", {}, evs.length
      ? (other === 0
        ? `За ${S.months.length} месяца – ${fmtInt.format(cnt("continue"))} продолжений типов; слияний, разделений, появлений и исчезновений нет. Типы устойчивы: ни один не распался и не возник заново.`
        : `За ${S.months.length} месяца: ${fmtInt.format(cnt("continue"))} продолжений типов, ${cnt("merge")} слияний, ${cnt("split")} разделений, ${cnt("birth", "form")} появлений, ${cnt("death", "dissolve")} исчезновений.`)
      : "Событий не обнаружено")));
  }

  function fmt2(x) { return x == null ? "–" : new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2, minimumFractionDigits: 2 }).format(x); }

  function markSizesMonth() {
    const gd = document.getElementById("sizes");
    if (gd && gd.layout) Plotly.relayout(gd, { shapes: [monthLine(S.t)] });
  }

  // ---------------------------------------------------------------- 5. methods
  function renderMethods() {
    const M = S.methods;
    if (!M) { $("#methods").hidden = true; return; }
    const dirs = { ...DEFAULT_DIRECTIONS, ...(M.directions || {}) };
    const ours = S.meta.method;
    $("#methods-lede").textContent = `Шесть методов кластеризации (KEFRiN в двух вариантах, поэтому в таблице семь строк) сравниваются на одних и тех же признаках и одном и том же графе при равном числе типов K = ${M.k_fixed ?? S.K}. Признаковые индексы (SW, CH, S_Dbw) оценивают компактность в пространстве трат, сетевые (AVI, AVU, ANUI, MQ) – согласованность типов с графом связей МО. Итоговый рейтинг – по Борда, пороговому агрегированию и доле побед в бутстрэпе.`;
    const rows = M.rows || [];
    const preferred = ["SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "ANUI", "borda", "threshold_rank", "win_rate"];
    const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))].filter((k) => k !== "method" && rows.some((r) => typeof r[k] === "number"));
    const cols = preferred.filter((k) => keys.includes(k)).concat(keys.filter((k) => !preferred.includes(k)));
    const best = {};
    for (const c of cols) {
      const vals = rows.map((r) => r[c]).filter((v) => typeof v === "number" && !Number.isNaN(v));
      if (!vals.length || !dirs[c]) continue;
      best[c] = dirs[c] === "min" ? Math.min(...vals) : Math.max(...vals);
    }
    const fmtCell = (c, v) => {
      if (v == null || Number.isNaN(v)) return "–";
      if (c === "win_rate" || c === "mean_win_rate") return fmtPct(v);
      if (c === "n1" || c === "n2" || c === "n3") return fmtInt.format(Math.round(v));
      if (c === "CH" || c === "borda" || c === "threshold_rank" || Math.abs(v) >= 100) return fmtInt.format(v);
      return new Intl.NumberFormat("ru-RU", { minimumFractionDigits: 3, maximumFractionDigits: 3 }).format(v);
    };
    const arrow = (c) => (dirs[c] === "min" ? "↓" : dirs[c] === "max" ? "↑" : "");
    const thead = el("thead", {}, el("tr", {}, el("th", {}, "Метод"), ...cols.map((c) => el("th", { title: dirs[c] === "min" ? "меньше – лучше" : "больше – лучше" }, METRIC_NAMES[c] || c, el("span", { class: "dir" }, arrow(c))))));
    const tbody = el("tbody", {}, ...rows.map((r) => el("tr", { class: r.method === ours ? "ours" : "" },
      el("td", {}, METHOD_NAMES[r.method] || r.method),
      ...cols.map((c) => el("td", { class: best[c] != null && r[c] === best[c] ? "best" : "" }, fmtCell(c, r[c]))))));
    $("#icvi").replaceChildren(thead, tbody);

    // Парето
    const paretoSet = new Set((M.pareto || []).map((p) => p.method));
    const front = rows.filter((r) => paretoSet.has(r.method)).sort((a, b) => a.SW - b.SW);
    const name = (m) => METHOD_NAMES[m] || m;
    const sws = rows.map((r) => r.SW), mqs = rows.map((r) => r.MQ);
    const pad = (v, f) => { const lo = Math.min(...v), hi = Math.max(...v), d = (hi - lo || 0.1) * f; return [lo - d, hi + d]; };
    // Подписи: для близких точек (k-means и два KEFRiN) — выноски со стрелками в разные стороны
    const LABEL_OFFSETS = { kefrin_balanced: [-35, -85], kefrin: [-96, 8], kmeans: [-40, 40] };
    const annots = rows.map((r) => {
      const isOurs = r.method === ours, off = LABEL_OFFSETS[r.method];
      const txt = isOurs ? `<b>${name(r.method)}</b>` : name(r.method);
      const font = { size: 12, color: isOurs ? ACCENT : "#1b1f24" };
      return off
        ? { x: r.SW, y: r.MQ, text: txt, font, showarrow: true, arrowhead: 0, arrowwidth: 1, arrowcolor: "#9aa1ab", ax: off[0], ay: off[1], standoff: 8, xanchor: off[1] < -50 ? "center" : "right" }
        : { x: r.SW, y: r.MQ, text: txt, font, showarrow: false, xanchor: "left", xshift: 11 };
    });
    Plotly.newPlot("pareto", [
      { type: "scatter", mode: "lines", x: front.map((r) => r.SW), y: front.map((r) => r.MQ), line: { color: "#9aa1ab", width: 1.5, dash: "dash" }, hoverinfo: "skip", name: "фронт Парето" },
      {
        type: "scatter", mode: "markers", x: rows.map((r) => r.SW), y: rows.map((r) => r.MQ), text: rows.map((r) => name(r.method)),
        marker: {
          size: rows.map((r) => (r.method === ours ? 16 : 13)),
          color: rows.map((r) => (r.method === ours ? ACCENT : paretoSet.has(r.method) ? "#4a5360" : "#ffffff")),
          line: { color: rows.map((r) => (r.method === ours ? ACCENT : "#4a5360")), width: 2 },
        },
        customdata: rows.map((r) => (paretoSet.has(r.method) ? "на фронте Парето" : "доминируется")),
        hovertemplate: "<b>%{text}</b><br>SW = %{x:.3f}<br>MQ = %{y:.3f}<br>%{customdata}<extra></extra>", name: "методы",
      },
    ], baseLayout({ showlegend: false, margin: { l: 56, r: 16, t: 16, b: 46 }, annotations: annots,
      xaxis: axis({ title: { text: "силуэт SW (признаки) →", font: { size: 12 } }, tickformat: ".2f", range: pad(sws, 0.12) }),
      yaxis: axis({ title: { text: "модулярность MQ (граф) →", font: { size: 12 } }, tickformat: ".2f", range: pad(mqs, 0.12) }) }), STATIC_CONFIG);

    // Абляция слоёв
    const METHOD_RU = { kefrin_balanced: "KEFRiN сбаланс.", kefrin: "KEFRiN", leiden: "Leiden", spectral: "Spectral", kmeans: "k-means", ward: "Ward" };
    const LORDER = { behaviour: 0, comovement: 1, gravity: 2, fused: 3 };
    const layers = (M.layers || []).slice().sort((a, b) => (LORDER[a.layer] ?? 9) - (LORDER[b.layer] ?? 9) || String(a.method).localeCompare(String(b.method)));
    const lcols = ["SW", "MQ", "ari_vs_fused"].filter((c) => layers.some((l) => typeof l[c] === "number"));
    const lbest = {};
    for (const c of ["SW", "MQ"]) { const v = layers.map((l) => l[c]).filter((x) => typeof x === "number"); if (v.length) lbest[c] = Math.max(...v); }
    const lname = { SW: "SW ↑", MQ: "MQ ↑", ari_vs_fused: "ARI с итогом" };
    $("#layers").replaceChildren(
      el("thead", {}, el("tr", {}, el("th", {}, "Слой графа"), el("th", {}, "Метод"), ...lcols.map((c) => el("th", {}, lname[c] || c)))),
      el("tbody", {}, ...layers.map((l) => el("tr", { class: l.layer === "fused" ? "ours" : "" },
        el("td", {}, LAYER_NAMES[l.layer] || l.layer),
        el("td", {}, METHOD_RU[l.method] || l.method || ""),
        ...lcols.map((c) => el("td", { class: lbest[c] != null && l[c] === lbest[c] ? "best" : "" }, l[c] == null ? "–" : new Intl.NumberFormat("ru-RU", { minimumFractionDigits: c === "ari_vs_fused" ? 2 : 3, maximumFractionDigits: c === "ari_vs_fused" ? 2 : 3 }).format(l[c])))))),
    );
    $("#methods-notes").textContent = M.notes || "";
  }

  // ---------------------------------------------------------------- 6. cases
  function renderCases() {
    const wrap = $("#case-cards");
    if (!S.cases.length) { $("#cases").hidden = true; return; }
    wrap.replaceChildren();
    S.cases.forEach((c, n) => {
      const hasId = c.territory_id != null, id = String(c.territory_id), L = hasId ? (S.labels[id] || {}) : {};
      const chartId = `case-chart-${n}`;
      const card = el("article", { class: "card case-card" },
        el("h3", {}, c.title),
        el("div", { class: "case-meta" }, [L.name, L.region].filter(Boolean).join(", ") || (hasId ? `МО ${id}` : "Все МО")),
        el("div", { class: "case-chart", id: chartId }),
        el("div", { class: "case-text" }, ...String(c.text || "").split(/\n\s*\n/).map((p) => el("p", {}, p))),
        el("button", { type: "button", class: "btn", onclick: () => {
          const i = (c.months_highlight || []).length ? S.months.indexOf(c.months_highlight[0]) : -1;
          if (i >= 0) setMonth(i);
          if (hasId) selectMO(id, { scroll: true }); else $("#map-section").scrollIntoView({ behavior: "smooth" });
        } }, "Показать на карте"));
      wrap.append(card);
      drawCaseChart(chartId, c, L);
    });
  }

  function drawCaseChart(target, c, L) {
    const x = S.months, hl = (c.months_highlight || []).filter((m) => x.includes(m));
    const shapes = hl.length ? [{ type: "rect", xref: "x", yref: "paper", x0: hl[0], x1: hl[hl.length - 1], y0: 0, y1: 1, fillcolor: "rgba(213,94,0,.12)", line: { width: 0 }, layer: "below" }] : [];
    // для одного месяца — расширяем полосу на ±0.5 категории
    if (hl.length && hl[0] === hl[hl.length - 1]) { const i = x.indexOf(hl[0]); Object.assign(shapes[0], { x0: i - 0.5, x1: i + 0.5 }); }
    if (hl.length > 1) { Object.assign(shapes[0], { x0: x.indexOf(hl[0]) - 0.5, x1: x.indexOf(hl[hl.length - 1]) + 0.5 }); }
    const xa = axis({ type: "category", tickvals: x.filter((_, i) => i % 3 === 0), ticktext: x.filter((_, i) => i % 3 === 0).map(monthShort) });
    const conf = L.confirmed || L.type || [];
    if (c.chart === "shares" && L.shares) {
      const cats = Object.keys(L.shares);
      Plotly.newPlot(target, cats.map((cat, j) => ({
        type: "scatter", mode: "lines", name: cat, x, y: L.shares[cat], line: { width: 2, color: CAT_RAMP[j % CAT_RAMP.length] === "#dbe9ed" ? "#9fb7c0" : CAT_RAMP[j % CAT_RAMP.length] },
        hovertemplate: `${cat}: %{y:.1%}<extra></extra>`,
      })), baseLayout({ margin: { l: 44, r: 8, t: 6, b: 64 }, hovermode: "x unified", legend: { orientation: "h", x: 0, y: -0.22, yanchor: "top", font: { size: 10.5 } }, xaxis: xa, yaxis: axis({ type: "log", tickvals: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ticktext: ["1 %", "2 %", "5 %", "10 %", "20 %", "50 %"], title: { text: "доля, лог. шкала", font: { size: 10.5 } } }), shapes }), STATIC_CONFIG);
    } else if (c.chart === "changes" && S.trans && S.trans.changes_by_month) {
      const cbm = S.trans.changes_by_month, xs = x.filter((m) => cbm[m]);
      const hlSet = new Set(hl);
      const bar = (key, nm, col, colHl) => ({
        type: "bar", name: nm, x: xs, y: xs.map((m) => cbm[m][key] ?? 0),
        marker: { color: xs.map((m) => (hlSet.has(m) ? colHl : col)) },
        hovertemplate: `${nm}: %{y:,.0f}<extra></extra>`,
      });
      Plotly.newPlot(target, [bar("raw", "сырые смены типа", "#C4C9CF", "#8A929C"), bar("confirmed", "подтверждённые", ACCENT, "#06803F")],
        baseLayout({ margin: { l: 44, r: 8, t: 6, b: 56 }, barmode: "group", bargap: 0.25, bargroupgap: 0.05, hovermode: "x unified",
          legend: { orientation: "h", x: 0, y: -0.2, yanchor: "top", font: { size: 10.5 } },
          xaxis: axis({ type: "category", categoryarray: x, tickvals: x.filter((_, i) => i % 3 === 0), ticktext: x.filter((_, i) => i % 3 === 0).map(monthShort) }),
          yaxis: axis({ tickformat: ",.0f", title: { text: "число МО", font: { size: 10.5 } } }), shapes }), STATIC_CONFIG);
    } else if (c.chart === "marketplaces_by_type" && S.trans && S.trans.marketplaces_by_type) {
      const mbt = S.trans.marketplaces_by_type;
      const ks = Object.keys(mbt).map(Number).sort((a, b) => a - b);
      Plotly.newPlot(target, ks.map((k) => ({
        type: "scatter", mode: "lines", name: typeShort(k), x, y: mbt[String(k)], line: { width: 2, color: typeColor(k) },
        hovertemplate: `${typeShort(k)}: %{y:.1f} %<extra></extra>`,
      })), baseLayout({ margin: { l: 44, r: 8, t: 6, b: 64 }, hovermode: "x unified", legend: { orientation: "h", x: 0, y: -0.2, yanchor: "top", font: { size: 10.5 } },
        xaxis: xa, yaxis: axis({ ticksuffix: " %", title: { text: "медианная доля маркетплейсов", font: { size: 10.5 } } }), shapes }), STATIC_CONFIG);
    } else if (L.level) {
      Plotly.newPlot(target, [{
        type: "scatter", mode: "lines", x, y: L.level, line: { color: "#4a5360", width: 2 },
        marker: { size: 7, color: x.map((_, i) => typeColor(conf[i])), line: { color: "#fff", width: 1 } },
        hovertemplate: "%{x}: %{y:,.0f} ₽<extra></extra>",
      }], baseLayout({ margin: { l: 52, r: 8, t: 6, b: 28 }, showlegend: false, xaxis: xa, yaxis: axis({ tickformat: ",.0f", ticksuffix: " ₽" }), shapes }), STATIC_CONFIG);
    }
  }

  // ---------------------------------------------------------------- TOC
  function initToc() {
    const links = [...document.querySelectorAll(".toc a")];
    const map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    const obs = new IntersectionObserver((entries) => {
      for (const e of entries) if (e.isIntersecting) {
        links.forEach((a) => a.classList.remove("active"));
        const a = map.get(e.target.id); if (a) a.classList.add("active");
      }
    }, { rootMargin: "-30% 0px -60% 0px" });
    map.forEach((_, id) => { const s = document.getElementById(id); if (s) obs.observe(s); });
  }

  // ---------------------------------------------------------------- boot
  async function boot() {
    try {
      if (!window.Plotly) throw new Error("не загрузилась библиотека Plotly.js (CDN недоступен?)");
      await loadData();
    } catch (e) {
      console.error(e);
      const box = $("#load-error");
      box.textContent = `Не удалось загрузить данные: ${e.message}. Если страница открыта как файл, запустите локальный сервер: python -m http.server 8765 -d site`;
      box.hidden = false;
      return;
    }
    renderHeader();
    initControls();
    renderLegend();
    initToc();
    await initMap();
    renderTypes();
    renderDynamics();
    renderMethods();
    renderCases();
    // для отладки и проверки производительности
    window.pulsar = { S, setMonth, selectMO, setFocus, updateMap };
    // якорь из адреса: после асинхронной отрисовки разделы сдвинулись — докручиваем заново
    if (location.hash.length > 1) setTimeout(() => { const tgt = document.getElementById(location.hash.slice(1)); if (tgt) tgt.scrollIntoView({ behavior: "instant" }); }, 250);
  }

  document.addEventListener("DOMContentLoaded", boot);
})();

// PDF viewer
(function () {
  const modal = document.getElementById('pdf-modal');
  if (!modal) return;
  const frame = modal.querySelector('iframe');
  const close = () => { modal.hidden = true; document.body.classList.remove('pdf-open'); };
  document.querySelectorAll('a[data-pdf]').forEach(a => a.addEventListener('click', e => {
    e.preventDefault();
    if (window.innerWidth < 700) { window.open(a.getAttribute('href'), '_blank', 'noopener'); return; }
    if (!frame.src) frame.src = frame.dataset.src;
    modal.hidden = false; document.body.classList.add('pdf-open');
  }));
  document.getElementById('pdf-close').addEventListener('click', close);
  modal.addEventListener('click', e => { if (e.target === modal) close(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && !modal.hidden) close(); });
})();
