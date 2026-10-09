"""Загрузка исходных данных: скачивание, проверка SHA-256, распаковка, чтение.

Запуск: `python -m pulsar.data` (или `make data`). Данные не хранятся в репозитории.
"""
from __future__ import annotations

import hashlib
import pathlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_config(name: str = "data") -> dict:
    with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: pathlib.Path, insecure: bool = False) -> pathlib.Path:
    """Скачивает файл, если его ещё нет. `insecure` — обход проверки TLS (сертификаты Минцифры на sber.ru)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    print(f"↓ {url}")
    if insecure:
        subprocess.run(["curl", "-k", "-sS", "-L", "--max-time", "900", "-o", str(dst), url], check=True)
    else:
        urllib.request.urlretrieve(url, dst)
    return dst


def unzip_cp866(archive: pathlib.Path, dst: pathlib.Path) -> None:
    """Распаковка zip, имена файлов в котором закодированы в cp866 (macOS unzip не умеет -O)."""
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename
            if not (info.flag_bits & 0x800):  # нет UTF-8 флага → имя в cp866
                try:
                    name = name.encode("cp437").decode("cp866")
                except UnicodeError:
                    pass
            target = dst / name
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def fetch_all(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    s = cfg["sberindex"]
    arc = download(s["url"], ROOT / s["archive"], insecure=True)
    got = sha256(arc)
    if got != s["sha256"]:
        sys.exit(f"SHA-256 не совпадает: {got} != {s['sha256']}. Удалите {arc} и повторите.")
    print(f"SHA-256 ok: {arc.name}")
    if not (ROOT / s["files"]["consumption"]).exists():
        unzip_cp866(arc, ROOT / s["extract_to"])
    d = cfg["municipal_dict"]
    rar = download(d["url"], ROOT / d["archive"], insecure=True)
    if not (ROOT / d["xlsx"]).exists():
        out = ROOT / d["extract_to"]
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["bsdtar", "-xf", str(rar), "-C", str(out)], check=True)  # bsdtar читает RAR5
    r = cfg["rosstat_bdmo"]
    for key, rel in r["indicators"].items():
        download(f"{r['base']}/{rel}", ROOT / r["dir"] / pathlib.Path(rel).name)
    print("Все данные на месте")


# ---- чтение -----------------------------------------------------------------

def read_consumption(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = pd.read_parquet(ROOT / cfg["sberindex"]["files"]["consumption"])
    df["territory_id"] = df["territory_id"].astype("int32")
    return df


def read_market_access(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_parquet(ROOT / cfg["sberindex"]["files"]["market_access"])


def read_connection(cfg: dict | None = None, kind: str | None = "highway") -> pd.DataFrame:
    cfg = cfg or load_config()
    df = pd.read_parquet(ROOT / cfg["sberindex"]["files"]["connection"])
    return df[df["type"] == kind] if kind else df


def read_municipal_dict(cfg: dict | None = None, year: int = 2024) -> pd.DataFrame:
    """Версии справочника, актуальные в `year` (year_from ≤ year < year_to), одна строка на territory_id."""
    cfg = cfg or load_config()
    d = pd.read_excel(ROOT / cfg["municipal_dict"]["xlsx"])
    cur = d[(d.year_from <= year) & (d.year_to > year)]
    missing = set(d.territory_id) - set(cur.territory_id)
    # для МО без актуальной версии берём последнюю доступную
    last = d[d.territory_id.isin(missing)].sort_values("year_to").drop_duplicates("territory_id", keep="last")
    out = pd.concat([cur, last]).drop_duplicates("territory_id")
    out["oktmo8"] = out["oktmo"].astype(str).str.replace("-", "", regex=False).str.zfill(8).str[:8]
    return out.reset_index(drop=True)


if __name__ == "__main__":
    fetch_all()
