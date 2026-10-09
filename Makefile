# Pulsar — воспроизводимый конвейер. `make all` — от скачивания данных до отчётных артефактов.
PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: all env data external features geo graphs cluster compare dynamics interpret site figures report test clean

all: data external features geo graphs cluster compare dynamics interpret site figures report

env:                       ## окружение: python3.12 -m venv .venv && make env
	$(PY) -m pip install -q -r requirements.txt && $(PY) -m pip install -q -e .

data:                      ## скачать и проверить данные (SHA-256), распаковать
	$(PY) -m pulsar.data

external:                  ## Росстат БДМО → таблица внешних переменных (только для проверки типов)
	$(PY) -m pulsar.external

geo:                       ## упрощённая география МО для лендинга
	$(PY) -m pulsar.geo --no-preview

features: data             ## признаки МО по месяцам + матрица расстояний
	$(PY) -m pulsar.features

graphs: features           ## помесячные графы: три слоя + слияние
	$(PY) -m pulsar.build_graphs

cluster: graphs            ## методы кластеризации × месяцы × K (помесячные) + temporal Leiden (отдельно, долго)
	$(PY) -m pulsar.run_cluster monthly
	$(PY) -m pulsar.run_cluster temporal

compare: cluster           ## устойчивость по K, ICVI, нулевые модели, бутстрэп, рейтинг, абляция рёбер
	$(PY) -m pulsar.k_stability
	$(PY) -m pulsar.compare

dynamics: compare          ## сквозные типы, переходы, события, out-of-time
	$(PY) -m pulsar.run_dynamics

interpret: dynamics        ## профили, внешняя проверка, кейсы
	$(PY) -m pulsar.run_interpret

site: interpret            ## данные для лендинга
	$(PY) -m pulsar.site

figures: site              ## рисунки для отчёта и слайдов
	$(PY) -m pulsar.figures

report:                    ## PDF отчёта (docs/report/report.tex, tectonic) и слайдов (docs/slides.md, Chrome)
	$(PY) scripts/build_pdf.py

test:                      ## юнит-тесты
	$(PY) -m pytest tests -q

clean:
	rm -rf outputs
