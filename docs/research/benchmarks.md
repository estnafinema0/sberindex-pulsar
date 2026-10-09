# Эталонные подходы и бенчмарки для трека «Кластеризация» (СберИндекс)

Дата проверки источников: 2026-10-07. Контекст: ~2000 МО × 24 месяца (2023–2024), 6 категорий трат (`consumption.parquet`), плюс `market_access.parquet`, `connection.parquet` (авто/ж-д расстояния между центрами МО); критерии жюри и обязательный набор ICVI (SW, CH, S_Dbw, AVI, AVU, MQ) – см. `contest_rules.md` и `polozhenie.txt` (стр. 192–195).

Легенда: **ДЕЛАТЬ** – входит в минимальный конвейер; **ОПЦ.** – расширенный конвейер, если остаётся время; **НЕ ДЕЛАТЬ** – осознанно отказываемся и объясняем почему в отчёте. Все названия функций проверены по официальной документации (ссылки даны); там, где проверить не удалось, стоит пометка «⚠ сверить».

---

## 0. Сводная таблица решений

| № | Что | Решение | Зачем (одна фраза) | Библиотека / функция |
|---|-----|---------|--------------------|----------------------|
| 1a | Консенсус-кластеризация (Monti 2003) по признакам | ДЕЛАТЬ | даёт матрицу согласия + выбор K по ΔAUC CDF, жюри узнаёт классику | `consensusclustering.ConsensusClustering` или свой код (~40 строк) |
| 1b | Консенсус для графовых разбиений (Lancichinetti–Fortunato 2012) | ДЕЛАТЬ | устраняет зависимость Leiden от seed, даёт «устойчивое» разбиение | `leidenalg.find_partition(..., seed=s)` × N прогонов → матрица со-встречаемости |
| 1c | Бутстрэп-устойчивость кластеров по Жаккару (Hennig 2007) | ДЕЛАТЬ | кластер-специфичный показатель «растворился/устойчив» с общепринятыми порогами | свой код; `sklearn.metrics.adjusted_rand_score` для парных ARI |
| 1d | SigClust | НЕ ДЕЛАТЬ | только R-пакет, рассчитан на HDLSS (p ≫ n), у нас n ≫ p | – |
| 1e | Gap statistic | ОПЦ. | только для k-means-базы, как доп. свидетельство по K | `gap_statistic.OptimalK` (pip `gap-stat`) |
| 1f | Нулевая модель для ICVI/модулярности (перестановки, конфигурационная модель) | ДЕЛАТЬ | без неё модулярность и силуэт ничего не доказывают (Guimerà 2004) | `networkx.double_edge_swap`, `igraph.Graph.Degree_Sequence(method="vl")`, `snf.metrics.affinity_zscore` |
| 2a | Взаимный kNN + самонастраивающееся ядро (Zelnik-Manor & Perona 2004) | ДЕЛАТЬ (главное правило) | разреженный, адаптивный к плотности граф, стандарт для спектральной кластеризации | `sklearn.neighbors.kneighbors_graph(mode="distance")` + формула σ_i σ_j |
| 2b | SNF (Wang 2014) | ДЕЛАТЬ (второе правило) | честно сливает разнородные «виды» (уровни трат, динамика, сезонность, доступность рынков) | `snf.make_affinity`, `snf.snf`, `snf.get_n_clusters` (pip `snfpy`) |
| 2c | Корреляции с усадкой Ледуа–Вольфа | ОПЦ. | закрывает требование жюри «корреляции между временными рядами», с корректной регуляризацией при T=24 | `sklearn.covariance.LedoitWolf` / `ledoit_wolf` |
| 2d | DTW с окном Сако–Чиба | ОПЦ. (как сравнение, не как основа) | жюри прямо называет DTW; показать, что для месячных рядов выигрыш мал | `dtaidistance.dtw.distance_matrix_fast(window=…)`, `tslearn.metrics.cdist_dtw(global_constraint="sakoe_chiba")` |
| 2e | Гравитационные/пространственные веса | ОПЦ. (как отдельный «вид» в SNF или sensitivity) | в чистом виде воспроизводит географию, а не тип экономики | свой код по `connection.parquet` (гравитация) или `sklearn.neighbors.BallTree(metric="haversine")` |
| 3a | База: k-means, Ward на признаках | ДЕЛАТЬ | без базы нельзя показать, что сеть что-то добавляет | `sklearn.cluster.KMeans`, `AgglomerativeClustering(linkage="ward")` |
| 3b | Spectral clustering на предвычисленной матрице | ДЕЛАТЬ | классический метод на графе близости, прямо сопоставим с k-means при том же K | `sklearn.cluster.SpectralClustering(affinity="precomputed")` |
| 3c | Leiden с весами (RB-модулярность / CPM) | ДЕЛАТЬ | де-факто стандарт community detection; гарантирует связные сообщества | `leidenalg.find_partition`, `igraph.Graph.community_leiden` |
| 3d | Temporal/multislice Leiden (Mucha 2010) | ДЕЛАТЬ | даёт согласованные метки по 24 срезам без пост-сопоставления | `leidenalg.find_partition_temporal(interslice_weight=ω)` |
| 3e | Louvain | НЕ ДЕЛАТЬ (упомянуть) | Traag 2019: до 16 % сообществ несвязны; Leiden доминирует | – |
| 3f | DMoN (Tsitsulin 2023) | ОПЦ. | единственный GNN-метод, который реально запускается за вечер на 2000 узлах | `torch_geometric.nn.dense.DMoNPooling` |
| 3g | Единый выбор K | ДЕЛАТЬ | иначе сравнение методов нечестно | сетка K; для Leiden – подбор resolution под K; eigengap как проверка |
| 4a | Сопоставление кластеров между срезами по Жаккару + венгерский алгоритм | ДЕЛАТЬ | стабильные цвета/метки на картах и Sankey | `scipy.optimize.linear_sum_assignment(maximize=True)` |
| 4b | События Greene 2010 (birth/death/merge/split/continue) | ДЕЛАТЬ | готовая реализация, понятная терминология | `cdlib.LifeCycle(tc).compute_events("greene", threshold=…)` |
| 4c | Матрица переходов + индекс Шоррокса | ДЕЛАТЬ | один эконом-читаемый скаляр мобильности | свой код (5 строк) |
| 4d | Фильтр «реальный переход vs шум» | ДЕЛАТЬ | иначе карта динамики – мерцание | run-length ≥ 2–3 мес. + бутстрэп-вероятность метки + чувствительность к ω |
| 4e | tnetwork / dynetx / TILES | НЕ ДЕЛАТЬ | tnetwork слабо поддерживается; dynetx – только структура данных; TILES требует поток взаимодействий | – |
| 5a | Профили кластеров «по Миркину» (относит. отклонение от общего среднего) | ДЕЛАТЬ | самое эконом-читаемое: «общепит +35 % к среднему по стране» | pandas |
| 5b | Краскел–Уоллис + η²_H | ДЕЛАТЬ | ранжирует признаки по «различающей силе», не только p-value | `scipy.stats.kruskal`; η² = (H−k+1)/(n−k) |
| 5c | Дерево решений как суррогат (правила) | ДЕЛАТЬ | 3–5 правил вида «если доля маркетплейсов > … и транспорт < …» | `sklearn.tree.DecisionTreeClassifier`, `export_text` |
| 5d | SHAP + LightGBM | ОПЦ. | красиво, но менее читаемо экономистом, чем дерево | `shap.TreeExplainer` |
| 5e | Внешняя валидация: V Крамера / NMI с типологиями МО | ДЕЛАТЬ | доказывает, что кластеры ≠ просто регионы или ≠ просто размер | `scipy.stats.contingency.association(method="cramer")`, `sklearn.metrics.normalized_mutual_info_score` |
| 6a | Границы МО: датасет СберИндекса | ДЕЛАТЬ | единственный источник с `territory_id` в постоянных границах | mapshaper / geopandas + `topojson` |
| 6b | Интерактивная карта с ползунком месяцев на GitHub Pages | ДЕЛАТЬ | «отлично» по критерию визуализации = интерактив | Plotly.js (предрасчитанный JSON + TopoJSON) |
| 6c | Аллювиальная диаграмма (Sankey) переходов | ДЕЛАТЬ | динамика кластеров в одном кадре | `plotly.graph_objects.Sankey` |
| 6d | «Паспорт МО» | ОПЦ. | сильный интерактив, но съедает время | один HTML + dropdown + JSON |
| 6e | Картограмма / hex-grid | НЕ ДЕЛАТЬ (hex – ОПЦ.) | нет проверенной Python-библиотеки картограмм для 2000 полигонов; врезка Европейской части решает ту же задачу | `h3` (опц.) |
| 7 | Воспроизводимость: Makefile, YAML, seeds, uv.lock, тесты, таблица времени | ДЕЛАТЬ | 30 % веса; тай-брейк по интерпретации | `uv lock`, `uv sync --locked`, pytest |

---

## 1. Устойчивость и значимость кластеров

### 1.1 Консенсус-кластеризация (Monti et al. 2003) – ДЕЛАТЬ
- **Источник:** Monti, Tamayo, Mesirov, Golub (2003) «Consensus Clustering: A Resampling-Based Method for Class Discovery…», *Machine Learning* 52:91–118, DOI 10.1023/A:1023949509487 – https://mlanthology.org/mlj/2003/monti2003mlj-consensus/
- **Суть:** H субсэмплов (≈80 % объектов), на каждом – кластеризация при K; матрица согласия M_K(i,j) = доля прогонов, где i и j вместе; K выбирают по максимуму прироста площади под CDF согласия (ΔAUC) или по минимуму PAC.
- **Python:** пакет `consensusclustering` (PyPI 0.2.4, 19.09.2025, Python ≥ 3.11, MIT) – https://pypi.org/project/consensusclustering/. Конструктор (проверено по исходнику https://github.com/burtonrj/consensusclustering): `ConsensusClustering(clustering_obj, min_clusters, max_clusters, n_resamples, resample_frac=0.5, k_param="n_clusters", rng=None)`; методы `fit()`, `best_k()`, `consensus_k()`, `cdf()`, `area_under_cdf()`, `change_in_area_under_cdf()`, `plot_cdf()`, `plot_change_area_under_cdf()`, `plot_clustermap()`, `plot_hist()`. Любой sklearn-совместимый `clustering_obj` с `fit_predict` и `set_params` (KMeans, SpectralClustering на признаках).
- **Если не хочется зависимости:** своя реализация – 40 строк (цикл по K и субсэмплам, `np.add.at` в матрицу согласия). Для 2000 объектов матрица 2000×2000 float32 = 16 МБ, H=100 – секунды.
- **Для графовых методов** – консенсус по Lancichinetti & Fortunato (2012) «Consensus clustering in complex networks», *Sci. Rep.* 2:336, https://doi.org/10.1038/srep00336 (arXiv 1203.6093): N прогонов Leiden с разными `seed`, матрица со-встречаемости D, обнуление D < τ, повторный Leiden на D как на взвешенном графе до сходимости. Авторы отдельно отмечают пригодность для мониторинга эволюции сообществ во времени. Реализация: `leidenalg.find_partition(G, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=γ, seed=s)` (сигнатура `find_partition(graph, partition_type, initial_membership=None, weights=None, n_iterations=2, max_comm_size=0, seed=None, **kwargs)` – https://leidenalg.readthedocs.io/en/stable/reference.html).

### 1.2 Устойчивость по бутстрэпу/субсэмплингу – ДЕЛАТЬ
- **Ben-Hur, Elisseeff, Guyon (2002)** «A stability based method for discovering structure in clustered data», *PSB* 2002, pp. 6–17: для каждого K берут пары субсэмплов, кластеризуют, считают сходство разбиений на пересечении; распределение сходства, сосредоточенное у 1, = устойчивое K. Python: цикл + `sklearn.metrics.adjusted_rand_score` (или `fowlkes_mallows_score`).
- **Lange, Roth, Braun, Buhmann (2004)** «Stability-Based Validation of Clustering Solutions», *Neural Computation* 16:1299–1323, DOI 10.1162/089976604773717621 – https://mlanthology.org/neco/2004/lange2004neco-stabilitybased/ : устойчивость как риск классификации (обучить классификатор на разбиении половины A, предсказать половину B, сравнить с кластеризацией B; нормировать на случайное). Полезно как обоснование, реализовывать необязательно – 1.1 + Hennig достаточно.
- **Hennig (2007)** «Cluster-wise assessment of cluster stability», *CSDA* 52:258–271; реализация `fpc::clusterboot` (R) – https://search.r-project.org/CRAN/refmans/fpc/html/clusterboot.html. Для каждого исходного кластера: на B бутстрэп-выборках кластеризуют заново и берут максимальный Жаккар с любым новым кластером; среднее по B = устойчивость кластера. Пороги из документации fpc: ≥ 0.75 – устойчивый, 0.6–0.75 – «паттерн есть, границы неточны», < 0.5 – «растворился». Порта на Python нет → свой код (~40 строк): `idx = rng.choice(n, n, replace=True)` (или субсэмпл 80 % без возврата – для спектральных методов стабильнее), пересчёт графа и разбиения на подвыборке, Жаккар по множествам `territory_id`. Для Leiden по графу то же самое – удалять 20 % узлов (`G.subgraph`) и перезапускать.
- **Вывод в отчёт:** таблица «кластер → средний Жаккар (B=100), доля растворений», плюс ARI между 5 seed-прогонами (среднее ± sd).

### 1.3 Значимость кластерной структуры
- **SigClust (Liu, Hayes, Nobel, Marron 2008, *JASA* 103(483):1281–1293)** – НЕ ДЕЛАТЬ: реализация только в R (`sigclust`, https://cran.r-project.org/web/packages/sigclust/), тест рассчитан на p ≫ n, проверяет только расщепление на 2 кластера (итеративно); у нас n≈2000 ≫ p – гипотеза «одна гауссиана» отвергнется тривиально. Упомянуть в отчёте как осознанный отказ.
- **Gap statistic (Tibshirani, Walther, Hastie 2001, *JRSS-B* 63:411–423)** – ОПЦ. только для k-means: `pip install gap-stat`, `from gap_statistic import OptimalK; optimalK = OptimalK(n_jobs=4, parallel_backend="joblib"); k = optimalK(X, cluster_array=np.arange(2, 15), n_refs=50)`; `optimalK.gap_df` содержит `gap_value, gap*, sk, diff` – https://github.com/milesgranger/gap_statistic. Для графовых методов gap неприменим напрямую.
- **ICVI против нулевой модели – ДЕЛАТЬ (главный «тест значимости»):**
  1. *Перестановочная нулевая для признаковых ICVI (SW, CH, S_Dbw):* независимо перемешать каждый столбец признаков (разрушает корреляции, сохраняет маргиналы – та же логика, что «reference distribution» у Tibshirani 2001), прогнать тот же метод при том же K, 200 повторов → z-score и p-value для наблюдаемого индекса. Просто перестановка меток даёт слишком слабую нулевую (силуэт ≈ 0 всегда) – использовать только как sanity check.
  2. *Нулевая для графовых индексов (модулярность, MQ, AVI, AVU):* Guimerà, Sales-Pardo, Amaral (2004) «Modularity from fluctuations in random graphs and complex networks», *PRE* 70:025101 – https://doi.org/10.1103/PhysRevE.70.025101 : у случайных графов модулярность заметно > 0, поэтому нужна Q_null с тем же распределением степеней. Генерация: `networkx.double_edge_swap(G, nswap=10*m, max_tries=…, seed=…)` (перестановка рёбер с сохранением степеней; https://networkx.org/documentation/stable/reference/algorithms/swap.html) или `networkx.configuration_model(deg_sequence, seed=…)` → `nx.Graph(G)`; `G.remove_edges_from(nx.selfloop_edges(G))` (мультиграф, нужна очистка – https://networkx.org/documentation/stable/reference/generated/networkx.generators.degree_seq.configuration_model.html); в igraph – `Graph.Degree_Sequence(degs, method="vl")` (Viger–Latapy, простые связные графы) или `Graph.rewire()` (⚠ сверить сигнатуру под установленной версией: https://python.igraph.org/en/stable/api/igraph.Graph.html). Для взвешенного графа – перемешивать веса по рёбрам после свопа. 100 реплик → z = (Q_obs − mean Q_null)/sd.
  3. *Z-score аффинности внутри кластеров:* `snf.metrics.affinity_zscore(arr, labels, n_perms=…, seed=…)` – https://snfpy.readthedocs.io/en/latest/api.html – готовая перестановочная оценка для любой матрицы сходства.

### Что реально за день
Матрица согласия + ΔAUC (1.1), бутстрэп-Жаккар по кластерам (1.2), z-score модулярности и SW против нулевых моделей (1.3) – всё вместе ≈ 3–4 часа кода при N=2000, если граф пересчитывается за секунды (kNN) – SNF пересчитывать на каждом бутстрэпе дороже (dense 2000², t=20 итераций), поэтому для SNF брать B=30–50.

---

## 2. Правила ребра для атрибутированной сети временных рядов

Исходные «виды» (views) одного МО: (а) вектор уровней трат по 6 категориям (лог, нормировка на «Все категории» → структура потребления), (б) 24-месячная динамика (темпы роста г/г, сезонная амплитуда), (в) индекс доступности рынков, (г) расстояния из `connection.parquet`. Правило ребра = как из них получить W_ij.

| Правило | Формула / реализация | Сильная сторона (одной фразой) | Слабая сторона (одной фразой) |
|---|---|---|---|
| **2a. Взаимный kNN + локальное масштабирование** (Zelnik-Manor & Perona, NIPS 2004 – https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering) | `D = kneighbors_graph(X, k, mode="distance")` (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html); σ_i = расстояние до k-го соседа (в оригинале k=7); W_ij = exp(−d_ij²/(σ_i σ_j)); взаимность: `W = W.minimum(W.T)` (оставить ребро, только если i∈kNN(j) и j∈kNN(i)) | адаптируется к разной плотности (мегаполисы vs. сельские районы), разреженный граф, нет глобального σ | при малом k рвётся на компоненты – проверять `nx.number_connected_components` и брать k≈10–15 или добавлять MST-рёбра |
| **2b. SNF** (Wang et al. 2014, *Nature Methods* 11:333) – https://github.com/rmarkello/snfpy | `aff = snf.make_affinity(X_a, X_b, X_c, metric="euclidean", K=20, mu=0.5)`; `W = snf.snf(aff, K=20, t=20)`; `best, second = snf.get_n_clusters(W)`; далее `spectral_clustering(W, n_clusters=best)` (пример из README) | сливает разнородные виды без ручных весов, усиливает согласованные рёбра и гасит шумовые | плотная N×N матрица, чувствительность к K/mu, пакет давно не обновлялся (Python ≥ 3.5) – проверить установку в первый час |
| **2c. Корреляции с усадкой Ледуа–Вольфа** (Ledoit & Wolf 2004, *J. Multivariate Anal.* 88:365–411) | матрица X формы (24 месяца × 2000 МО) по стандартизованным темпам роста; `LedoitWolf().fit(X).covariance_` → корреляции; `shrinkage_` показывает долю усадки (https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html); далее порог/kNN по |ρ| | отвечает на формулировку жюри «корреляция между временными рядами», при T=24 ≪ N=2000 ковариация без усадки вырождена – LW решает это | 24 точки – корреляции шумны, усадка тянет всё к нулю (ожидать `shrinkage_` близко к 1), ловит синхронность, а не тип экономики |
| **2d. DTW с окном Сако–Чиба** | `dtaidistance.dtw.distance_matrix_fast(S, window=2, use_pruning=True, parallel=True)` (https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html) или `tslearn.metrics.cdist_dtw(S, global_constraint="sakoe_chiba", sakoe_chiba_radius=2, n_jobs=-1)` (https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.cdist_dtw.html); ряды z-нормировать; ядро exp(−d²/σ²) | терпим к сдвигу на 1–2 месяца (жюри называет «один регион опережает другой»), 2 млн пар по 24 точкам считаются за минуты | для месячных макрорядов сдвиг редко содержателен; без нормировки доминирует масштаб; DTW не метрика (нет неравенства треугольника) – для Ward некорректен, для спектральной с ядром допустим |
| **2e. Гравитационные / пространственные веса** | w_ij = (P_i P_j)/d_ij^β по `connection.parquet` (distance, type=highway), либо ядро exp(−d/λ) по координатам центров (`sklearn.neighbors.BallTree(metric="haversine")`); использовать как отдельный вид в SNF или как множитель W_attr ⊙ W_geo | интерпретируемые «региональные рынки», согласуется с индексом доступности рынков | в чистом виде кластеры = география (V Крамера с регионом → 1), маскирует экономическое сходство удалённых МО |

### Как показать влияние правила ребра математически
1. **Таблица метрик сети по каждому правилу** (все функции networkx 3.x):
   - число рёбер, плотность `nx.density(G)`; доля узлов в гигантской компоненте, `nx.number_connected_components(G)`;
   - ассортативность по степени `nx.degree_assortativity_coefficient(G, weight="weight")`; по атрибуту «регион» `nx.attribute_assortativity_coefficient(G, "region")`; по численному атрибуту (лог-траты) `nx.numeric_assortativity_coefficient(G, "log_total")` (https://networkx.org/documentation/stable/reference/algorithms/assortativity.html);
   - спектральный зазор: собственные значения нормализованного лапласиана `scipy.sparse.linalg.eigsh(nx.normalized_laplacian_matrix(G), k=K_max+1, which="SM")` → λ_{K+1} − λ_K (eigengap-эвристика, von Luxburg 2007, *Statistics and Computing* 17(4) – https://arxiv.org/abs/0711.0189); алгебраическая связность `nx.algebraic_connectivity(G, weight="weight", normalized=True)` (https://networkx.org/documentation/stable/reference/generated/networkx.linalg.algebraicconnectivity.algebraic_connectivity.html);
   - средний кластерный коэффициент `nx.average_clustering(G, weight="weight")`; модулярность лучшего разбиения.
2. **Матрица согласия разбиений между правилами** при одном K и одном методе: ARI/NMI/AMI (`sklearn.metrics.adjusted_rand_score`, `normalized_mutual_info_score`, `adjusted_mutual_info_score`) → тепловая карта 5×5; плюс та же матрица «метод × метод» при фиксированном правиле. Это и есть «сравнение подходов к рёбрам» из Приложения 1.
3. **Чувствительность ICVI к правилу:** признаковые ICVI (SW, CH, S_Dbw) считать в одном и том же пространстве признаков для всех правил, графовые (MQ, AVI, AVU, модулярность) – на графе соответствующего правила и дополнительно на общем «референсном» графе (взаимный kNN), чтобы сравнение было честным.

---

## 3. Методы кластеризации и честное сравнение

### Состав сравнения (по таксономии обзоров)
- **Chunaev (2020)** «Community detection in node-attributed social networks: a survey», *Computer Science Review* 37 – https://arxiv.org/abs/1912.09816 : классификация по моменту слияния структуры и атрибутов – early fusion (атрибуты → веса рёбер, потом структурный метод), simultaneous (совместная оптимизация), late fusion (объединение разбиений).
- **Bothorel, Cruz, Magnani, Micenková (2015)** «Clustering attributed graphs: models, measures and methods», *Network Science* 3(3):408–444 – https://arxiv.org/abs/1501.01676 : обзор моделей, мер качества и методов, отдельно – проблема оценки.

У нас сеть сама строится из атрибутов, поэтому честный набор:

| Группа | Метод | Функция | Роль |
|---|---|---|---|
| Только атрибуты | k-means (`KMeans(n_clusters=K, n_init=10, random_state=s)`), Ward (`AgglomerativeClustering(n_clusters=K, linkage="ward")`) | sklearn | база: «что даёт сеть сверх признаков» |
| Только структура (на графе правила 2a/2b) | Spectral (`SpectralClustering(n_clusters=K, affinity="precomputed", assign_labels="cluster_qr", random_state=s)` – https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html; `cluster_qr` без случайной инициализации, `discretize` менее чувствителен к seed), Leiden (`leidenalg.find_partition(G, la.RBConfigurationVertexPartition, weights="weight", resolution_parameter=γ, n_iterations=-1, seed=s)`; эквивалент `igraph.Graph.community_leiden(objective_function="modularity", weights="weight", resolution=γ)` – https://python.igraph.org/en/stable/api/igraph.Graph.html) | sklearn, leidenalg/igraph | основные |
| Early fusion | те же Spectral/Leiden на SNF-графе | snfpy | показывает вклад многовидового слияния |
| Временная структура | `leidenalg.find_partition_temporal([G_1,…,G_24], la.CPMVertexPartition, interslice_weight=ω, resolution_parameter=γ)` (возвращает membership по срезам; пример в документации использует CPM – https://leidenalg.readthedocs.io/en/stable/multiplex.html); теория – Mucha et al. 2010, *Science* 328:876 – https://arxiv.org/abs/0911.1824 | leidenalg | динамика без пост-сопоставления |
| Simultaneous (GNN) – ОПЦ. | DMoN: `torch_geometric.nn.dense.DMoNPooling(channels, k, dropout=0.0)` возвращает (S, X', A', spectral_loss, ortho_loss, cluster_loss) – https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.DMoNPooling.html; статья Tsitsulin, Palowitch, Perozzi, Müller (2023) *JMLR* 24(127):1–21 – https://www.jmlr.org/papers/v24/20-998.html; официальный TF2-код – https://github.com/google-research/google-research/tree/master/graph_embedding/dmon | PyG | «мы попробовали GNN»; 2000 узлов – dense adjacency помещается в память; фиксировать `torch.manual_seed` и усреднять по 5 seed |

Louvain не включать в основную таблицу: Traag, Waltman, van Eck (2019) «From Louvain to Leiden», *Sci. Rep.* 9:5233 – https://arxiv.org/abs/1810.08473 – показывают до 25 % плохо связанных и до 16 % несвязных сообществ.

### Единый выбор K
1. Фиксированная сетка K = 4…12 для всех методов (экономическая интерпретация > 12 типов невозможна).
2. Для Leiden/temporal Leiden K не задаётся напрямую → для каждого K подбирать `resolution_parameter` бисекцией до получения ровно K сообществ (или ближайшего; `Optimiser.resolution_profile(resolution_range=…)` строит профиль разрешений – https://leidenalg.readthedocs.io/en/stable/reference.html); при нескольких γ с одним K брать максимум модулярности. Сообщества размером < 10 МО сливать в «прочее» и честно показывать это.
3. Один и тот же набор ICVI для всех (см. ниже) + устойчивость (1.1–1.2) → правило выбора записать в `configs/selection.yaml` (например: «K* = argmax среднего ранга по SW, CH, −S_Dbw, MQ, AVI среди K с бутстрэп-Жаккаром ≥ 0.75»). Eigengap (п. 2) – как независимое подтверждение.

### Обязательный набор ICVI (Положение, стр. 192–195)
- **SW** `sklearn.metrics.silhouette_score(X, labels)` и на графе – `silhouette_score(1 − W, labels, metric="precomputed")`; **CH** `calinski_harabasz_score`; **DB** (не требуется, но бесплатно) `davies_bouldin_score`.
- **S_Dbw** (Halkidi & Vazirgiannis, ICDM 2001): `pip install s-dbw`; `from s_dbw import S_Dbw; S_Dbw(X, labels, centers_id=None, method="Tong", alg_noise="bind", centr="mean", nearest_centr=True, metric="euclidean")`, меньше – лучше; методы `Halkidi` / `Kim` / `Tong` – https://pypi.org/project/s-dbw/ , https://github.com/alashkov83/S_Dbw .
- **AVI, AVU** – Biswas & Biswas (2017) «Defining quality metrics for graph clustering evaluation», *Expert Systems with Applications* 71:1–17, https://doi.org/10.1016/j.eswa.2016.11.011. AVI = среднее по кластерам «изолируемости» = доля внутренних связей кластера среди всех связей его узлов (формула подтверждена по arXiv 2212.10797, уравнения 19–20, ссылающимся на эту статью). ⚠ Определение AVU (unifiability) взять дословно из статьи – готовой Python-реализации нет, писать свою (десятки строк на sparse-матрице) и покрыть тестом на SBM-графе.
- **MQ** – Mancoridis, Mitchell, Rorres, Chen, Gansner (1998) «Using automatic clustering to produce high-level system organizations of source code», IWPC'98 – https://researchdiscovery.drexel.edu/esploro/outputs/conferenceProceeding/Using-automatic-clustering-to-produce-high-level/991019167445804721. Используемая в Bunch (TurboMQ) форма: MF_k = i_k/(i_k + j_k/2) при i_k > 0 (i – вес внутренних рёбер, j – вес внешних), MQ = Σ_k MF_k; в отчёте явно указать, какая версия (BasicMQ 1998 или TurboMQ) реализована. Реализация – 10 строк на `scipy.sparse`.
- Модулярность и кондактанс – как дополнительные: `networkx.community.modularity(G, communities, weight="weight")`, `igraph.Graph.modularity`.
- Все ICVI сопровождать z-score против нулевой модели (п. 1.3) – это и есть «комплексный анализ качества» на «отлично».

---

## 4. Динамика кластеров

### 4.1 Сопоставление между срезами – ДЕЛАТЬ
- Матрица Жаккара J[a,b] = |C_a^t ∩ C_b^{t+1}| / |C_a^t ∪ C_b^{t+1}| по множествам `territory_id`; выравнивание меток один-к-одному: `row, col = scipy.optimize.linear_sum_assignment(J, maximize=True)` (работает для прямоугольных матриц – https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html). Нужно для постоянных цветов на картах и Sankey.
- Если основной метод – temporal Leiden, метки уже согласованы по срезам (узел i в срезе t и t+1 связан межслойным ребром веса ω) – сопоставление требуется только для per-slice методов.

### 4.2 События Greene et al. 2010 – ДЕЛАТЬ
- Greene, Doyle, Cunningham (2010) «Tracking the Evolution of Communities in Dynamic Social Networks», ASONAM'10, pp. 176–183 – https://research.google/pubs/tracking-the-evolution-of-communities-in-dynamic-social-networks/ : many-to-many сопоставление по Жаккару с порогом θ; события birth/death/merge/split/continue.
- CDlib (0.4.1, июль 2026, Python ≥ 3.8 – https://pypi.org/project/cdlib/), проверено по https://cdlib.readthedocs.io/en/latest/reference/events.html :
  ```python
  from cdlib import TemporalClustering, LifeCycle, NodeClustering
  tc = TemporalClustering()
  for t, labels_t in enumerate(partitions):          # 24 среза
      coms = [list(ids) for ids in groups(labels_t)]  # списки territory_id
      tc.add_clustering(NodeClustering(coms, graph=None, method_name="leiden"), t)
  events = LifeCycle(tc)
  events.compute_events("greene", threshold=0.3)      # по умолчанию 0.1; "facets" / "asur" – альтернативы
  events.get_event_types(); ev = events.get_event("1_2"); ev.out_flow; events.analyze_flow("1_2", "+")
  from cdlib.viz import plot_flow, plot_event_radar
  fig = plot_flow(events)
  ```
  `tc.clustering_stability_trend(cdlib.evaluation.adjusted_rand_index)` даёт ряд ARI(t, t+1) – https://cdlib.readthedocs.io/en/latest/reference/classes/temporal_clustering.html.
- Порог θ: для ~2000 МО и K≈6–10 брать 0.3–0.5 (0.1 из статьи рассчитан на мелкие соцсети); показать чувствительность числа событий к θ.

### 4.3 Матрица переходов и индекс Шоррокса – ДЕЛАТЬ
- P_t (K×K) – доли МО кластера a в месяце t, оказавшихся в b в t+1 (после выравнивания меток). Индекс Шоррокса M(P) = (K − tr P)/(K − 1): 0 – никто не меняет кластер, 1 – полная перемешанность. Shorrocks (1978) «The Measurement of Mobility», *Econometrica* 46(5):1013–1024 – https://www.econometricsociety.org/publications/econometrica/1978/09/01/measurement-mobility. Считать помесячно и за весь период (t=1 → t=24), плюс долю МО, ни разу не сменивших кластер.

### 4.4 Реальный переход vs шум – ДЕЛАТЬ
1. **Run-length фильтр:** смена метки засчитывается, если новая метка держится ≥ 2 (лучше 3) месяца подряд; одиночные «вспышки» помечаются как шум и показываются серым.
2. **Бутстрэп-вероятность метки:** B=50 субсэмплов (п. 1.2) → для МО i и месяца t доля прогонов с данной меткой; переход «значим», если P(label_t ≠ label_{t−1}) ≥ 0.8.
3. **Чувствительность к ω** (temporal Leiden): переход считать устойчивым, если он есть при ω ∈ {0.5, 1, 2} (Mucha 2010: ω – сила связи узла с собой между срезами; больше ω – меньше мерцания).
4. **Нулевая для Шоррокса:** перемешать метки внутри месяца → M≈1; наблюдаемый M ≪ 1 говорит об устойчивости типологии.
5. Отдельно отфильтровать сезонность: кластеризовать по рядам, очищенным от сезонности (г/г темпы или STL), иначе «переходы» в декабре/январе будут везде.

### 4.5 Библиотеки: что реально работает
- **cdlib** – да: `TemporalClustering`, `LifeCycle`, `compute_events("greene"|"facets"|"asur")`, `viz.plot_flow` (проверено по документации; активно поддерживается).
- **tnetwork** (Cazabet) – есть `tnetwork.DCD.iterative_match` (по Greene) и `label_smoothing`, параметры `CDalgo`, `match_function` (Жаккар), `threshold` – https://tnetwork.readthedocs.io/en/latest/ ; проект небольшой (16 звёзд), устанавливать только если cdlib не хватит. ОПЦ./НЕ ДЕЛАТЬ.
- **dynetx** (0.3.2, июнь 2023 – https://pypi.org/project/dynetx/) – только динамическая структура графа поверх networkx, детекции сообществ нет. **TILES** (`cdlib.algorithms.tiles(dg, obs)`) требует DyNetx-поток взаимодействий, а не атрибутированные срезы – НЕ ДЕЛАТЬ.

---

## 5. Интерпретация кластеров

| Приём | Решение | Реализация | Эконом-читаемость |
|---|---|---|---|
| **Профили по Миркину** – относительное отклонение среднего кластера от общего среднего (c_kv − g_v)/g_v и вклад признака в объяснённый разброс (Mirkin 2005, *Clustering for Data Mining: A Data Recovery Approach*, Chapman & Hall/CRC – https://www.kdnuggets.com/news/2005/n10/25i.html ; главы об интерпретации кластеров) | ДЕЛАТЬ | `df.groupby(label).mean()`; тепловая карта «кластер × категория, % к среднему»; формулировки «в кластере 3 доля общепита на 35 % выше средней по стране» | высшая: это язык аналитических записок |
| **Краскел–Уоллис + η²_H** | ДЕЛАТЬ | `scipy.stats.kruskal(*[x[labels==k] for k in K])`; η²_H = (H − k + 1)/(n − k) (Tomczak & Tomczak 2014, *Trends in Sport Sciences* 21(1):19–25 – https://www.wbc.poznan.pl/dlibra/publication/413565); поправка на множественность `scipy.stats.false_discovery_control` | средняя: ранжирование признаков «что сильнее всего различает типы»; p-value при n=2000 всегда ~0, поэтому показывать только η² |
| **Дерево решений как суррогат** | ДЕЛАТЬ | `DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=s).fit(X, labels)`; `sklearn.tree.export_text(tree, feature_names=…)` (https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html); сообщать fidelity (accuracy суррогата) | высокая: 5–7 правил вида «если доля маркетплейсов > 18 % и доступность рынков < 200 → тип B» |
| **SHAP + LightGBM** | ОПЦ. | `lightgbm.LGBMClassifier(random_state=s)` → `shap.TreeExplainer(model)` (поддерживает XGBoost/LightGBM/CatBoost/sklearn – https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) → `.shap_values(X)` → beeswarm по классам | средняя: красиво для приложения, но требует объяснять, что такое SHAP |
| **Внешняя валидация** | ДЕЛАТЬ | V Крамера `scipy.stats.contingency.association(pd.crosstab(labels, ext), method="cramer")` (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html); NMI `sklearn.metrics.normalized_mutual_info_score(labels, ext)`. Внешние типологии: регион/федеральный округ, тип МО (городской округ / муниципальный район / округ – из справочника СберИндекса), класс численности, квантиль индекса доступности рынков, наличие ж/д (`connection.parquet`, type=railway) | высокая: «кластеры совпадают с регионами лишь на V=0.35, то есть это типы экономик, а не география» |

Порядок в отчёте: имена кластеров + профиль Миркина → карта → правила дерева → таблица η² → внешняя валидация; SHAP – в приложение.

---

## 6. Визуализация муниципальных данных России

### 6.1 Границы МО и подготовка геометрии – ДЕЛАТЬ
- **Источник границ:** датасет СберИндекса «Границы и преобразования муниципальных образований» – https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities (ссылка извлечена из статьи команды СберИндекса на Хабре https://habr.com/ru/articles/849682/ ; страница рендерится JS, имена файлов проверить при скачивании – по условию задачи архив `t_dict_municipal.rar` с GPKG внутри). Поля по описанию в статье: `oktmo`, `municipal_district_name(_short)`, `year_from`/`year_to`, `territory_id` (постоянные границы – тот же ключ, что в `consumption.parquet`), координаты административного центра, тип и статус МО. Запасной вариант: geoBoundaries ADM2 (https://developers.google.cn/earth-engine/datasets/catalog/WM_geoLab_geoBoundaries_600_ADM2?hl=ru) – но без `territory_id`, поэтому только как fallback.
- **Упрощение до < 5 МБ (mapshaper, проверено по https://mapshaper.org/docs/reference.html):**
  ```bash
  npx mapshaper borders.gpkg \
    -clean \
    -simplify 5% keep-shapes \
    -filter-fields territory_id,oktmo,name_short \
    -o format=topojson precision=0.001 quantization=100000 mo.topojson
  ```
  `keep-shapes` не даёт исчезнуть мелким МО; `precision=0.00001` ≈ 1 м на экваторе – для карты страны хватит 0.001; TopoJSON с квантованием обычно в несколько раз меньше GeoJSON. Если > 5 МБ – `-simplify 2%` или `interval=500` (метры). Python-альтернатива: `geopandas.GeoSeries.simplify(tolerance, preserve_topology=True)` + пакет `topojson` (mattijn): `topojson.Topology(gdf, prequantize=1e5, toposimplify=0.01).to_json()` – https://mattijn.github.io/topojson/.
- **Чукотка и 180-й меридиан:** полигоны, пересекающие антимеридиан, надо разрезать по 180°, иначе «полоса через всю карту» (руководство mapshaper https://mapshaper.org/docs/guides/geojson-for-web-maps.html). Python: `pip install antimeridian; antimeridian.fix_geojson(gj)` – https://antimeridian.readthedocs.io. Для Web-Mercator карт (MapLibre) после разреза части Чукотки оказываются по разные стороны – это нормально при `renderWorldCopies=true`; для конических проекций (Plotly geo, d3) разрез + поворот решают проблему полностью.
- **Проекция Альберса для России:** `+proj=aea +lat_1=52 +lat_2=64 +lon_0=100 +lat_0=0 +ellps=WGS84 +units=m` – типовой выбор (параллели внутри основного широтного пояса страны, осевой меридиан ~100° в.д.; официального ГОСТ-стандарта для этих параметров найти не удалось). Готовая проверенная альтернатива – ESRI:102025 «Asia North Albers Equal Area Conic» (`+proj=aea +lat_0=30 +lon_0=95 +lat_1=15 +lat_2=65`, https://epsg.io/102025). В Plotly geo: `fig.update_geos(projection_type="conic equal area", projection_rotation=dict(lon=100), projection_parallels=[52, 64], fitbounds="locations")` – `projection.type` поддерживает `"albers"`, `"conic equal area"`, `"conic conformal"`, `projection.parallels` – «только для конических» (https://plotly.com/python/reference/layout/geo/). В d3: `d3.geoConicEqualArea().parallels([52, 64]).rotate([-100, 0])` (https://d3js.org/d3-geo/conic); Observable Plot использует те же d3-проекции (`projection: {type: "conic-equal-area", parallels: [52, 64], rotate: [-100, 0], domain: topo}`; ⚠ страница docs отдала 429, синтаксис сверить при использовании).

### 6.2 Стек для статического GitHub Pages – ДЕЛАТЬ
- **Данные:** предрасчитанные JSON: `labels.json` (territory_id → массив из 24 меток), `profiles.json`, `transitions.json`, `events.json`, `mo.topojson`. Никакого бэкенда.
- **Карта с ползунком месяцев:** Plotly Python → `fig.write_html("map.html", include_plotlyjs="cdn")` (`px.choropleth(..., geojson=…, locations="territory_id", featureidkey="properties.territory_id", animation_frame="month")` + `update_geos` выше). Для 2000 полигонов и 24 кадров файл будет 10–30 МБ, поэтому: либо `animation_frame` только по кварталам, либо один слой + JS-переключение `z` по месяцам из `labels.json` (Plotly.js `Plotly.restyle`).
- **Если нужны плавные зумы:** MapLibre GL JS + PMTiles – один файл тайлов, читается range-запросами, хостится на GitHub Pages без сервера (https://til.simonwillison.net/gis/pmtiles ; https://protomaps.com/). Это ОПЦ.: Plotly geo достаточно для жюри.
- **Быстрые альтернативы для черновиков:** kepler.gl в Jupyter – `KeplerGl().add_data(gdf, name=…)`, `save_to_html(file_name=…, read_only=True)` (https://docs.kepler.gl/docs/keplergl-jupyter); pydeck `GeoJsonLayer` / `H3HexagonLayer` → `Deck.to_html()` (https://deckgl.readthedocs.io/en/latest/). Минус: данные встраиваются в HTML, легенды кастомизируются хуже.

### 6.3 Остальные виды – что делать
- **Аллювиальная диаграмма переходов** – ДЕЛАТЬ: `go.Sankey(node=dict(label=…, color=…), link=dict(source=…, target=…, value=…), arrangement="snap")` (https://plotly.com/python/sankey-diagram/); узлы = (месяц, кластер), 24 колонки нечитаемы → показывать квартальные срезы или 4–6 опорных месяцев; потоки < 1 % скрывать.
- **Small multiples по месяцам** – ДЕЛАТЬ в статике для отчёта: `geopandas.plot` в сетке 4×6 (matplotlib) или `px.choropleth(facet_col="month", facet_col_wrap=6)`; в интерактиве они избыточны.
- **«Паспорт МО»** – ОПЦ.: одна HTML-страница с выпадающим списком (datalist по 2000 названиям), спарклайны 6 категорий (Plotly small multiples), лента принадлежности к кластерам по 24 месяцам, профиль МО vs. среднее кластера; данные подтягиваются из `labels.json`/`series.json`.
- **Картограмма** – НЕ ДЕЛАТЬ (нет проверенной Python-библиотеки для 2000 полигонов, трудоёмко); **hex-grid** – ОПЦ.: `h3` (hex-биннинг центров МО) или `plotly.figure_factory.create_hexbin_map(lat=…, lon=…, nx_hexagon=…, agg_func=…, min_count=…)` (https://plotly.com/python/hexbin-mapbox/) – только если карта МО окажется нечитаемой из-за огромных северных районов. Вместо этого обязательно: врезка «Европейская часть» и/или нормировка площади цветом + точечная карта центров МО (координаты есть в справочнике).
- **Примеры для ориентира:** «Если быть точным» (https://tochno.st/ – рейтинги регионов, тематические карты, датасет муниципальной статистики https://tochno.st/datasets/bdmo – полезен и как внешняя типология); исследования СберИндекса (https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities и соседние материалы); FT Visual Vocabulary – выбор типа графика по задаче, разделы «Flow» (Sankey/chord) и «Change over time» (https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/README.md); NYT Upshot – карта результатов выборов 2020 по участкам как образец статической, предрасчитанной векторной карты с масштабированием (https://flowingdata.com/2021/02/02/precinct-level-map-of-2020-election-results/ ; данные https://github.com/nytimes/presidential-precinct-map-2024 – 2024-я версия).
- **Цвет:** качественная палитра ≤ 10 категорий (ColorBrewer `Set2`/`Tableau 10`), один и тот же цвет кластера везде (карта, Sankey, профили) – обеспечивается выравниванием меток из п. 4.1.

---

## 7. Воспроизводимость – минимальный чек-лист

1. `make all` = `data → features → graphs → cluster → validate → dynamics → interpret → viz → report`; отдельные цели `make test`, `make clean`, `make timing`. Makefile + `uv run` внутри целей.
2. Один источник правды для гиперпараметров: `configs/*.yaml` (правило ребра, k соседей, K-сетка, γ, ω, θ, B, seeds). Каждый артефакт (таблица, рисунок) записывает в имя/метаданные хеш конфига.
3. Seeds: `numpy.random.default_rng(seed)`, `random_state=seed` в sklearn, `seed=` в `leidenalg.find_partition`/`find_partition_temporal`, `torch.manual_seed` для DMoN; для `eigen_solver="amg"` sklearn требует ещё `np.random.seed` (из документации SpectralClustering). Прогон на 5 seed, в отчёте – среднее ± sd ARI между прогонами.
4. Окружение: `pyproject.toml` + `uv.lock` (`uv lock`, установка `uv sync --locked`; для pip-пользователей `uv export --format requirements.txt > requirements.txt` – https://docs.astral.sh/uv/concepts/projects/sync/), `.python-version`. Только open-source зависимости (п. 10.3 Положения).
5. Данные не коммитить (п. 9.2 Положения): `make data` скачивает по URL и проверяет SHA-256 (`shasum -a 256`), в README – дата скачивания и ссылки для цитирования из `data_description.txt`.
6. Тесты метрик (pytest, < 1 мин): ARI/NMI идентичных разбиений = 1; Жаккар на известных множествах; Шоррокс: единичная матрица → 0, равномерная → 1; MQ/AVI/AVU на планированном SBM (`nx.stochastic_block_model`) – истинное разбиение даёт максимум; взаимный kNN – симметричен, без петель; Leiden на SBM восстанавливает разбиение (ARI > 0.9); η²_H на двух одинаковых группах ≈ 0.
7. Таблица времени прогона: каждая стадия пишет `runs/timing.csv` (стадия, wall time, пик памяти, железо); в отчёте – таблица «стадия → минуты».
8. `METHODOLOGY.md` с принятыми и отвергнутыми решениями (SigClust, Louvain, картограмма, TILES – и почему), `CHANGELOG` не нужен, DVC не нужен.
9. Один `notebooks/00_report.ipynb` или `make report`, который собирает все рисунки из `results/` – жюри запускает одной командой.

---

## 8. Конвейеры под 2 суток (оценка трудозатрат)

### Минимальный (≈ 19–21 ч чистой работы, 2 человека → укладывается в 1,5 суток)
| # | Этап | Содержание | Часы |
|---|------|------------|------|
| 1 | Данные и признаки | загрузка + SHA-256; матрица МО×(6 категорий × {лог-уровень, доля в «Все категории», г/г темп, сезонная амплитуда}); сшивка с `market_access`, координатами центров, типом МО | 2 |
| 2 | Графы | 2a взаимный kNN с локальным σ (k=10–15) + 2b SNF из 3 видов; таблица сетевых метрик (п. 2) | 2 |
| 3 | Кластеризация | KMeans, Ward, Spectral(precomputed), Leiden; K=4…12; подбор γ под K; полный набор ICVI (SW, CH, S_Dbw, AVI, AVU, MQ) + модулярность | 3,5 |
| 4 | Устойчивость/значимость | консенсус ΔAUC по K; бутстрэп-Жаккар по кластерам (B=100 для kNN, 30 для SNF); z-score модулярности и SW против нулевых моделей | 2,5 |
| 5 | Динамика | temporal Leiden (ω ∈ {0.5,1,2}) + per-slice Leiden с выравниванием; события Greene (cdlib), матрица переходов, Шоррокс, run-length фильтр | 3 |
| 6 | Интерпретация | профили Миркина, η², дерево-суррогат, V Крамера/NMI с регионом/типом/размером; имена кластеров | 2 |
| 7 | Визуализация | mapshaper → TopoJSON < 5 МБ; Plotly-карта с переключением месяцев; Sankey; тепловая карта профилей; GitHub Pages | 3 |
| 8 | Воспроизводимость и отчёт | Makefile, YAML, uv.lock, тесты, timing, METHODOLOGY.md, краткий отчёт | 2,5 |

### Расширенный (+ 10–13 ч, если команда 3–4 человека или остаётся второй день)
| Добавка | Часы |
|---|---|
| 2c корреляции Ледуа–Вольфа и 2d DTW как альтернативные правила + тепловая карта ARI «правило × правило» и чувствительность ICVI | 3 |
| 3f DMoN на PyG (5 seed, те же K) | 3 |
| 4.4 бутстрэп-вероятности меток по месяцам + карта «уверенность перехода» | 2 |
| 5d SHAP + LightGBM (приложение) | 1 |
| 6.3 «Паспорт МО» | 2 |
| 6.2 MapLibre + PMTiles вместо Plotly geo | 2 |

### Что сознательно не делаем (и пишем об этом в отчёте)
SigClust (R, HDLSS), Louvain как основной метод, DTW/гравитация как основное правило ребра, картограммы, TILES/dynetx/tnetwork, кастомные GNN помимо DMoN, DVC.

---

## Список источников (первоисточники и документация)
- Monti et al. 2003 – https://mlanthology.org/mlj/2003/monti2003mlj-consensus/
- Lancichinetti & Fortunato 2012 – https://doi.org/10.1038/srep00336
- Ben-Hur, Elisseeff, Guyon 2002 (PSB) – см. Bioconductor `clusterStab` manual: https://bioc.r-universe.dev/clusterStab/doc/manual.html
- Lange et al. 2004 – https://mlanthology.org/neco/2004/lange2004neco-stabilitybased/
- Hennig 2007 / fpc::clusterboot – https://search.r-project.org/CRAN/refmans/fpc/html/clusterboot.html
- Liu et al. 2008 SigClust (R) – https://cran.r-project.org/web/packages/sigclust/
- Tibshirani, Walther, Hastie 2001 – https://ideas.repec.org/a/bla/jorssb/v63y2001i2p411-423.html ; gap-stat – https://github.com/milesgranger/gap_statistic
- Guimerà, Sales-Pardo, Amaral 2004 – https://arxiv.org/abs/cond-mat/0403660
- Reichardt & Bornholdt 2006 (resolution γ) – https://arxiv.org/abs/cond-mat/0603718
- Zelnik-Manor & Perona 2004 – https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering
- von Luxburg 2007 – https://arxiv.org/abs/0711.0189
- SNF / snfpy – https://github.com/rmarkello/snfpy ; https://snfpy.readthedocs.io/en/latest/api.html
- Ledoit & Wolf 2004 / sklearn – https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
- dtaidistance – https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html ; tslearn – https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.cdist_dtw.html
- Chunaev 2020 – https://arxiv.org/abs/1912.09816 ; Bothorel et al. 2015 – https://arxiv.org/abs/1501.01676
- Traag, Waltman, van Eck 2019 – https://arxiv.org/abs/1810.08473 ; leidenalg – https://leidenalg.readthedocs.io/en/stable/reference.html , https://leidenalg.readthedocs.io/en/stable/multiplex.html
- Mucha et al. 2010 – https://arxiv.org/abs/0911.1824
- Tsitsulin et al. 2023 DMoN – https://www.jmlr.org/papers/v24/20-998.html ; PyG – https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.DMoNPooling.html
- Biswas & Biswas 2017 (AVI/AVU) – https://doi.org/10.1016/j.eswa.2016.11.011 ; Mancoridis et al. 1998 (MQ) – https://researchdiscovery.drexel.edu/esploro/outputs/conferenceProceeding/Using-automatic-clustering-to-produce-high-level/991019167445804721 ; s-dbw – https://pypi.org/project/s-dbw/
- Greene, Doyle, Cunningham 2010 – https://research.google/pubs/tracking-the-evolution-of-communities-in-dynamic-social-networks/ ; CDlib events – https://cdlib.readthedocs.io/en/latest/reference/events.html
- Shorrocks 1978 – https://www.econometricsociety.org/publications/econometrica/1978/09/01/measurement-mobility
- Tomczak & Tomczak 2014 – https://www.wbc.poznan.pl/dlibra/publication/413565
- Mirkin 2005 – https://www.kdnuggets.com/news/2005/n10/25i.html
- SciPy association – https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html ; linear_sum_assignment – https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
- shap.TreeExplainer – https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html ; export_text – https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html
- Границы МО СберИндекс – https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities ; статья на Хабре – https://habr.com/ru/articles/849682/
- mapshaper – https://mapshaper.org/docs/reference.html , https://mapshaper.org/docs/guides/geojson-for-web-maps.html ; antimeridian – https://antimeridian.readthedocs.io ; topojson (py) – https://mattijn.github.io/topojson/
- Plotly geo – https://plotly.com/python/reference/layout/geo/ ; Sankey – https://plotly.com/python/sankey-diagram/ ; hexbin – https://plotly.com/python/hexbin-mapbox/
- d3-geo conic – https://d3js.org/d3-geo/conic ; ESRI:102025 – https://epsg.io/102025
- PMTiles/MapLibre – https://til.simonwillison.net/gis/pmtiles ; kepler.gl Jupyter – https://docs.kepler.gl/docs/keplergl-jupyter ; pydeck – https://deckgl.readthedocs.io/en/latest/
- Если быть точным – https://tochno.st/ ; FT Visual Vocabulary – https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/README.md
- uv – https://docs.astral.sh/uv/concepts/projects/sync/
