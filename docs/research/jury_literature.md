# Литература жюри: что читать, на что ссылаться, какие определения ждёт жюри

Трек «Кластеризация» конкурса СберИндекса. Жюри: Д. Цыплакова, С. Шалилех (Лаборатория СберИндекс), А. Савченко (Sber AI Lab), Ф. Алескеров, С. Кузнецов (ВШЭ).
Дата сверки источников: 2026-10-07. Все DOI проверены через Crossref/OpenAlex; цитаты формул – из первоисточников, если не оговорено иное. Код из KEFRiN/CANUS не копировался, использованы только формулы и описания.

Условные обозначения: $N$ – число узлов (МО), $V$ – число признаков, $K$ – число кластеров, $Y=(y_{iv})$ – матрица признаков, $P=(p_{ij})$ (или $A$) – матрица связей/сходств, $S_k$ – кластер, $n_k=|S_k|$, $e_{kl}$ – суммарный вес рёбер между $S_k$ и $S_l$, $\mathrm{cut}_k=\sum_{l\ne k}e_{kl}$, $m$ – суммарный вес рёбер.

---

## 1. Shalileh, Antonov, Tsyplakova – ICVI для атрибутированных сетей (ESWA 2026) + статья-предшественник в «Докладах РАН» (2025)

### Источники
- Shalileh S., Antonov E., Tsyplakova D. Internal cluster validity indices for attributed networks: A controlled comparative study // Expert Systems with Applications. 2026 (онлайн; номер датирован январём 2027). Vol. 333. Art. 133912. DOI: 10.1016/j.eswa.2026.133912. Open access (по данным OpenAlex/Unpaywall). 51 ссылка в списке литературы.
- Shalileh S., Antonov E. A., Tsyplakova D. A. Cluster Validity across Attribute and Network Spaces: Empirical Benchmarks for Attributed Networks Clustering // Doklady Mathematics. 2025. Vol. 112, No. 3. P. 553–564. DOI: 10.1134/S1064562425700589. Русская версия: Доклады РАН. Математика, информатика, процессы управления. 2025. Т. 527. С. 400–414 (Math-Net: mi.mathnet.ru/danma697). Аффилиация авторов на Math-Net – Сбербанк, Москва.

### Что удалось получить
Полные аннотации обеих статей (OpenAlex) и список литературы «Докладов». **Полный текст не получен**: ScienceDirect (403/JS-челлендж), Springer PDF (JS-челлендж), Math-Net (PDF недоступен), ResearchGate (403). Поэтому формулы индексов в разделе 11 приведены по первоисточникам индексов, а не в нотации авторов; где нотация могла отличаться – оговорено.

### Содержание (по аннотациям)
**ESWA 2026.** Качество кластеризации в атрибутированной сети оценивается с двух сторон: сходство атрибутов узлов (attribute space) и связность (network space). Авторы сравнивают *четыре классических индекса в пространстве признаков* и *четыре сетевых критерия*, плюс вводят два новых индекса в пространстве признаков: **SCR (Separation–Cohesion Ratio)** и **KDSE (KNN Density Separation Entropy)** (определений в открытом доступе не нашли). Построен синтетический генератор атрибутированных сетей; проведён one-factor-at-a-time анализ чувствительности по: числу узлов, числу кластеров, числу признаков, внутрикластерной дисперсии, дисбалансу размеров кластеров, вероятностям связей внутри/между кластерами. Принципиально: индексы считаются **на истинных разбиениях (ground truth) и на случайных назначениях**, чтобы отделить поведение самого индекса от ошибок алгоритма. Вывод: индексы существенно различаются по стабильности, дискриминирующей способности и интерпретации в разных режимах; внутренняя валидация в атрибутированных сетях контекстно-зависима и **не должна опираться на один индекс – нужна небольшая взаимодополняющая панель индексов**.

**Doklady 2025** (явно перечислены индексы): признаки – Silhouette Width (SW), Calinski–Harabasz (CH), Davies–Bouldin (DBI), $S_{Dbw}$; сеть – Average Isolability (AVI), Average Unifiability (AVU), ANUI. Генератор: гауссовы «blobs» для признаков + стохастическая блочная модель (SBM) для графа с общими размерами кластеров. Выводы (цитируется по аннотации):
- SW стабилен и «насыщается», как только признаков достаточно;
- CH сильно растёт с объёмом выборки $N$ – авторы предлагают **отчитывать CH/N**;
- DBI и $S_{Dbw}$ отделяют истинное разбиение от случайного, но их **случайные базовые уровни зависят от $K$** – нужна **нормировка на базовый уровень** (baseline normalization);
- в сетевом пространстве **AVI растёт с ассортативностью и падает примерно как $1/K$**; **AVU падает с ростом $K$ к «полу»**; ANUI повторяет эти тренды;
- все индексы приближаются к случайным базовым уровням по мере роста смешивания/перекрытия; доверительные интервалы сужаются с ростом $N$ и числа информативных признаков.
Список литературы «Докладов» содержит: Arbelaitz et al. (extensive comparative study of CVIs), Biswas & Biswas 2017, Bothorel et al. 2015, Caliński–Harabasz 1974, Chunaev 2020, Davies–Bouldin 1979, Fortunato–Barthélemy 2007 (resolution limit), Fortunato–Hric 2016, Halkidi et al. 2001 (validity measures), Lancichinetti–Fortunato–Radicchi (LFR), Li et al. 2008 (modularity density), Newman 2006 (modularity), Newman–Girvan 2004, Rousseeuw 1987, Tsitsulin et al. 2023 (DMoN), Vendramin et al. 2010 (relative validity criteria), Yang–Leskovec 2015 (ground-truth communities).

### Как использовать в нашей работе
1. Отчитывать именно эту панель: SW, CH (и CH/N), DBI, $S_{Dbw}$ по признакам; AVI, AVU, ANUI (+ модулярность) по графу – и **для каждого $K$ приводить случайный базовый уровень** (перестановочный тест: та же разметка размеров, случайное назначение узлов), а не только сырые значения.
2. Не выбирать $K$ по одному индексу; показать таблицу «индекс × $K$» с нормировкой на базовый уровень ($\text{index} - \text{baseline}$ или $\text{index}/\text{baseline}$) и доверительными интервалами (bootstrap по узлам или по инициализациям).
3. Явно ссылаться на обе статьи как на методологическую основу раздела «валидация» – это работы двух членов жюри.

---

## 2. «Profiling Consumption Using Attributed Network Clustering» (Complex Networks & Their Applications XIV, 2026)

### Источник
Shalileh S., Antonov E., Tsyplakova D. Profiling Consumption Using Attributed Network Clustering // Complex Networks & Their Applications XIV. Studies in Computational Intelligence. Cham: Springer Nature Switzerland, 2026. P. 77–88. DOI: 10.1007/978-3-032-16723-1_7. Опубликовано онлайн 25.02.2026. ISBN 978-3-032-16722-4 / 978-3-032-16723-1. Аффилиация: SberBank of Russia, Moscow.

### Что удалось получить
Аннотация (через поисковую выдачу) и полный список литературы (Crossref). **Полный текст не получен** (платный доступ, Springer редиректит на авторизацию). Поэтому точное правило ребра (kNN / порог / полный взвешенный граф), способ выбора $K$ и формат интерпретации – **не верифицированы**.

### Содержание (по аннотации)
- Данные: проприетарные банковские транзакции, **11 городов Дальнего Востока**, май–июль 2025, **по 5 000 наблюдений на город**.
- Для каждого города строится **атрибутированная сеть**: **рёбра отражают сходство расходов по категориям** (**42 категории, онлайн/офлайн**); **атрибуты узлов** – состав расходов, временная активность (утро/день/вечер/ночь), индексы мобильности и лояльности, возраст.
- Результат: связные, специфичные для города потребительские сегменты, профили которых согласуются с социально-демографическими и поведенческими паттернами; выявлены устойчивые межгородские различия в необходимых vs дискреционных расходах, принятии онлайна и мобильности.

### Что видно по списку литературы (инструментарий авторов)
Методы: KEFRiN (Shalileh & Mirkin, Entropy 2022), data-recovery подход (Mirkin & Shalileh, J. Classification 2022), least-squares извлечение сообществ по данным сходства (Shalileh & Mirkin, PLoS ONE), Louvain (Blondel et al. 2008), EVA / «homogeneous communities in labeled networks» (Citraro & Rossetti 2020), DMoN (Tsitsulin et al. 2023), контрастивные методы для атрибутированных графов (Zheng et al. 2024; Yang et al. 2024; van den Oord et al. 2018), WSNMF (Berahmand et al. 2024), CESNA (Yang, McAuley, Leskovec 2013), SBM/аннотированные сети (Karrer–Newman 2011; Newman–Clauset 2016; Stanley et al. 2019). **Валидация: Silhouette (Rousseeuw 1987), Calinski–Harabasz (1974), Biswas & Biswas 2017 (AVI/AVU/ANUI)**. Прикладной контекст: Galdeman et al. 2021 (city consumption profile), Leo et al. 2018, Di Clemente et al. 2018 (lifestyles из транзакций), Shumovskaia et al. 2021 (GNN на транзакциях), Dev & Hamooni 2020, Bahrami et al. 2023.

### Как использовать в нашей работе
Воспроизвести «архитектуру» эталонного решения на уровне МО: узел = МО; ребро = сходство векторов долей расходов по категориям (косинус; взвешенный граф или kNN-разрежение – выбор обосновать и проверить чувствительность); атрибуты = состав трат + временные/онлайн-доли + динамические признаки. Базовые методы: KEFRiN (три метрики), Louvain/Leiden на графе сходства, k-means только по признакам – как «однопространственные» бейзлайны; валидация – панель из п. 1. Интерпретация сегментов – через отклонения от среднего по совокупности (п. 6), как в «city consumption profile».

---

## 3. KEFRiN и CANUS (Shalileh & Mirkin; Shalileh)

### 3.1 KEFRiN – источник и лицензия
- Shalileh S., Mirkin B. Community Partitioning over Feature-Rich Networks Using an Extended K-Means Method // Entropy. 2022. Vol. 24, No. 5. Art. 626. DOI: 10.3390/e24050626. CC BY 4.0. Полный текст прочитан (PDF из репозитория).
- Репозиторий: https://github.com/Sorooshi/KEFRiN – **лицензия MIT** (README: «This project is licensed under the MIT License»). Файлы: `kefrin.py`, `processing_tools.py`, `demo.py`, `reproduce_table9.py`, `KEFRIN.pdf`, данные.
- Предшественники: Shalileh & Mirkin, Summable and nonsummable data-driven models for community detection in feature-rich networks // Social Network Analysis and Mining. 2021. Vol. 11. Art. 67. DOI: 10.1007/s13278-021-00774-8; Mirkin & Shalileh, Community Detection in Feature-Rich Networks Using Data Recovery Approach // Journal of Classification. 2022. Vol. 39, No. 3. P. 432–462. DOI: 10.1007/s00357-022-09416-w; Shalileh & Mirkin, Least-squares community extraction in feature-rich networks using similarity data // PLoS ONE. 2021. Vol. 16. e0254377. DOI: 10.1371/journal.pone.0254377 (open access, прочитан частично).

### 3.2 Математическая постановка KEFRiN (Entropy 2022, уравнения (1)–(7))
Модель восстановления данных (data recovery). Признаки:
$$y_{iv}=\sum_{k=1}^{K} c_{kv}\,s_{ik}+f_{iv},\qquad i\in I,\ v\in V \tag{1}$$
где $s_{ik}\in\{0,1\}$ – индикатор принадлежности, $c_k=(c_{kv})$ – центр кластера в пространстве признаков.

**Суммируемость vs несуммируемость** (summability / nonsummability). В режиме суммируемости все веса связей считаются измеренными в одной шкале и сообщество описывается одной интенсивностью $\lambda_k$:
$$p_{ij}=\sum_{k=1}^{K}\lambda_k s_{ik}s_{jk}+e_{ij}. \tag{2}$$
В режиме **несуммируемости** связи каждого узла считаются измеренными в своей шкале (пример авторов: сайты-учебники vs интернет-магазины различаются и по числу посетителей, и по времени), и вводятся **столбцово-зависимые интенсивности** $\lambda_{kj}$:
$$p_{ij}=\sum_{k=1}^{K}\lambda_{kj}\,s_{ik}+e_{ij}. \tag{4}$$
Тем самым каждое сообщество получает «двойной центр»: $c_k\in\mathbb{R}^V$ (признаки) и $\lambda_k=(\lambda_{kj})\in\mathbb{R}^N$ (строка в пространстве узлов).

**Критерий наименьших квадратов:**
$$F(s_{ik},c_{kv},\lambda_{kj})=\rho\sum_{i,v}\Big(y_{iv}-\sum_k c_{kv}s_{ik}\Big)^2+\xi\sum_{i,j}\Big(p_{ij}-\sum_k\lambda_{kj}s_{ik}\Big)^2, \tag{5}$$
в матричной форме $Y\approx SC$, $P\approx S\Lambda$, $F=\rho\,\mathrm{Tr}[(Y-SC)^T(Y-SC)]+\xi\,\mathrm{Tr}[(P-S\Lambda)^T(P-S\Lambda)]$. $\rho,\xi$ – «экспертные» константы баланса источников данных; **в статье взяты $\rho=\xi=1$**; автоматизация их выбора названа направлением будущей работы.

Поскольку $s_{ik}=1$ ровно для одного $k$:
$$F(S,c,\lambda)=\sum_{k=1}^{K}\sum_{i\in S_k}\big[\rho\,d_e(y_i,c_k)+\xi\,d_e(p_i,\lambda_k)\big], \tag{6}$$
$d_e(y_i,c_k)=\sum_v (y_{iv}-c_{kv})^2$, $d_e(p_i,\lambda_k)=\sum_j (p_{ij}-\lambda_{kj})^2$, $p_i$ – $i$-я строка $P$.

**Шаги (чередующаяся минимизация, аналог batch K-means):**
1. Стандартизация признаков и сети (ниже).
2. Инициализация: задать $K>1$; начальные центры $(c_k,\lambda_k)$ – вариант K-Means++ в сочетании с MaxMin: первый узел случайно; далее для каждого оставшегося $i$ считается $f(i)=\sum_{k'} d_e(i,k')$, где $d_e(i,k)=\rho d_e(y_i,c_k)+\xi d_e(p_i,\lambda_k)$, и следующим центром берётся узел с максимальным $f(i)$ (в репозитории – вероятностный K-Means++ по $d^2$).
3. Правило минимального расстояния: $i\mapsto\arg\min_k\big[\rho d(y_i,c_k)+\xi d(p_i,\lambda_k)\big]$.
4. Остановка, если разбиение не изменилось (в репозитории – по изменению inertia меньше tolerance; `n_init=10` запусков, лучший по inertia).
5. Обновление центров – внутрикластерные средние:
$$c_{kv}=\frac{\sum_{i\in S_k}y_{iv}}{|S_k|},\qquad \lambda_{kj}=\frac{\sum_{i\in S_k}p_{ij}}{|S_k|}. \tag{7}$$

**Три версии:** KEFRiNe (квадрат евклидова), KEFRiNm (манхэттенское $d_m(f,g)=\sum_t|f_t-g_t|$, уравнение (8); центры по-прежнему средние), KEFRiNc (косинусное $d_c=1-\cos(f,g)$, уравнение (9); все векторы $y_i,p_i,c_k,\lambda_k$ нормируются, после усреднения центры **повторно нормируются** – это нарушает оптимальность центров по критерию и может влиять на сходимость; $d_c$ = половина квадрата евклидова расстояния между нормированными векторами).

**Предобработка (разд. 4.1):** признаки – (Z) z-scoring или (R) центрирование и деление на размах; сеть – (M) «модулярностная» трансформация $p_{ij}\leftarrow p_{ij}-\dfrac{p_{i+}p_{+j}}{p_{++}}$ (вычитание случайных взаимодействий) или (S) сдвиг шкалы $p_{ij}\leftarrow p_{ij}-\pi$, $\pi=\sum_{ij}p_{ij}/N^2$. Рекомендация авторов: **Z для признаков, M для сети**; при категориальных признаках – S вместо M. В репозитории дополнительно есть опция Laplacian-трансформации сети.

**Выбор $K$:** в статье $K$ задаётся заранее (на реальных данных = числу классов); определение «правильного» $K$ – явно отнесено к будущей работе (п. (e) заключения). Предшествующие алгоритмы авторов (SEFNAC, ICESi) извлекают кластеры по одному и находят $K$ автоматически: в PLoS ONE 2021 правило остановки – «вклад текущего кластера в разброс данных меньше, скажем, 5% от $Q(R,P)$, при этом накопленный вклад $\ge 50\%$».

**Оценка:** ARI (и NMI) против ground truth; конкуренты CESNA, SIAN, SEANAC, DMoN. Выводы: KEFRiNc лучше на реальных многомерных данных, KEFRiNm – на большинстве синтетических; все версии на 1–3 порядка быстрее конкурентов; «универсального победителя нет».

### 3.3 CANUS – источник, постановка, лицензия
- Статья: Shalileh S. A Filtered Gradient Descent Clustering Method to Recover Communities in Attributed Networks // IEEE Access. 2025. Vol. 13. P. 169698–169719. DOI: 10.1109/ACCESS.2025.3614989. **Полный текст не читался** (IEEE Xplore); описание ниже – из docstring и README репозитория, где оно изложено автором.
- Предшественники: Shalileh S., Mirkin B. Community Detection in Feature-Rich Networks Using Gradient Descent Approach // Complex Networks & Their Applications XII (2023). Studies in Computational Intelligence. Springer, 2024. P. 185–196. DOI: 10.1007/978-3-031-53499-7_15; Shalileh S. Gradient Descent Clustering with Regularization to Recover Communities in Transformed Attributed Networks // ASONAM 2024 proc., LNCS. Springer, 2025. P. 137–148. DOI: 10.1007/978-3-031-78538-2_12 (трансформация пространства softmax-scaled-dot-product + регуляризация).
- Репозиторий: https://github.com/Sorooshi/CANUS – файлы `canus.py` (PyTorch), `canus_jax.py`, ноутбук, датасеты. **Файла LICENSE в листинге репозитория не обнаружено** (на дату проверки) – по умолчанию это «all rights reserved»; использовать только формулы/описание, код не копировать.

**Постановка (по описанию автора в репозитории):** жёсткая кластеризация по атрибутам $X\in\mathbb{R}^{N\times D}$ и строкам смежности $A\in\mathbb{R}^{N\times N}$; центры $C\in\mathbb{R}^{K\times D}$ (атрибуты) и $\Lambda\in\mathbb{R}^{K\times N}$ (узловое пространство); совместное расстояние (уравнение (1) статьи)
$$D_{ik}=\rho\, f(x_i,c_k)+\zeta\, h(a_i,\lambda_k),$$
назначение $i\mapsto\arg\min_k D_{ik}$; функция потерь $\sum_k\sum_{i\in S_k}D_{ik}$ минимизируется **градиентным спуском по центрам** (full-batch Adam или «vanilla»), а не аналитическим усреднением. По умолчанию $f,h$ – косинусные расстояния (есть Minkowski с показателем $p$, Canberra); $\rho=\zeta=1$; инициализация K-Means++ в совместном пространстве; `lr=1e-3`, `epochs=300`, early stopping опционален. **Фильтрация (CANUSf):** после первого назначения для каждого кластера считаются нормы градиентов по отдельным объектам, оцениваются $\mu_k,\sigma_k$ (bootstrap, 50 выборок), и в обновлении центров учитываются только объекты с нормой градиента в полосе $[\mu_k-\tau\sigma_k,\ \mu_k+\tau\sigma_k]$, $\tau=1$ по умолчанию; по статье полосы считаются один раз (Algorithm 2) и далее переиспользуются. Интуиция: отсечь выбросы и «слишком лёгкие» объекты, стабилизируя центры.

### Как использовать в нашей работе
Реализовать KEFRiN самостоятельно по (5)–(7) (три метрики, Z+M предобработка, $\rho=\xi=1$ как дефолт и сетка по $\rho/\xi$ как анализ чувствительности); MIT-лицензия позволяет и импорт, но собственная реализация по формулам безопаснее с точки зрения условий конкурса. CANUS – упомянуть как градиентную альтернативу (тот же критерий, другой оптимизатор + фильтрация); при желании реализовать «ванильный» вариант по описанию выше, не заимствуя код. Для динамики: применять KEFRiN к каждому срезу с «тёплым стартом» центров из предыдущего месяца – это естественное расширение чередующейся минимизации (наш вклад, в статьях не описан).

---

## 4. Biswas & Biswas (2017): AVI, AVU, ANUI – определения и аналитика вырожденных случаев

### Источник
Biswas A., Biswas B. Defining quality metrics for graph clustering evaluation // Expert Systems with Applications. 2017. Vol. 71. P. 1–17. DOI: 10.1016/j.eswa.2016.11.011. **Первоисточник недоступен** (платный; ResearchGate/DSpace IIT BHU не отдали текст). Авторы определяют три метрики (AVI, AVU, ANUI), удовлетворяющие шести «свойствам качества» (список свойств не получен).

Определения ниже взяты из их воспроизведения в: Howie J., Srinivasan V., Thomo A. Scaling Up Structural Clustering to Large Probabilistic Graphs Using Lyapunov Central Limit Theorem // PVLDB. 2023. Vol. 16, No. 11. P. 3165–3177. DOI: 10.14778/3611479.3611516 (уравнения (36)–(38); там $p(u,v)$ – вероятность ребра, для детерминированного графа $p(u,v)=a_{uv}$). Они согласуются с описанием поведения индексов в аннотации «Докладов» (п. 1).

### Определения
Изолируемость (isolability) кластера – доля связей его узлов, остающихся внутри:
$$I(C_k)=\frac{\sum_{u\in C_k,\ v\in C_k}a_{uv}}{\sum_{u\in C_k,\ v\in C_k}a_{uv}+\sum_{u\in C_k,\ v\notin C_k}a_{uv}}=\frac{\mathrm{in}_k}{\mathrm{in}_k+\mathrm{cut}_k},$$
где $\mathrm{in}_k=\sum_{u,v\in C_k}a_{uv}$ – сумма по **упорядоченным** парам (т.е. каждое внутреннее ребро учтено дважды: $\mathrm{in}_k=2e_{kk}$), если суммировать по матрице смежности напрямую.
Унифицируемость (unifiability) пары кластеров:
$$U(C_k,C_l)=\frac{e_{kl}}{\mathrm{cut}_k+\mathrm{cut}_l-e_{kl}}=\frac{e_{kl}}{e_{kl}+\sum_{r\notin\{k,l\}}(e_{kr}+e_{lr})}.$$
Усреднения: $\mathrm{AVI}=\frac1K\sum_k I(C_k)$; $\mathrm{AVU}$ – среднее $U$ «по кластерам» (точная конвенция усреднения – по неупорядоченным парам или как среднее по кластерам сумм по партнёрам – **в доступных источниках не зафиксирована**, см. ниже); комбинированный индекс в записи Howie et al. (38):
$$\mathrm{ANUI}=\frac{\mathrm{AVI}}{1+\mathrm{AVI}\cdot\mathrm{AVU}}$$
(требует сверки с оригиналом; содержательно – «отношение» изолируемости к унифицируемости: ANUI растёт с AVI и падает с AVU). Направления: AVI ↑ лучше, AVU ↓ лучше, ANUI ↑ лучше.

### Проверка утверждений (аналитически)
**(а) AVU вырожден при $K\le 3$.**
- $K=2$: $\mathrm{cut}_1=\mathrm{cut}_2=e_{12}$, поэтому $U(C_1,C_2)=\dfrac{e_{12}}{e_{12}+e_{12}-e_{12}}=1$ для **любого** разбиения с $e_{12}>0$. Значит $\mathrm{AVU}\equiv 1$ – утверждение верно.
- $K=3$: $\mathrm{cut}_1=e_{12}+e_{13}$, $\mathrm{cut}_2=e_{12}+e_{23}$, откуда $U(C_1,C_2)=\dfrac{e_{12}}{e_{12}+e_{13}+e_{23}}=\dfrac{e_{12}}{E_{\text{между}}}$, аналогично для других пар, и $\sum_{k<l}U(C_k,C_l)=1$ тождественно. Тогда: при усреднении по трём неупорядоченным парам $\mathrm{AVU}=1/3$; при усреднении «по кластерам» вида $\mathrm{AVU}=\frac1K\sum_k\sum_{l\ne k}U(C_k,C_l)=\frac{2}{K}\sum_{k<l}U=\frac23$. То есть значение **2/3 получается при второй конвенции** (сумма по партнёрам каждого кластера, усреднённая по $K$ кластерам); при первой – 1/3. В обоих случаях при $K=3$ AVU – **константа, не зависящая от разбиения**, т.е. индекс неинформативен при $K\le3$. Утверждение верно с точностью до конвенции усреднения.
- $K\ge4$: $U(C_k,C_l)=e_{kl}/\big(e_{kl}+\sum_{r\notin\{k,l\}}(e_{kr}+e_{lr})\big)$ уже зависит от распределения межкластерных рёбер; знаменатель включает связи с $K-2$ «третьими» кластерами, поэтому с ростом $K$ типичные $U$ убывают – это и есть «AVU падает с $K$ к полу» из «Докладов».

**(б) AVI случайного разбиения $\approx 1/K$.** При случайном назначении узлов в кластеры размеров $n_k$ ожидаемая доля связей узла, попавших внутрь его кластера, $\approx (n_k-1)/(N-1)\approx n_k/N$ (независимо от структуры графа, в т.ч. для взвешенного графа сходств). Числитель и знаменатель $I(C_k)$ – суммы по одним и тем же узлам, поэтому $I(C_k)\approx n_k/N$ и
$$\mathrm{AVI}_{\text{rand}}\approx\frac1K\sum_k\frac{n_k}{N}=\frac1K$$
**при любом распределении размеров**. Если бы внутренние рёбра считались один раз ($e_{kk}$ вместо $2e_{kk}$), получилось бы $I\approx\frac{n_k/N}{2-n_k/N}\approx\frac{1}{2K-1}$. Эмпирическое «$\approx 1/K$» в «Докладах» подтверждает конвенцию суммирования по матрице смежности (упорядоченные пары). Для SBM на истинном разбиении $I(C_k)\approx\dfrac{p_{in}n_k}{p_{in}n_k+p_{out}(N-n_k)}$ – растёт с ассортативностью $p_{in}/p_{out}$, что также согласуется с «Докладами».

### Как использовать в нашей работе
Отчитывать AVI вместе с базой $1/K$ (например, $\mathrm{AVI}\cdot K$ или $\mathrm{AVI}-1/K$); AVU/ANUI – только для $K\ge4$ и с перестановочным базовым уровнем; явно указать, по какой конвенции усредняется AVU и как считаются внутренние рёбра; на взвешенном графе сходств использовать веса $a_{uv}$. Сослаться на Biswas & Biswas 2017 как на первоисточник и на Shalileh et al. 2025/2026 как на обоснование нормировок.

---

## 5. MQ: модулярность Ньюмана–Гирвана или Modularization Quality?

**Вывод по контексту работ Шалилеха и соавторов.** В перечне индексов «Докладов» (SW, CH, DBI, $S_{Dbw}$, AVI, AVU, ANUI) обозначения MQ нет; в списках литературы обеих статей и главы цитируются **Newman & Girvan 2004 (модулярность $Q$), Newman 2006, Li et al. 2008 (modularity density $D$), Fortunato & Barthélemy 2007 (resolution limit), Blondel et al. 2008 (Louvain)** и **не цитируется** Mancoridis et al. Четвёртый «сетевой критерий» ESWA-статьи в открытом доступе не раскрыт; наиболее вероятно – модулярность $Q$ (или $D$). Итак: если в наших материалах «MQ» означает сетевой критерий жюри – это **модулярность Ньюмана–Гирвана** (писать $Q$); Modularization Quality из software clustering – другой индекс, его стоит упомянуть только как «не путать».

**Модулярность Ньюмана–Гирвана** (Newman M. E. J., Girvan M. Finding and evaluating community structure in networks // Phys. Rev. E. 2004. Vol. 69. 026113. DOI: 10.1103/PhysRevE.69.026113):
$$Q=\frac{1}{2m}\sum_{i,j}\Big(a_{ij}-\frac{k_ik_j}{2m}\Big)\delta(c_i,c_j)=\sum_{k=1}^{K}\Big[\frac{e_{kk}}{m}-\Big(\frac{d_k}{2m}\Big)^2\Big],$$
$k_i=\sum_j a_{ij}$, $d_k=\sum_{i\in C_k}k_i$, $m=\frac12\sum_{ij}a_{ij}$; для взвешенных графов – те же формулы с весами. $Q\in[-1/2,1]$, ↑ лучше; известны предел разрешения (Fortunato & Barthélemy 2007) и обобщение с параметром разрешения $\gamma$ (Reichardt & Bornholdt 2006, Phys. Rev. E 74, 016110, DOI: 10.1103/PhysRevE.74.016110). Для полного графа сходств $Q$ сильно зависит от разрежения (порог/kNN) – это надо фиксировать в протоколе.

**Modularity density** (Li Z., Zhang S., Wang R.-S., Zhang X.-S., Chen L. Quantitative function for community detection // Phys. Rev. E. 2008. Vol. 77. 036109. DOI: 10.1103/PhysRevE.77.036109):
$$D=\sum_{k=1}^{K}\frac{L(V_k,V_k)-L(V_k,\bar V_k)}{|V_k|},\qquad L(V_1,V_2)=\sum_{i\in V_1,\ j\in V_2}a_{ij},$$
↑ лучше; смягчает предел разрешения.

**Modularization Quality (MQ), Mancoridis et al.** (Mancoridis S., Mitchell B. S., Rorres C., Chen Y., Gansner E. R. Using automatic clustering to produce high-level system organizations of source code // Proc. 6th Int. Workshop on Program Comprehension (IWPC'98). IEEE, 1998. P. 45–52. DOI: 10.1109/WPC.1998.693283). Для разбиения графа зависимостей на $k$ модулей с $N_i$ узлами, $\mu_i$ внутренними и $\varepsilon_{ij}$ межмодульными рёбрами:
$$A_i=\frac{\mu_i}{N_i^2},\qquad E_{ij}=\frac{\varepsilon_{ij}}{2N_iN_j},\qquad \mathrm{MQ}=\frac1k\sum_{i=1}^{k}A_i-\frac{1}{k(k-1)/2}\sum_{i<j}E_{ij}\ (k>1),\quad \mathrm{MQ}=A_1\ (k=1),$$
$\mathrm{MQ}\in[-1,1]$, ↑ лучше. Поздняя версия TurboMQ (Mitchell B. S., Mancoridis S. On the automatic modularization of software systems using the Bunch tool // IEEE TSE. 2006. Vol. 32, No. 3. P. 193–208. DOI: 10.1109/TSE.2006.31):
$$\mathrm{MQ}=\sum_{i=1}^{k}CF_i,\qquad CF_i=\frac{2\mu_i}{2\mu_i+\sum_{j\ne i}(\varepsilon_{ij}+\varepsilon_{ji})}\ (\text{0, если }\mu_i=0).$$
Заметим: $CF_i$ структурно совпадает с изолируемостью $I(C_i)$ Biswas (внутреннее/(внутреннее+внешнее)), так что $\mathrm{TurboMQ}=K\cdot\mathrm{AVI}$ при одинаковой конвенции подсчёта рёбер – полезное замечание для раздела «связь индексов».

**Как использовать:** в отчёте писать «модулярность $Q$ (Newman–Girvan)», при необходимости $D$; MQ Mancoridis упомянуть в одну строку как омоним.

---

## 6. Интерпретация кластеров «по Миркину»: вклады в объяснённую дисперсию и относительные отклонения

### Источники
- Mirkin B. Clustering: A Data Recovery Approach. 2nd ed. Boca Raton: CRC Press (Chapman & Hall/CRC), 2012 (1-е изд.: Clustering for Data Mining: A Data Recovery Approach, 2005). Именно эта книга – ссылка [9] в KEFRiN.
- Mirkin B. Core Concepts in Data Analysis: Summarization, Correlation and Visualization. London: Springer, 2011 (гл. о K-means и интерпретации кластеров).

### Формулы
Модель K-means как модель восстановления данных: $y_{iv}=c_{kv}+e_{iv}$ для $i\in S_k$ (признаки предварительно стандартизованы: центрированы по общему среднему $m_v$ и отмасштабированы, Миркин рекомендует деление на размах). Пифагорово разложение разброса данных:
$$T(Y)=\sum_{i,v}y_{iv}^2=\underbrace{\sum_{k=1}^{K}\sum_{v}n_k\,c_{kv}^2}_{B(S,c)\ \text{объяснённая часть}}+\underbrace{\sum_{k}\sum_{i\in S_k}\sum_v (y_{iv}-c_{kv})^2}_{W(S,c)\ \text{критерий K-means}}.$$
Отсюда:
- **вклад кластера $k$** в разброс: $B_k=n_k\sum_v c_{kv}^2$, относительный $B_k/T(Y)$;
- **вклад пары кластер × признак**: $B_{kv}=n_k\,c_{kv}^2$, относительный $B_{kv}/T(Y)$ (и/или в долях от $B_k$); сумма по всем $(k,v)$ даёт долю объяснённой дисперсии $B/T$;
- **относительное отклонение** (в исходной шкале, для таблиц профилей): $\delta_{kv}=\dfrac{\bar y_{kv}-m_v}{m_v}\cdot100\%$, где $\bar y_{kv}$ – среднее признака $v$ в кластере, $m_v$ – общее среднее; признак «характерен» для кластера, если $|\delta_{kv}|$ велико **и** $B_{kv}$ входит в число наибольших;
- для категориальных признаков / долей: индекс Кетле $q_{kv}=\dfrac{p(v\mid k)-p(v)}{p(v)}=\dfrac{p(k,v)}{p(k)p(v)}-1$, а вклад категории $v$ в кластер $k$ при стандартизации «dummy / $\sqrt{p(v)}$» равен $n_k\,(p(v\mid k)-p(v))^2/p(v)$ – слагаемое статистики $\chi^2$ Пирсона (суммарный вклад категориального признака = $\chi^2/N$-подобная величина).

### Как использовать в нашей работе
Для каждого кластера МО выдать: (1) таблицу топ-признаков по $B_{kv}/T$ с указанием $\delta_{kv}$ и знака; (2) тепловую карту «кластер × категория трат» по $\delta_{kv}$; (3) долю объяснённого разброса $B/T$ в признаках и аналог для сетевой части ($\sum_k n_k\|\lambda_k\|^2$ по (7) KEFRiN). Для динамики – те же вклады по месяцам (куб МО × категория × месяц).

---

## 7. Алескеров: пороговое агрегирование и паттерн-анализ

### 7.1 Пороговое агрегирование (threshold rule)
**Источники:** Aleskerov F., Chistyakov V. V., Kalyagin V. The threshold aggregation // Economics Letters. 2010. Vol. 107, No. 2. P. 261–262. DOI: 10.1016/j.econlet.2010.01.041 (полный текст прочитан). Трёхградационный случай: Aleskerov F. T., Yakuba V. I., Yuzbashev D. A. A 'threshold aggregation' of three-graded rankings // Mathematical Social Sciences. 2007. Vol. 53. P. 106–110. Произвольные наборы оценок: Chistyakov V. V., Kalyagin V. A. A model of noncompensatory aggregation with an arbitrary collection of grades // Doklady Mathematics. 2008. Vol. 78. P. 617–620. Приложения: Aleskerov, Chistyakov, Kalyagin. Multiple Criteria Threshold Decision Making Algorithms. SSRN, 2010. DOI: 10.2139/ssrn.1680157 (оценка отделений банков, некомпенсаторное ранжирование студентов).

**Определение.** Альтернатива $x$ оценивается $n$ критериями (агентами) по $m$-балльной шкале $1<2<\dots<m$ (1 – худшая оценка). Пусть $v_j(x)=|\{i: x_i=j\}|$ – число критериев, по которым $x$ получила оценку $j$. Отношение $P$, порождаемое пороговым правилом: при $m=2$ $(x,y)\in P \iff v_1(x)<v_1(y)$; при $m\ge3$
$$(x,y)\in P\iff v_1(x)<v_1(y)\ \ \text{или}\ \ \exists k\in[2,m-1]:\ v_j(x)=v_j(y)\ \forall j<k\ \text{и}\ v_k(x)<v_k(y),$$
т.е. **лексикографическое сравнение векторов $(v_1,\dots,v_{m-1})$: сначала меньше худших оценок, при равенстве – меньше вторых снизу и т.д.** Безразличие $\iff v_j(x)=v_j(y)\ \forall j$. $P$ – слабый порядок. Аксиомы: Pairwise Compensation (анонимность оценок), Pareto Domination, Noncompensatory Threshold («хотя бы одна оценка „плохо“ не компенсируется никаким числом „хорошо“: такая альтернатива ниже любой, оценённой всеми как „средне“»), Contraction; теорема представления: $P$ порождается пороговым правилом $\iff$ существует функция $\varphi$, удовлетворяющая аксиомам (A.1)–(A.3), с $P=\{(x,y):\varphi(x)>\varphi(y)\}$. Простая сумма баллов $\sum_i x_i$ удовлетворяет (A.1),(A.2), но не (A.3). Есть **двойственное правило** (сравнение сверху: больше лучших оценок).

**Применение к сводному рейтингу методов по нескольким индексам.** Пусть методы (или пары «метод, $K$») – альтернативы, индексы (SW, CH/N, DBI, $S_{Dbw}$, AVI, AVU, ANUI, $Q$, стабильность) – критерии. Шаг 1: каждый индекс перевести в $m$-балльную оценку, не зависящую от направления и шкалы: например, $m=3$ – «плохо/средне/хорошо» по положению относительно перестановочного базового уровня и лучшего значения (или по терцилям среди сравниваемых методов). Шаг 2: для каждого метода посчитать $v_1,\dots,v_{m-1}$. Шаг 3: отранжировать лексикографически. Результат – **некомпенсаторный рейтинг**: метод, проваливший хоть один индекс, не «вытянется» за счёт остальных; это уместно именно потому, что ESWA-статья жюри требует панели индексов, а не одного. Дополнительно можно показать устойчивость рейтинга к $m$ и к способу градуировки.

### 7.2 Паттерн-анализ
**Источники:** Алескеров Ф. Т., Белоусова В. Ю., Егорова Л. Г., Миркин Б. Г. Анализ паттернов в статике и динамике, часть 1: обзор литературы и уточнение понятия // Бизнес-информатика. 2013. № 3(25). С. 3–18; часть 2: примеры применения к анализу социально-экономических процессов // Бизнес-информатика. 2013. № 4(26). Aleskerov F., Egorova L., Gokhberg L., Myachin A., Sagieva G. Pattern Analysis in the Study of Science, Education and Innovative Activity in Russian Regions // Procedia Computer Science. 2013. Vol. 17. P. 687–694. DOI: 10.1016/j.procs.2013.05.089. Aleskerov F., Egorova L., Gokhberg L., Myachin A., Sagieva G. A Method of Static and Dynamic Pattern Analysis of Innovative Development of Russian Regions in the Long Run // Springer Proceedings in Mathematics & Statistics. 2014. P. 1–8. DOI: 10.1007/978-3-319-09758-9_1. **Полные тексты не получены** (PDF bijournal – таймаут; Procedia – не найден через alphaXiv); описание ниже – по аннотациям и общеизвестному содержанию метода, что следует оговорить при цитировании.

**Суть.** Паттерн-анализ (по определению авторов) – область анализа данных, связанная с поиском взаимосвязей между объектами, построением классификации объектов и изучением их изменений во времени. Процедура (статика): показатели объектов нормируются; объект представляется профилем (ломаной) в параллельных координатах; **паттерн** – группа объектов с одинаковой/близкой формой профиля (у Мячина – порядково-инвариантное сравнение соседних координат, у Миркина – кластеризация нормированных профилей); объекты распределяются по паттернам. Динамика: та же процедура по каждому периоду с фиксированной системой паттернов, затем анализируются **переходы объектов между паттернами** (матрицы переходов, устойчивые/мобильные объекты, типичные траектории).

**Как использовать.** Это прямой методологический аналог нашей задачи «динамика кластеров МО»: кластеры = паттерны структуры потребления, переходы МО между кластерами по месяцам = динамический паттерн-анализ; матрица переходов + индекс Шоррокса (п. 10) + типичные траектории. Полезно так и назвать раздел и сослаться на эти работы.

---

## 8. Кузнецов: FCA и трикластеризация (OAC-triclustering)

### Источники
- Ignatov D. I., Kuznetsov S. O., Magizov R. A., Zhukov L. E. From Triconcepts to Triclusters // RSFDGrC 2011. LNCS (LNAI) 6743. Springer, 2011. P. 257–264. DOI: 10.1007/978-3-642-21881-1_41 (полный текст прочитан).
- Ignatov D. I., Kuznetsov S. O., Poelmans J., Zhukov L. E. Can triconcepts become triclusters? // International Journal of General Systems. 2013. Vol. 42, No. 6. P. 572–593. DOI: 10.1080/03081079.2013.798899.
- Ignatov D. I., Gnatyshak D. V., Kuznetsov S. O., Mirkin B. G. Triadic Formal Concept Analysis and triclustering: searching for optimal patterns // Machine Learning. 2015. Vol. 101, No. 1–3. P. 271–302. DOI: 10.1007/s10994-015-5487-y (определения «оптимальных» триадических паттернов: формальный триконцепт как максимальный полностью плотный кубоид; релаксации – OAC-трикластеры и трикластеры, оптимальные по МНК; экспериментальное сравнение пяти алгоритмов).
- Ganter B., Wille R. Formal Concept Analysis: Mathematical Foundations. Springer, 1999; Lehmann F., Wille R. A triadic approach to formal concept analysis // ICCS 1995. LNCS 954. P. 32–43.

### Определения (из статьи 2011 г.)
Триадический контекст $\mathbb{K}=(G,M,B,Y)$: объекты $G$, атрибуты $M$, условия $B$, тернарное отношение $Y\subseteq G\times M\times B$; $(g,m,b)\in Y$ – объект $g$ имеет атрибут $m$ при условии $b$. **Триконцепт** $(A_1,A_2,A_3)$ – максимальный кубоид, полностью заполненный «крестиками» (замкнутость по операторам $(\cdot)^{(i)}$). Штрих-операторы 1-множеств: $g'=\{(m,b)\mid (g,m,b)\in Y\}$, $m'=\{(g,b)\mid(g,m,b)\in Y\}$, $b'=\{(g,m)\mid (g,m,b)\in Y\}$. **Box-операторы:**
$$g^{\square}=\{\tilde g\mid (\tilde g,b_i)\in m'\ \text{или}\ (\tilde g,m_i)\in b'\},\quad m^{\square}=\{\tilde m\mid(\tilde m,b_i)\in g'\ \text{или}\ (g_i,\tilde m)\in b'\},\quad b^{\square}=\{\tilde b\mid (g_i,\tilde b)\in m'\ \text{или}\ (m_i,\tilde b)\in g'\}.$$
Для тройки $(g,m,b)\in Y$ **OAC-трикластер** (box-вариант) $T=(g^{\square},m^{\square},b^{\square})$; prime-вариант (Ignatov et al. 2013): $T=\big((m,b)',(g,b)',(g,m)'\big)$. **Плотность** $\rho(A,B,C)=\dfrac{|Y\cap A\times B\times C|}{|A||B||C|}$; трикластер плотный, если $\rho(T)\ge\rho_{min}$; для триконцептов $\rho=1$; при $\rho_{min}=0$ каждый триконцепт содержится в некотором трикластере (Proposition 1). Алгоритм TRICL: один проход по $Y$, для каждой тройки – проверка плотности и добавление в хеш; сложность $O(|Y|^2\log|Y|\cdot|G||M||B|)$ в худшем случае, с вероятностной оценкой плотности (случайная выборка ~10% ячеек) – на порядки быстрее Trias.

### Как это применить к кубу «МО × категория × месяц» (раздел «дальнейшая работа»)
Бинаризовать куб: $(\text{МО},\text{категория},\text{месяц})\in Y$, если доля категории в тратах МО в этом месяце превышает порог (например, медиану по всем МО или $+\tau$ стандартных отклонений от среднего по стране). Тогда плотные OAC-трикластеры = **группы МО, которые одновременно «переразмечены» по группе категорий в группе месяцев** – интерпретируемые пространственно-временные паттерны потребления (сезонные, региональные), без предварительного выбора $K$ и с допуском перекрытий. Альтернатива для вещественных долей – МНК-трикластеризация (Ignatov et al. 2015). Это дополняет (а не заменяет) кластеризацию атрибутированной сети: трикластеры дают «что именно и когда», кластеры – «кто на кого похож».

---

## 9. Савченко: на что уместно сослаться

Профиль: Andrey V. Savchenko – Sber AI Lab и ВШЭ (ведущий научный сотрудник; ранее ЛАТАС, Нижний Новгород). Основные темы – компьютерное зрение, распознавание эмоций, эффективные нейросети; в последние годы (Sber AI Lab) – **последовательности событий и временные ряды в банковских данных**. Прямых работ по кластеризации временных рядов или динамических графов **не найдено** (проверка через OpenAlex по автору: запросы cluster/temporal/time series/graph). Уместные ссылки:
- Временные графы на транзакционных данных: Makarov I., Savchenko A., Korovko A., Sherstyuk L., Severin N., Kiselev D., Mikheev A., Babaev D. Temporal network embedding framework with causal anonymous walks representations // PeerJ Computer Science. 2022. Vol. 8. e858. DOI: 10.7717/peerj-cs.858 (первый сравнительный фреймворк для temporal network representation; приложение – кредитный скоринг по банковским транзакциям). Также: Proshian H., Severin N., Nikolenko S., Kireev I., Savchenko A., Sergeev I., Postnova M., Makarov I. Beyond Isolated Clients: Integrating Graph-Based Embeddings into Event Sequence Models. arXiv:2604.09085, 2026.
- Последовательности событий в банке: Mollaev D., Kostin A., Postnova M., Karpukhin I., Kireev I., Gusev G., Savchenko A. Multimodal Banking Dataset: Understanding Client Needs through Event Sequences. arXiv:2409.17587, 2024 (CIKM 2025); Karpukhin I., Savchenko A. V. Detecting the Future: All-at-Once Event Sequence Forecasting with Horizon Matching // Proc. AAAI. 2026. Vol. 40. DOI: 10.1609/aaai.v40i27.39413.
- Временные ряды: Tsururu: A Python-based Time Series Forecasting Strategies Library // IJCAI 2025. DOI: 10.24963/ijcai.2025/1266; HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting. arXiv:2511.08340, 2025.
- Кластеризация (методологические аналоги): Savchenko A. V. Clustering and maximum likelihood search for efficient statistical classification with medium-sized databases // Optimization Letters. 2017. DOI: 10.1007/s11590-015-0948-6; Savchenko A. V. Efficient facial representations for age, gender and identity recognition in organizing photo albums using multi-output ConvNet // PeerJ Computer Science. 2019. Vol. 5. e197. DOI: 10.7717/peerj-cs.197 (группировка/кластеризация в фотоальбомах); Savchenko A. V. Event Recognition with Automatic Album Detection based on Sequential Grouping of Confidence Scores and Neural Attention // IJCNN 2020. DOI: 10.1109/IJCNN48605.2020.9207675 (последовательная группировка – аналог сегментации временной последовательности на однородные отрезки).

**Как использовать.** В разделе о динамике сослаться на temporal-graph работу (PeerJ CS 2022) как на контекст «временные сети на банковских данных»; при построении признаков МО из транзакционных последовательностей – на MBD/ESQA-линию; если будем делать сегментацию временной оси (режимы), упомянуть «sequential grouping» как аналог. Не приписывать Савченко работ по кластеризации графов – их нет.

---

## 10. Динамическое выявление сообществ: ключевые ссылки и формулы

- **Mucha P. J., Richardson T., Macon K., Porter M. A., Onnela J.-P.** Community Structure in Time-Dependent, Multiscale, and Multiplex Networks // Science. 2010. Vol. 328, No. 5980. P. 876–878. DOI: 10.1126/science.1184819. Мультислойная модулярность:
$$Q_{\text{multislice}}=\frac{1}{2\mu}\sum_{ijsr}\Big[\Big(A_{ijs}-\gamma_s\frac{k_{is}k_{js}}{2m_s}\Big)\delta_{sr}+\delta_{ij}\,C_{jsr}\Big]\delta(g_{is},g_{jr}),$$
$s,r$ – срезы (месяцы), $\gamma_s$ – разрешение, $C_{jsr}=\omega$ – межсрезовая связь узла с самим собой (сглаживание), $2\mu=\sum_{jr}\kappa_{jr}$. Для корреляционных/сходственных сетей см. также Bazzi M., Porter M. A., Williams S., McDonald M., Fenn D. J., Howison S. D. Community Detection in Temporal Multilayer Networks, with an Application to Correlation Networks // Multiscale Modeling & Simulation. 2016. Vol. 14, No. 1. P. 1–41. DOI: 10.1137/15M1009615 (нуль-модель для корреляционных матриц – релевантно графу сходств МО).
- **Greene D., Doyle D., Cunningham P.** Tracking the Evolution of Communities in Dynamic Social Networks // ASONAM 2010. IEEE. P. 176–183. DOI: 10.1109/ASONAM.2010.17. Жизненный цикл сообществ: birth, death, merge, split, expansion, contraction, continuation; сопоставление сообществ соседних срезов по Жаккару $J(C_t,C_{t+1})=|C_t\cap C_{t+1}|/|C_t\cup C_{t+1}|\ge\theta$ (типично $\theta\approx0.3$–$0.5$), «динамическое сообщество» = цепочка сопоставленных фронтов.
- **Rossetti G., Cazabet R.** Community Discovery in Dynamic Networks: A Survey // ACM Computing Surveys. 2018. Vol. 51, No. 2. Art. 35. DOI: 10.1145/3172867. Таксономия: Instant-optimal (независимая кластеризация срезов + сопоставление), Temporal trade-off (инкрементально, с учётом прошлого), Cross-time (сразу на всей истории); компромисс «стабильность vs точность»; оценка динамических сообществ.
- **Shorrocks A. F.** The Measurement of Mobility // Econometrica. 1978. Vol. 46, No. 5. P. 1013–1024. DOI: 10.2307/1911433. Для матрицы переходов $\Pi$ ($K\times K$, строки суммируются в 1) индекс мобильности
$$M(\Pi)=\frac{K-\mathrm{tr}(\Pi)}{K-1}\in\Big[0,\frac{K}{K-1}\Big],$$
$M=0$ – полная неподвижность (все МО остаются в своих кластерах), $M=1$ – независимость от предыдущего состояния; предполагает фиксированную систему классов между периодами (нужно сопоставление меток кластеров, например венгерским алгоритмом по пересечению).
- Оптимизация модулярности: Blondel V. D., Guillaume J.-L., Lambiotte R., Lefebvre E. Fast unfolding of communities in large networks // J. Stat. Mech. 2008. P10008. DOI: 10.1088/1742-5468/2008/10/P10008; Traag V. A., Waltman L., van Eck N. J. From Louvain to Leiden: guaranteeing well-connected communities // Scientific Reports. 2019. Vol. 9. 5233. DOI: 10.1038/s41598-019-41695-z.

**Как использовать.** Протокол динамики: (1) Instant-optimal – KEFRiN/Louvain по месяцам + сопоставление кластеров по Жаккару (Greene) + события жизненного цикла; (2) Cross-time – мультислойная модулярность Мухи (Leiden с межсрезовым $\omega$) как кросс-проверка стабильности; (3) матрицы переходов и индекс Шоррокса + ARI/NMI между соседними месяцами как меры стабильности; классификация по Rossetti–Cazabet – в описании методологии.

---

## 11. Точные определения ICVI, которых будет ждать жюри (сводная таблица)

Обозначения: $d$ – расстояние (евклидово по стандартизованным признакам, если не указано иное), $\bar c$ – общий центроид, $c_k$ – центроид кластера, $\sigma(C)$ – вектор дисперсий по признакам.

| Индекс | Формула | Направление | На чём | Граничные случаи / базовые уровни |
|---|---|---|---|---|
| Silhouette Width, SW (Rousseeuw 1987) | $s_i=\dfrac{b_i-a_i}{\max(a_i,b_i)}$, $a_i$ – среднее $d$ до своего кластера, $b_i=\min_{l\ne k}$ среднего $d$ до чужого; $\mathrm{SW}=\frac1N\sum_i s_i$ | ↑ (в $[-1,1]$) | признаки (любая метрика; можно и графовое расстояние) | не определён при $K=1$; для синглтона $s_i:=0$ (sklearn); случайное разбиение $\approx0$; стабилен, насыщается с ростом числа признаков (Doklady 2025) |
| Calinski–Harabasz, CH (1974) | $\mathrm{CH}=\dfrac{B/(K-1)}{W/(N-K)}$, $B=\sum_k n_k\|c_k-\bar c\|^2$, $W=\sum_k\sum_{i\in S_k}\|y_i-c_k\|^2$ | ↑ | признаки | растёт $\sim N$ – отчитывать CH/N (Doklady 2025); не определён при $K=1$, $K=N$; благоволит выпуклым равным кластерам |
| Davies–Bouldin, DBI (1979) | $\mathrm{DBI}=\frac1K\sum_k\max_{l\ne k}\dfrac{\bar\sigma_k+\bar\sigma_l}{d(c_k,c_l)}$, $\bar\sigma_k$ – среднее $d$ до центроида | ↓ ($\ge0$) | признаки | базовый уровень случайного разбиения зависит от $K$ → нормировать на baseline; синглтон даёт $\bar\sigma=0$ (занижает) |
| $S_{Dbw}$ (Halkidi & Vazirgiannis 2001) | $S_{Dbw}=\mathrm{Scat}(K)+\mathrm{Dens\_bw}(K)$; $\mathrm{Scat}=\frac1K\sum_k\dfrac{\|\sigma(C_k)\|}{\|\sigma(Y)\|}$; $\mathrm{Dens\_bw}=\dfrac{1}{K(K-1)}\sum_k\sum_{l\ne k}\dfrac{\mathrm{dens}(u_{kl})}{\max(\mathrm{dens}(c_k),\mathrm{dens}(c_l))}$, $u_{kl}$ – середина отрезка $c_kc_l$, $\mathrm{dens}(u)=\sum_{x\in C_k\cup C_l}\mathbb{1}[d(x,u)\le \mathrm{stdev}]$, $\mathrm{stdev}=\frac1K\sqrt{\sum_k\|\sigma(C_k)\|}$ | ↓ | признаки | $K$-зависимый случайный базовый уровень (Doklady 2025); чувствителен к определению плотности/радиуса; не определён при $K=1$ |
| Average Isolability, AVI (Biswas & Biswas 2017) | $I(C_k)=\dfrac{\mathrm{in}_k}{\mathrm{in}_k+\mathrm{cut}_k}$ (суммы по матрице смежности, внутренние пары упорядочены), $\mathrm{AVI}=\frac1K\sum_k I(C_k)$ | ↑ (в $[0,1]$) | граф (взвешенный допустим) | случайное разбиение $\approx1/K$ при любых размерах (вывод в п. 4); растёт с ассортативностью; для SBM $\approx\frac{p_{in}n_k}{p_{in}n_k+p_{out}(N-n_k)}$ |
| Average Unifiability, AVU | $U(C_k,C_l)=\dfrac{e_{kl}}{\mathrm{cut}_k+\mathrm{cut}_l-e_{kl}}$; AVU – среднее $U$ по парам/кластерам (конвенцию зафиксировать) | ↓ (в $[0,1]$) | граф | **вырожден при $K\le3$**: $K=2\Rightarrow\mathrm{AVU}\equiv1$; $K=3\Rightarrow$ константа ($1/3$ при усреднении по парам, $2/3$ при усреднении по кластерам сумм по партнёрам); падает с $K$ к «полу» |
| ANUI | по Howie et al. 2023 (38): $\mathrm{ANUI}=\dfrac{\mathrm{AVI}}{1+\mathrm{AVI}\cdot\mathrm{AVU}}$ (сверить с оригиналом) | ↑ | граф | наследует $1/K$-базу AVI и вырожденность AVU при $K\le3$ |
| Модулярность $Q$ (Newman & Girvan 2004) | $Q=\sum_k\big[\frac{e_{kk}}{m}-(\frac{d_k}{2m})^2\big]$ | ↑ (в $[-\frac12,1]$) | граф | предел разрешения; случайное разбиение $\approx0$; на графе сходств зависит от разрежения; обобщение с $\gamma$ |
| Modularity density $D$ (Li et al. 2008) | $D=\sum_k\frac{L(V_k,V_k)-L(V_k,\bar V_k)}{|V_k|}$ | ↑ | граф | смягчает предел разрешения; не нормирован |
| Conductance (как дополнительный) | $\phi(C_k)=\dfrac{\mathrm{cut}_k}{\min(\mathrm{vol}(C_k),\mathrm{vol}(V\setminus C_k))}$, среднее по $k$ | ↓ | граф | $\phi=1-I$ при $\mathrm{vol}(C_k)\le \mathrm{vol}(\bar C_k)$ – дублирует AVI; приводить только одно |
| SCR, KDSE (Shalileh et al. 2026) | определения в открытом доступе не найдены | – | признаки | упомянуть как новые индексы авторов; не воспроизводить без текста |

Общие рекомендации (из п. 1): для каждого $K$ и каждого индекса считать перестановочный базовый уровень (случайное назначение узлов при тех же размерах кластеров) и доверительный интервал; отчитывать и сырое значение, и нормированное; в пространстве признаков использовать те же стандартизованные признаки, что подавались в алгоритм; в сетевом – ту же матрицу $A$/$P$ (до «модулярностной» трансформации, т.к. индексы Biswas определены для неотрицательных весов).

---

## 12. Что не удалось найти (честный список)
1. Полные тексты ESWA 2026 и Doklady 2025 (JS-челлендж/403 на ScienceDirect, Springer, Math-Net, ResearchGate). Следствие: нотация индексов в п. 11 – по первоисточникам индексов; четвёртый сетевой критерий ESWA и определения SCR/KDSE неизвестны.
2. Полный текст главы «Profiling Consumption…» (платный доступ): правило ребра, выбор $K$, формат интерпретации не верифицированы.
3. Первоисточник Biswas & Biswas 2017: конвенция усреднения AVU, точная формула ANUI и шесть «свойств качества» – взяты из вторичного источника (Howie et al. 2023) и помечены как требующие сверки.
4. Полный текст IEEE Access 2025 (CANUS): описание по авторскому README/docstring.
5. Полные тексты статей Алескерова и соавторов по паттерн-анализу (bijournal – таймаут; Procedia – не нашлось в alphaXiv): процедура описана по аннотациям.
6. Файл LICENSE в репозитории CANUS не обнаружен (возможно, отсутствует).

---

## 13. Список литературы (для отчёта)

1. Shalileh S., Antonov E., Tsyplakova D. Internal cluster validity indices for attributed networks: A controlled comparative study // Expert Systems with Applications. 2026. Vol. 333. Art. 133912. DOI: 10.1016/j.eswa.2026.133912.
2. Shalileh S., Antonov E. A., Tsyplakova D. A. Cluster Validity across Attribute and Network Spaces: Empirical Benchmarks for Attributed Networks Clustering // Doklady Mathematics. 2025. Vol. 112, No. 3. P. 553–564. DOI: 10.1134/S1064562425700589.
3. Shalileh S., Antonov E., Tsyplakova D. Profiling Consumption Using Attributed Network Clustering // Complex Networks & Their Applications XIV. Studies in Computational Intelligence. Cham: Springer, 2026. P. 77–88. DOI: 10.1007/978-3-032-16723-1_7.
4. Shalileh S., Mirkin B. Community Partitioning over Feature-Rich Networks Using an Extended K-Means Method // Entropy. 2022. Vol. 24, No. 5. Art. 626. DOI: 10.3390/e24050626.
5. Shalileh S., Mirkin B. Summable and nonsummable data-driven models for community detection in feature-rich networks // Social Network Analysis and Mining. 2021. Vol. 11. Art. 67. DOI: 10.1007/s13278-021-00774-8.
6. Mirkin B., Shalileh S. Community Detection in Feature-Rich Networks Using Data Recovery Approach // Journal of Classification. 2022. Vol. 39, No. 3. P. 432–462. DOI: 10.1007/s00357-022-09416-w.
7. Shalileh S., Mirkin B. Least-squares community extraction in feature-rich networks using similarity data // PLoS ONE. 2021. Vol. 16. e0254377. DOI: 10.1371/journal.pone.0254377.
8. Shalileh S., Mirkin B. Community Detection in Feature-Rich Networks Using Gradient Descent Approach // Complex Networks & Their Applications XII. Studies in Computational Intelligence. Springer, 2024. P. 185–196. DOI: 10.1007/978-3-031-53499-7_15.
9. Shalileh S. Gradient Descent Clustering with Regularization to Recover Communities in Transformed Attributed Networks // ASONAM 2024, LNCS. Springer, 2025. P. 137–148. DOI: 10.1007/978-3-031-78538-2_12.
10. Shalileh S. A Filtered Gradient Descent Clustering Method to Recover Communities in Attributed Networks // IEEE Access. 2025. Vol. 13. P. 169698–169719. DOI: 10.1109/ACCESS.2025.3614989.
11. Biswas A., Biswas B. Defining quality metrics for graph clustering evaluation // Expert Systems with Applications. 2017. Vol. 71. P. 1–17. DOI: 10.1016/j.eswa.2016.11.011.
12. Howie J., Srinivasan V., Thomo A. Scaling Up Structural Clustering to Large Probabilistic Graphs Using Lyapunov Central Limit Theorem // Proc. VLDB Endowment. 2023. Vol. 16, No. 11. P. 3165–3177. DOI: 10.14778/3611479.3611516.
13. Rousseeuw P. J. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis // J. Comput. Appl. Math. 1987. Vol. 20. P. 53–65. DOI: 10.1016/0377-0427(87)90125-7.
14. Caliński T., Harabasz J. A dendrite method for cluster analysis // Communications in Statistics. 1974. Vol. 3, No. 1. P. 1–27. DOI: 10.1080/03610927408827101.
15. Davies D. L., Bouldin D. W. A Cluster Separation Measure // IEEE TPAMI. 1979. Vol. PAMI-1, No. 2. P. 224–227. DOI: 10.1109/TPAMI.1979.4766909.
16. Halkidi M., Vazirgiannis M. Clustering validity assessment: finding the optimal partitioning of a data set // Proc. IEEE ICDM 2001. P. 187–194. DOI: 10.1109/ICDM.2001.989517.
17. Newman M. E. J., Girvan M. Finding and evaluating community structure in networks // Physical Review E. 2004. Vol. 69. 026113. DOI: 10.1103/PhysRevE.69.026113.
18. Li Z., Zhang S., Wang R.-S., Zhang X.-S., Chen L. Quantitative function for community detection // Physical Review E. 2008. Vol. 77. 036109. DOI: 10.1103/PhysRevE.77.036109.
19. Reichardt J., Bornholdt S. Statistical mechanics of community detection // Physical Review E. 2006. Vol. 74. 016110. DOI: 10.1103/PhysRevE.74.016110.
20. Mancoridis S., Mitchell B. S., Rorres C., Chen Y., Gansner E. R. Using automatic clustering to produce high-level system organizations of source code // Proc. IWPC'98. IEEE, 1998. P. 45–52. DOI: 10.1109/WPC.1998.693283.
21. Mitchell B. S., Mancoridis S. On the automatic modularization of software systems using the Bunch tool // IEEE Trans. Software Engineering. 2006. Vol. 32, No. 3. P. 193–208. DOI: 10.1109/TSE.2006.31.
22. Mirkin B. Clustering: A Data Recovery Approach. 2nd ed. Boca Raton: CRC Press, 2012.
23. Mirkin B. Core Concepts in Data Analysis: Summarization, Correlation and Visualization. London: Springer, 2011.
24. Aleskerov F., Chistyakov V. V., Kalyagin V. The threshold aggregation // Economics Letters. 2010. Vol. 107, No. 2. P. 261–262. DOI: 10.1016/j.econlet.2010.01.041.
25. Aleskerov F. T., Yakuba V. I., Yuzbashev D. A. A 'threshold aggregation' of three-graded rankings // Mathematical Social Sciences. 2007. Vol. 53. P. 106–110.
26. Aleskerov F., Chistyakov V., Kalyagin V. Multiple Criteria Threshold Decision Making Algorithms. SSRN Working Paper, 2010. DOI: 10.2139/ssrn.1680157.
27. Алескеров Ф. Т., Белоусова В. Ю., Егорова Л. Г., Миркин Б. Г. Анализ паттернов в статике и динамике, часть 1: обзор литературы и уточнение понятия // Бизнес-информатика. 2013. № 3(25). С. 3–18; часть 2 // Бизнес-информатика. 2013. № 4(26).
28. Aleskerov F., Egorova L., Gokhberg L., Myachin A., Sagieva G. Pattern Analysis in the Study of Science, Education and Innovative Activity in Russian Regions // Procedia Computer Science. 2013. Vol. 17. P. 687–694. DOI: 10.1016/j.procs.2013.05.089.
29. Ignatov D. I., Kuznetsov S. O., Magizov R. A., Zhukov L. E. From Triconcepts to Triclusters // RSFDGrC 2011. LNCS 6743. Springer, 2011. P. 257–264. DOI: 10.1007/978-3-642-21881-1_41.
30. Ignatov D. I., Kuznetsov S. O., Poelmans J., Zhukov L. E. Can triconcepts become triclusters? // International Journal of General Systems. 2013. Vol. 42, No. 6. P. 572–593. DOI: 10.1080/03081079.2013.798899.
31. Ignatov D. I., Gnatyshak D. V., Kuznetsov S. O., Mirkin B. G. Triadic Formal Concept Analysis and triclustering: searching for optimal patterns // Machine Learning. 2015. Vol. 101, No. 1–3. P. 271–302. DOI: 10.1007/s10994-015-5487-y.
32. Ganter B., Wille R. Formal Concept Analysis: Mathematical Foundations. Berlin: Springer, 1999.
33. Makarov I., Savchenko A., Korovko A., Sherstyuk L., Severin N., Kiselev D., Mikheev A., Babaev D. Temporal network embedding framework with causal anonymous walks representations // PeerJ Computer Science. 2022. Vol. 8. e858. DOI: 10.7717/peerj-cs.858.
34. Mollaev D., Kostin A., Postnova M., Karpukhin I., Kireev I., Gusev G., Savchenko A. Multimodal Banking Dataset: Understanding Client Needs through Event Sequences. arXiv:2409.17587, 2024.
35. Karpukhin I., Savchenko A. V. Detecting the Future: All-at-Once Event Sequence Forecasting with Horizon Matching // Proc. AAAI Conference on Artificial Intelligence. 2026. Vol. 40. DOI: 10.1609/aaai.v40i27.39413.
36. Savchenko A. V. Efficient facial representations for age, gender and identity recognition in organizing photo albums using multi-output ConvNet // PeerJ Computer Science. 2019. Vol. 5. e197. DOI: 10.7717/peerj-cs.197.
37. Mucha P. J., Richardson T., Macon K., Porter M. A., Onnela J.-P. Community Structure in Time-Dependent, Multiscale, and Multiplex Networks // Science. 2010. Vol. 328, No. 5980. P. 876–878. DOI: 10.1126/science.1184819.
38. Bazzi M., Porter M. A., Williams S., McDonald M., Fenn D. J., Howison S. D. Community Detection in Temporal Multilayer Networks, with an Application to Correlation Networks // Multiscale Modeling & Simulation. 2016. Vol. 14, No. 1. P. 1–41. DOI: 10.1137/15M1009615.
39. Greene D., Doyle D., Cunningham P. Tracking the Evolution of Communities in Dynamic Social Networks // Proc. ASONAM 2010. IEEE. P. 176–183. DOI: 10.1109/ASONAM.2010.17.
40. Rossetti G., Cazabet R. Community Discovery in Dynamic Networks: A Survey // ACM Computing Surveys. 2018. Vol. 51, No. 2. Art. 35. DOI: 10.1145/3172867.
41. Shorrocks A. F. The Measurement of Mobility // Econometrica. 1978. Vol. 46, No. 5. P. 1013–1024. DOI: 10.2307/1911433.
42. Blondel V. D., Guillaume J.-L., Lambiotte R., Lefebvre E. Fast unfolding of communities in large networks // J. Stat. Mech. 2008. P10008. DOI: 10.1088/1742-5468/2008/10/P10008.
43. Traag V. A., Waltman L., van Eck N. J. From Louvain to Leiden: guaranteeing well-connected communities // Scientific Reports. 2019. Vol. 9. 5233. DOI: 10.1038/s41598-019-41695-z.
44. Citraro S., Rossetti G. Identifying and exploiting homogeneous communities in labeled networks // Applied Network Science. 2020. Vol. 5. Art. 55. DOI: 10.1007/s41109-020-00302-1.
45. Galdeman A., Zignani M., Gaito S. City consumption profile: a city perspective on the spending behavior of citizens // Applied Network Science. 2021. Vol. 6. DOI: 10.1007/s41109-021-00406-2.
46. Tsitsulin A., Palowitch J., Perozzi B., Müller E. Graph clustering with graph neural networks // Journal of Machine Learning Research. 2023. Vol. 24, No. 127. P. 1–21.
47. Bothorel C., Cruz J. D., Magnani M., Micenková B. Clustering attributed graphs: models, measures and methods // Network Science. 2015. Vol. 3, No. 3. P. 408–444. DOI: 10.1017/nws.2015.9.
48. Chunaev P. Community detection in node-attributed social networks: A survey // Computer Science Review. 2020. Vol. 37. 100286.
