Файл со всеми идеями, задача из него создать 3 фазный план  по написанию дэшборд первой вкладки!

Важно!
1. дэшборд должен работать с single-asset backtester также хорошо как и с portfolio-backtester!!
2. Важно, чтобы streamlit выводил разные версии дэшборда при выборе single/portfolio.
3. Также вещи как PnL breakdown в Equity Curve на стратегии не нужны single asset backtest, но очень нужны portfolio backtest!
3.1 Такая вещь как Long/Short equity разделение в single asset нужно! В то время как в Portfolio, в Equity Curve нет!
4. То же самое касаеться и корреляционной матрицы, и других тяжелых расчетов.

-------------------------------------

Вкладка PnL Analysis

Типичная структура.

PnL Analysis
│
├─ Equity curve + лог вывод данных по доходности портфеля, точно также как сейчас(с p-value и прочим. Ничего не удалять!)
- При наведении на отдельную пнл отдельной стратегии вылезает окошко с данными отдельно по этой стратегии. (точно также как сейчас в single-asset backtester)
├─ Strategy PnL decomposition
├─ Strategy correlation
├─ Exposure correlation
├─ PnL distribution
└─ Exit analysis

----

Equity curve

portfolio (пнл порфтеля)
strategy breakdown (отдельно по стратегиям)
benchmark comparison (пнл ES)
rolling sharp (за 90 дней)

---

Strategy PnL decomposition (таблица)

достаточно:

Strategy PnL
PnL contribution
Volatility contribution
Drawdown contribution
VaR contribution
Signal vs Execution PnL

Подробнее:
Базовая структура данных (что нужно хранить)

Минимальная таблица трейдов:

trade_id	strategy	entry_time	exit_time	entry_price	exit_price	size	fees	slippage

Из этого считаются:

gross_pnl = (exit_price - entry_price) * size
net_pnl = gross_pnl - fees - slippage

Далее нужно агрегировать по времени (обычно по бару или по дню).

Например:

time	strategy	pnl
09:30	strat1	120
09:30	strat2	-30
09:30	strat3	40
2. Strategy PnL

Самое простое.

pnl_by_strategy = sum(net_pnl grouped by strategy)

Пример:

strategy	pnl
S1	40k
S2	20k
S3	-5k
3. Contribution to total PnL
contribution = strategy_pnl / total_portfolio_pnl

Пример:

total pnl = 55k

S1 = 40k / 55k = 73%
S2 = 20k / 55k = 36%
S3 = -5k / 55k = -9%
4. Contribution to Volatility

Сначала нужно получить time series PnL.

Пример:

time	S1	S2	S3
t1	10	-5	2
t2	5	3	-1
t3	-3	4	0

Далее считаем:

portfolio_pnl = S1 + S2 + S3
portfolio_vol = std(portfolio_pnl)

Contribution считается через covariance.

Формула:

𝑅
𝐶
𝑖
=
𝑐
𝑜
𝑣
(
𝑃
𝑛
𝐿
𝑖
,
𝑃
𝑛
𝐿
𝑝
𝑜
𝑟
𝑡
𝑓
𝑜
𝑙
𝑖
𝑜
)
𝑉
𝑎
𝑟
(
𝑃
𝑛
𝐿
𝑝
𝑜
𝑟
𝑡
𝑓
𝑜
𝑙
𝑖
𝑜
)
RC
i
	​

=
Var(PnL
portfolio
	​

)
cov(PnL
i
	​

,PnL
portfolio
	​

)
	​


Интуитивно:

если стратегия сильно двигает портфель → её вклад в волатильность большой.

5. Contribution to Drawdown

Drawdown считается от equity curve.

equity = cumulative_sum(pnl)
drawdown = equity - rolling_max(equity)

Далее:

strategy_drawdown_contribution =
strategy_pnl_during_drawdown / portfolio_drawdown

Пример:

портфель упал -20k

strategy	pnl during DD
S1	-12k
S2	-6k
S3	-2k
6. Contribution to VaR (коротко)

VaR = worst expected loss at confidence level.

Пример:

95% VaR = потеря, которая происходит 1 раз из 20 дней.

Пример:

daily pnl distribution
day	pnl
1	100
2	-50
3	40
4	-200

95% VaR ≈ 5th percentile.

Contribution к VaR считается через marginal VaR:

𝑉
𝑎
𝑅
𝑖
=
𝑤
𝑒
𝑖
𝑔
ℎ
𝑡
𝑖
×
∂
𝑉
𝑎
𝑅
∂
𝑤
𝑒
𝑖
𝑔
ℎ
𝑡
𝑖
VaR
i
	​

=weight
i
	​
×
∂weight
i
	​
∂VaR
	​

Но на практике делают проще:
Simulation approach
for each day:
    portfolio pnl
    strategy pnl

Смотрят дни когда портфель в VaR-хвосте.

И считают:
average strategy pnl in worst 5% days
Это и есть VaR contribution.

7. Signal PnL
Это PnL от движения цены, без издержек.
signal_pnl = (exit_price - entry_price) * size

8. Execution PnL
Потери на исполнении:
execution_cost = fees + slippage + spread

Можно разделить:
spread_cost
market_impact
fees

------------------------------------------

Strategy correlation

Матрица:

corr(strategy pnl)

Heatmap.

Exposure correlation

Считается так:

position_value / portfolio_value

Матрица:

corr(exposures)

Это показывает реальную зависимость риска, а не только PnL.

--------

**Важная техническая деталь (часто делают неправильно)

PnL корреляцию нельзя считать по cumulative equity.

Нужно считать по:

bar pnl
или
daily pnl

Иначе корреляция будет искажена.

-------

Для интрадей портфеля лучше считать корреляции на нескольких горизонтах:

1 day

1 week

1 month

Потому что стратегии могут выглядеть независимыми на сделках, но становиться коррелированными в стресс-дни.

--------------------------------------------------

PnL distribution

Типичные графики:

daily returns histogram
rolling returns
skew / kurtosis
tail losses

-------------------------------------------------------

Exit analysis

В систематических фондах это один из самых важных модулей.

Типичные метрики:

PnL vs holding time
PnL vs exit signal
PnL vs volatility regime

Пример графиков:

Holding period
x = holding days
y = avg pnl
PnL decay
PnL if exit at t
day 1
day 2
day 3
...

Это показывает оптимальный exit horizon.

Conditional exits

Например:

exit because stop
exit because signal
exit because time

PnL по типам выхода.

-----

ПОДРОБНЕЕ НИЖЕ!!!

Этот модуль существует для ответа на один вопрос: **правильно ли стратегия выходит из позиции**.
В систематических фондах его делают отдельно, потому что **exit часто определяет 30–60% PnL стратегии**.

Но важно: **этот анализ почти всегда делается на уровне отдельной стратегии**, а не сразу для портфеля. Причина — разные стратегии имеют разные логики входа и удержания позиции.

Поэтому твоя текущая реализация **на одну стратегию — это правильная архитектура**. Нужно лишь сделать её **обобщаемой**, чтобы запускать одинаковый анализ для каждой стратегии.

---

# 1. Что должен содержать trade log

Чтобы делать exit-analysis, трейды должны хранить дополнительные поля:

| trade_id | strategy | entry_time | exit_time | entry_price | exit_price | size | exit_reason |
| -------- | -------- | ---------- | --------- | ----------- | ---------- | ---- | ----------- |

`exit_reason` — ключевая колонка.

Пример:

* signal
* stop
* take_profit
* time_exit
* risk_close

---

# 2. PnL vs holding time

Сначала считаешь **holding time**:

[
holding = exit_time - entry_time
]

Например:

| trade | holding_minutes | pnl |
| ----- | --------------- | --- |
| 1     | 5               | 12  |
| 2     | 15              | -8  |
| 3     | 40              | 25  |

Далее группировка:

```text
holding_bucket = [0-5, 5-15, 15-30, 30-60, 60+]
```

Получается:

| holding bucket | avg pnl |
| -------------- | ------- |
| 0-5m           | -2      |
| 5-15m          | 5       |
| 15-30m         | 8       |
| 30-60m         | 3       |

Это показывает:

**где PnL начинает деградировать.**

Очень часто видно:

* быстрые сделки → шум
* средние → прибыль
* долгие → возврат к нулю

---

# 3. PnL decay (очень важный график)

Смысл:

**что произошло бы, если бы позицию закрывали раньше/позже.**

Для каждой сделки нужно построить **path PnL**.

Пример:

| time since entry | price |
| ---------------- | ----- |
| 0                | 100   |
| 5m               | 102   |
| 10m              | 103   |
| 20m              | 101   |
| 40m              | 104   |

Если exit был на 40m:

| hypothetical exit | pnl |
| ----------------- | --- |
| 5m                | 2   |
| 10m               | 3   |
| 20m               | 1   |
| 40m               | 4   |

Теперь усредняем по всем трейдам:

| holding | avg pnl |
| ------- | ------- |
| 5m      | 1.8     |
| 10m     | 2.1     |
| 20m     | 1.5     |
| 40m     | 1.2     |

Это показывает:

**оптимальный horizon выхода.**

Очень часто видно, что:

* стратегия держит позицию **слишком долго**.

---

# 4. PnL vs exit signal

Используется `exit_reason`.

Пример:

| exit reason | trades | avg pnl |
| ----------- | ------ | ------- |
| signal      | 500    | 12      |
| stop        | 200    | -20     |
| time_exit   | 300    | 2       |

Это показывает:

* где стратегия реально зарабатывает
* какие выходы просто **режут убыток**

---

# 5. PnL vs volatility regime

Нужно добавить **волатильность рынка во время сделки**.

Например:

```text
volatility = ATR / price
```

или

```text
realized volatility
```

Потом делаем buckets:

| vol regime | avg pnl |
| ---------- | ------- |
| low vol    | -2      |
| medium vol | 5       |
| high vol   | 12      |

Это показывает:

**в каком режиме стратегия работает.**

---

# 6. Нужно ли это делать сразу для 5 стратегий

Нет.

Правильная архитектура:

```
for strategy in strategies:
    run_exit_analysis(strategy)
```

То есть **анализ остаётся per-strategy**, просто код универсальный.

В результате ты получишь:

```
exit_report_strategy_1
exit_report_strategy_2
exit_report_strategy_3
...
```

Именно так делают почти все фонды.

---

# 7. Когда делают portfolio exit analysis

Редко.

Это имеет смысл только если:

* стратегии торгуют **одни и те же инструменты**
* у них **похожий holding horizon**

Иначе результаты будут смешаны.

---

# 8. Что реально стоит добавить (важнее многих графиков)

Есть один анализ, который почти всегда даёт инсайты.

### MFE / MAE

Maximum Favorable Excursion
Maximum Adverse Excursion

| trade | pnl | mfe | mae |
| ----- | --- | --- | --- |
| 1     | 10  | 30  | -5  |
| 2     | -8  | 12  | -20 |

Это показывает:

* **сколько прибыли стратегия не забирает**
* **насколько глубоко уходят убыточные сделки**

Это напрямую помогает:

* улучшить take-profit
* улучшить stop.

---

# 9. Практический вывод

Для портфеля из 5 стратегий **достаточно**:

Per-strategy:

1. PnL vs holding time
2. PnL decay
3. PnL vs exit reason
4. MFE / MAE

Этого уже достаточно для серьёзного анализа.

---

# 10. Важная реальность

Большинство систематических стратегий **ломаются не из-за входа**, а из-за:

* плохого exit horizon
* неправильного stop
* слишком долгого удержания позиции

Поэтому систематические фонды уделяют **exit analysis почти столько же времени, сколько сигналу**.
