(function () {
    const TerminalUI = (window.TerminalUI = window.TerminalUI || {});

    function buildEchartsSeries(series) {
        return (series || []).map((item) => ({
            name: item.name,
            type: "line",
            showSymbol: false,
            smooth: false,
            lineStyle: { width: 2, color: item.color || "#FFFFFF" },
            itemStyle: { color: item.color || "#FFFFFF" },
            data: (item.points || []).map((point) => [point.time, point.value]),
        }));
    }

    function attachResize(element, instance, lightweight) {
        const observer = new ResizeObserver(() => {
            const width = element.clientWidth || 320;
            const height = element.clientHeight || 220;
            if (lightweight) {
                instance.applyOptions({ width, height });
            } else {
                instance.resize({ width, height });
            }
        });
        observer.observe(element);
        const width = element.clientWidth || 320;
        const height = element.clientHeight || 220;
        if (lightweight) {
            instance.applyOptions({ width, height });
        } else {
            instance.resize({ width, height });
        }
    }

    function renderLineChart(element, payload) {
        if (!payload.series || payload.series.length === 0) {
            element.innerHTML = '<div class="terminal-empty-state">No chart data available.</div>';
            return;
        }
        const chart = echarts.init(element);
        const markLineData = (payload.thresholds || []).map((threshold) => ({
            yAxis: threshold.value,
            label: { formatter: threshold.label || "" },
        }));
        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            tooltip: { trigger: "axis" },
            legend: {
                top: 0,
                textStyle: { color: "#FFFFFF" },
            },
            grid: { left: 48, right: 20, top: 40, bottom: 28 },
            xAxis: {
                type: "time",
                axisLine: { lineStyle: { color: "#222222" } },
                axisLabel: { color: "#8A8A8A" },
                splitLine: { lineStyle: { color: "#111111" } },
            },
            yAxis: {
                type: "value",
                axisLine: { lineStyle: { color: "#222222" } },
                axisLabel: { color: "#8A8A8A" },
                splitLine: { lineStyle: { color: "#111111" } },
            },
            series: buildEchartsSeries(payload.series).map((item) => ({
                ...item,
                markLine: markLineData.length > 0 ? { symbol: "none", lineStyle: { color: "#444444" }, data: markLineData } : undefined,
            })),
        });
        attachResize(element, chart);
    }

    function renderHeatmap(element, payload) {
        if (!payload.values || payload.values.length === 0) {
            element.innerHTML = '<div class="terminal-empty-state">No heatmap data available.</div>';
            return;
        }
        const chart = echarts.init(element);
        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            tooltip: {},
            grid: { left: 90, right: 18, top: 20, bottom: 40 },
            xAxis: {
                type: "category",
                data: payload.xLabels || [],
                splitArea: { show: true },
                axisLabel: { color: "#8A8A8A", rotate: 20 },
                axisLine: { lineStyle: { color: "#222222" } },
            },
            yAxis: {
                type: "category",
                data: payload.yLabels || [],
                splitArea: { show: true },
                axisLabel: { color: "#8A8A8A" },
                axisLine: { lineStyle: { color: "#222222" } },
            },
            visualMap: {
                min: -1,
                max: 1,
                calculable: false,
                orient: "horizontal",
                left: "center",
                bottom: 0,
                textStyle: { color: "#8A8A8A" },
                inRange: { color: ["#EF4444", "#1A1A1A", "#22C55E"] },
            },
            series: [
                {
                    type: "heatmap",
                    data: payload.values,
                    label: { show: true, color: "#FFFFFF", formatter: ({ value }) => Number(value[2]).toFixed(2) },
                    emphasis: { itemStyle: { borderColor: "#FFFFFF", borderWidth: 1 } },
                },
            ],
        });
        attachResize(element, chart);
    }

    function renderBarChart(element, payload) {
        if (!payload.categories || payload.categories.length === 0) {
            element.innerHTML = '<div class="terminal-empty-state">No bar-chart data available.</div>';
            return;
        }
        const chart = echarts.init(element);
        const hasSecondAxis = (payload.series || []).some((s) => s.yAxisIndex === 1);
        const yAxis = [
            {
                type: "value",
                axisLabel: { color: "#8A8A8A" },
                axisLine: { lineStyle: { color: "#222222" } },
                splitLine: { lineStyle: { color: "#111111" } },
            },
        ];
        if (hasSecondAxis) {
            yAxis.push({
                type: "value",
                axisLabel: { color: "#3B82F6", formatter: "{value}%" },
                axisLine: { lineStyle: { color: "#3B82F6" } },
                splitLine: { show: false },
            });
        }
        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
            legend: { top: 0, textStyle: { color: "#FFFFFF" } },
            grid: { left: 56, right: hasSecondAxis ? 56 : 20, top: 36, bottom: 28 },
            xAxis: {
                type: "category",
                data: payload.categories,
                axisLabel: { color: "#8A8A8A" },
                axisLine: { lineStyle: { color: "#222222" } },
            },
            yAxis: yAxis,
            series: (payload.series || []).map((item, index) => ({
                name: item.name,
                type: "bar",
                yAxisIndex: item.yAxisIndex || 0,
                data: item.values,
                itemStyle: { color: ["#22C55E", "#3B82F6", "#EAB308"][index % 3] },
            })),
        });
        attachResize(element, chart);
    }

    function renderDistributionChart(element, payload) {
        if (!payload.bins || payload.bins.length === 0) {
            element.innerHTML = '<div class="terminal-empty-state">No distribution data available.</div>';
            return;
        }
        const chart = echarts.init(element);
        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            tooltip: { trigger: "axis" },
            grid: { left: 48, right: 20, top: 20, bottom: 28 },
            xAxis: {
                type: "category",
                data: payload.bins.map((bin) => bin.label),
                axisLabel: { color: "#8A8A8A", interval: Math.max(0, Math.floor(payload.bins.length / 10)) },
                axisLine: { lineStyle: { color: "#222222" } },
            },
            yAxis: {
                type: "value",
                axisLabel: { color: "#8A8A8A" },
                axisLine: { lineStyle: { color: "#222222" } },
                splitLine: { lineStyle: { color: "#111111" } },
            },
            series: [
                {
                    type: "bar",
                    data: payload.bins.map((bin) => bin.value),
                    itemStyle: { color: "#22C55E" },
                },
            ],
        });
        attachResize(element, chart);
    }

    function renderEquityChart(element, payload) {
        if (!payload.series || payload.series.length === 0) {
            element.innerHTML = '<div class="terminal-empty-state">No time-series data available.</div>';
            return;
        }

        element.innerHTML = "";
        const hasDrawdown = payload.series.some((s) => s.priceScaleId === "drawdown");

        const chart = LightweightCharts.createChart(element, {
            layout: {
                background: { color: "#0D0D0D" },
                textColor: "#FFFFFF",
                fontFamily: "JetBrains Mono",
            },
            grid: {
                vertLines: { color: "#111111" },
                horzLines: { color: "#111111" },
            },
            leftPriceScale: {
                visible: hasDrawdown,
                borderColor: "#222222",
                scaleMargins: { top: 0.75, bottom: 0 },
            },
            rightPriceScale: {
                borderColor: "#222222",
            },
            timeScale: {
                borderColor: "#222222",
                timeVisible: true,
            },
            crosshair: {
                vertLine: { color: "#444444" },
                horzLine: { color: "#444444" },
            },
        });

        payload.series.forEach((seriesItem) => {
            const opts = {
                color: seriesItem.color || "#FFFFFF",
                lineWidth: seriesItem.lineWidth || 2,
                title: seriesItem.name,
                priceScaleId: seriesItem.priceScaleId || "right",
            };
            const lineSeries = chart.addLineSeries(opts);
            lineSeries.setData(
                (seriesItem.points || []).map((point) => ({
                    time: Math.floor(new Date(point.time).getTime() / 1000),
                    value: point.value,
                })),
            );
        });

        if (hasDrawdown) {
            chart.priceScale("left").applyOptions({
                scaleMargins: { top: 0.75, bottom: 0 },
                borderColor: "#222222",
            });
        }

        chart.timeScale().fitContent();
        attachResize(element, chart, true);
    }

    function renderChart(element) {
        const endpoint = element.dataset.chartEndpoint;
        const renderer = element.dataset.chartRenderer;
        if (!endpoint || !renderer) {
            return;
        }
        fetch(endpoint, { headers: { Accept: "application/json" } })
            .then((response) => response.json())
            .then((payload) => {
                if (renderer === "equity") {
                    renderEquityChart(element, payload);
                    return;
                }
                if (renderer === "heatmap") {
                    renderHeatmap(element, payload);
                    return;
                }
                if (renderer === "bar") {
                    renderBarChart(element, payload);
                    return;
                }
                if (renderer === "distribution") {
                    renderDistributionChart(element, payload);
                    return;
                }
                renderLineChart(element, payload);
            })
            .catch(() => {
                element.innerHTML = '<div class="terminal-empty-state">Chart request failed.</div>';
            });
    }

    function wireCorrelationHorizon(root) {
        const select = root.querySelector ? root.querySelector("#correlation-horizon-select") : null;
        if (!select || select.dataset.terminalHorizonBound === "true") {
            return;
        }
        select.dataset.terminalHorizonBound = "true";
        select.addEventListener("change", function () {
            const horizon = select.value;
            const container = document.getElementById("correlation-heatmaps");
            if (!container) {
                return;
            }
            container.querySelectorAll(".terminal-chart[data-chart-endpoint]").forEach((el) => {
                const current = el.dataset.chartEndpoint || "";
                const updated = current.replace(/horizon=[^&]+/, "horizon=" + horizon);
                el.dataset.chartEndpoint = updated;
                el.innerHTML = "";
                renderChart(el);
            });
            const sidebarSelect = document.querySelector('#dashboard-filters select[name="correlation_horizon"]');
            if (sidebarSelect) {
                sidebarSelect.value = horizon;
            }
        });
    }

    TerminalUI.initCharts = function (root) {
        root.querySelectorAll(".terminal-chart[data-chart-endpoint]").forEach((element) => {
            renderChart(element);
        });
        wireCorrelationHorizon(root);
    };
})();
