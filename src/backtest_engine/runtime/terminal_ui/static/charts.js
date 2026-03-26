(function () {
    const TerminalUI = (window.TerminalUI = window.TerminalUI || {});
    let pendingChartRequests = 0;
    let loadingWordTimerId = null;
    let loadingEtaTimerId = null;
    let loadingWordIndex = 0;
    let loadingSessionStartedAtMs = 0;
    const recentRequestDurationsMs = [];
    const MAX_STORED_DURATIONS = 20;

    function resetEchartInstance(element) {
        const existing = echarts.getInstanceByDom(element);
        if (existing) {
            existing.dispose();
        }
    }

    function getLoadingOverlay() {
        return document.getElementById("terminal-loading-overlay");
    }

    function getLoadingConfig() {
        const overlay = getLoadingOverlay();
        if (!overlay) {
            return {
                words: ["Loading data", "Building correlations", "Syncing charts", "Computing metrics", "Finalizing view"],
                wordIntervalMs: 1100,
                etaPerRequestSeconds: 2.2,
            };
        }
        let words = [];
        try {
            words = JSON.parse(overlay.dataset.loadingWords || "[]");
        } catch (_) {
            words = [];
        }
        const cleanedWords = Array.isArray(words) ? words.filter((x) => typeof x === "string" && x.trim().length > 0) : [];
        const parsedInterval = Number(overlay.dataset.loadingWordIntervalMs || 1100);
        const parsedEta = Number(overlay.dataset.loadingEtaPerRequestSeconds || 2.2);
        return {
            words: cleanedWords.length > 0 ? cleanedWords : ["Loading data"],
            wordIntervalMs: Number.isFinite(parsedInterval) && parsedInterval > 150 ? parsedInterval : 1100,
            etaPerRequestSeconds: Number.isFinite(parsedEta) && parsedEta > 0 ? parsedEta : 2.2,
        };
    }

    function averageRequestDurationMs() {
        if (recentRequestDurationsMs.length === 0) {
            const cfg = getLoadingConfig();
            return cfg.etaPerRequestSeconds * 1000;
        }
        const sum = recentRequestDurationsMs.reduce((acc, value) => acc + value, 0);
        return sum / recentRequestDurationsMs.length;
    }

    function updateLoadingEtaText() {
        const etaElement = document.getElementById("terminal-loading-eta");
        if (!etaElement) {
            return;
        }
        if (pendingChartRequests <= 0) {
            etaElement.textContent = "ETA ~0.0s";
            return;
        }
        const elapsedMs = Math.max(0, Date.now() - loadingSessionStartedAtMs);
        const estimatedRemainingMs = Math.max(
            100,
            pendingChartRequests * averageRequestDurationMs() - elapsedMs,
        );
        etaElement.textContent = "ETA ~" + (estimatedRemainingMs / 1000).toFixed(1) + "s";
    }

    function rotateLoadingWord() {
        const wordElement = document.getElementById("terminal-loading-word");
        if (!wordElement) {
            return;
        }
        const cfg = getLoadingConfig();
        const words = cfg.words;
        if (words.length === 0) {
            return;
        }
        loadingWordIndex = (loadingWordIndex + 1) % words.length;
        wordElement.classList.add("is-fading");
        setTimeout(() => {
            wordElement.textContent = words[loadingWordIndex];
            wordElement.classList.remove("is-fading");
        }, 130);
    }

    function stopLoadingAnimationTimers() {
        if (loadingWordTimerId !== null) {
            clearInterval(loadingWordTimerId);
            loadingWordTimerId = null;
        }
        if (loadingEtaTimerId !== null) {
            clearInterval(loadingEtaTimerId);
            loadingEtaTimerId = null;
        }
    }

    function startLoadingAnimationTimers() {
        const cfg = getLoadingConfig();
        stopLoadingAnimationTimers();
        const wordElement = document.getElementById("terminal-loading-word");
        if (wordElement) {
            loadingWordIndex = 0;
            wordElement.textContent = cfg.words[loadingWordIndex] || "Loading data";
        }
        updateLoadingEtaText();
        loadingWordTimerId = window.setInterval(rotateLoadingWord, cfg.wordIntervalMs);
        loadingEtaTimerId = window.setInterval(updateLoadingEtaText, 180);
    }

    function setGlobalLoading(isLoading) {
        const overlay = document.getElementById("terminal-loading-overlay");
        if (!overlay) {
            return;
        }
        overlay.classList.toggle("is-active", isLoading);
        overlay.setAttribute("aria-hidden", isLoading ? "false" : "true");
    }

    function beginChartRequest() {
        const startedAtMs = Date.now();
        pendingChartRequests += 1;
        if (pendingChartRequests === 1) {
            loadingSessionStartedAtMs = startedAtMs;
            setGlobalLoading(true);
            startLoadingAnimationTimers();
        } else {
            updateLoadingEtaText();
        }
        return startedAtMs;
    }

    function endChartRequest(startedAtMs) {
        if (Number.isFinite(startedAtMs)) {
            const elapsedMs = Math.max(0, Date.now() - Number(startedAtMs));
            recentRequestDurationsMs.push(elapsedMs);
            if (recentRequestDurationsMs.length > MAX_STORED_DURATIONS) {
                recentRequestDurationsMs.splice(0, recentRequestDurationsMs.length - MAX_STORED_DURATIONS);
            }
        }
        pendingChartRequests = Math.max(0, pendingChartRequests - 1);
        if (pendingChartRequests === 0) {
            stopLoadingAnimationTimers();
            setGlobalLoading(false);
            loadingSessionStartedAtMs = 0;
            const etaElement = document.getElementById("terminal-loading-eta");
            if (etaElement) {
                etaElement.textContent = "ETA ~0.0s";
            }
        } else {
            updateLoadingEtaText();
        }
    }

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

    function formatCompactAxisValue(value) {
        if (!Number.isFinite(value)) {
            return "";
        }
        const absoluteValue = Math.abs(value);
        if (absoluteValue >= 1000000000) {
            return (value / 1000000000).toFixed(absoluteValue >= 10000000000 ? 0 : 1).replace(/\.0$/, "") + "B";
        }
        if (absoluteValue >= 1000000) {
            return (value / 1000000).toFixed(absoluteValue >= 10000000 ? 0 : 1).replace(/\.0$/, "") + "M";
        }
        if (absoluteValue >= 1000) {
            return (value / 1000).toFixed(absoluteValue >= 10000 ? 0 : 1).replace(/\.0$/, "") + "k";
        }
        if (Number.isInteger(value)) {
            return String(value);
        }
        return value.toFixed(Math.abs(value) >= 10 ? 0 : 1).replace(/\.0$/, "");
    }

    function formatFullAxisValue(value) {
        if (!Number.isFinite(value)) {
            return "n/a";
        }
        const absoluteValue = Math.abs(value);
        const fractionDigits = absoluteValue >= 1000 ? 0 : absoluteValue >= 10 ? 1 : 2;
        return new Intl.NumberFormat("en-US", {
            maximumFractionDigits: fractionDigits,
        }).format(value);
    }

    /**
     * Formats ECharts time-axis ticks without ISO-8601 "Z" (UTC) suffixes.
     * A bare "Z" can appear when the default formatter is clipped at the chart edge.
     */
    function formatTimeAxisLabel(value) {
        if (value == null || value === "") {
            return "";
        }
        const date = value instanceof Date ? value : new Date(value);
        if (!Number.isFinite(date.getTime())) {
            return "";
        }
        return date.toLocaleDateString(undefined, { month: "short", year: "numeric" });
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function computeNiceStep(rawStep) {
        if (!Number.isFinite(rawStep) || rawStep <= 0) {
            return 1;
        }
        const exponent = Math.floor(Math.log10(rawStep));
        const magnitude = Math.pow(10, exponent);
        const normalized = rawStep / magnitude;
        if (normalized <= 1) {
            return 1 * magnitude;
        }
        if (normalized <= 2) {
            return 2 * magnitude;
        }
        if (normalized <= 2.5) {
            return 2.5 * magnitude;
        }
        if (normalized <= 5) {
            return 5 * magnitude;
        }
        return 10 * magnitude;
    }

    function computeNiceAxisBounds(values, paddingRatio, targetTickCount) {
        const finiteValues = (values || []).filter((value) => Number.isFinite(value));
        if (finiteValues.length === 0) {
            return {
                min: null,
                max: null,
                interval: null,
                splitNumber: targetTickCount,
            };
        }
        const rawMin = Math.min(...finiteValues);
        const rawMax = Math.max(...finiteValues);
        const rawSpan = Math.abs(rawMax - rawMin);
        const referenceSpan = rawSpan > 0 ? rawSpan : Math.max(Math.abs(rawMax), Math.abs(rawMin), 1);
        const paddedMin = rawMin - referenceSpan * paddingRatio;
        const paddedMax = rawMax + referenceSpan * paddingRatio;
        const paddedSpan = Math.max(Math.abs(paddedMax - paddedMin), 1e-9);
        const resolvedTickCount = clamp(targetTickCount, 4, 6);
        const interval = computeNiceStep(paddedSpan / resolvedTickCount);
        const niceMin = Math.floor(paddedMin / interval) * interval;
        const niceMax = Math.ceil(paddedMax / interval) * interval;
        return {
            min: niceMin,
            max: niceMax,
            interval: interval,
            splitNumber: Math.max(1, Math.round((niceMax - niceMin) / interval)),
        };
    }

    function collectLineValues(series, thresholds) {
        const seriesValues = (series || [])
            .flatMap((item) => (item && Array.isArray(item.data) ? item.data : []))
            .map((point) => (Array.isArray(point) ? Number(point[1]) : Number.NaN))
            .filter((value) => Number.isFinite(value));
        const thresholdValues = (thresholds || [])
            .map((threshold) => Number(threshold && threshold.value))
            .filter((value) => Number.isFinite(value));
        return seriesValues.concat(thresholdValues);
    }

    function attachResize(element, instance, lightweight) {
        const MAX_DEFERRED_MEASURE_ATTEMPTS = 12;
        let deferredMeasureAttempts = 0;

        function measuredSize() {
            const width = element.clientWidth;
            const height = element.clientHeight;
            return { width, height };
        }

        function applySize() {
            const size = measuredSize();
            if (size.width < 16 || size.height < 16) {
                if (deferredMeasureAttempts < MAX_DEFERRED_MEASURE_ATTEMPTS) {
                    deferredMeasureAttempts += 1;
                    window.requestAnimationFrame(applySize);
                }
                return;
            }
            deferredMeasureAttempts = 0;
            if (lightweight) {
                instance.applyOptions({ width: size.width, height: size.height });
            } else {
                instance.resize({ width: size.width, height: size.height });
            }
        }

        const observer = new ResizeObserver(() => {
            applySize();
        });
        observer.observe(element);
        applySize();
    }

    function renderLineChart(element, payload) {
        if (!payload.series || payload.series.length === 0) {
            element.innerHTML = '<div class="terminal-empty-state">No chart data available.</div>';
            return;
        }
        resetEchartInstance(element);
        const isMini = element.classList.contains("terminal-chart--mini");
        const containerWidth = Math.max(320, element.clientWidth || 320);
        const axisFontSize = isMini ? (containerWidth <= 720 ? 9 : 10) : containerWidth <= 720 ? 10 : 11;
        const plotHeight = Math.max(120, (element.clientHeight || 0) - 88);
        const targetTickCount = clamp(Math.floor(plotHeight / 44), 4, 6);
        const defaultYAxisPadding = isMini ? 0.02 : 0.06;
        const yAxisPaddingRatio = Number.isFinite(Number(payload.yAxisPaddingRatio))
            ? Number(payload.yAxisPaddingRatio)
            : defaultYAxisPadding;
        const chart = echarts.init(element);
        const markLineData = (payload.thresholds || []).map((threshold) => ({
            yAxis: threshold.value,
            label: isMini
                ? { show: false }
                : {
                    formatter: threshold.label || "",
                    position: "start",
                },
        }));
        const builtSeries = buildEchartsSeries(payload.series);
        const axisBounds = computeNiceAxisBounds(
            collectLineValues(builtSeries, payload.thresholds),
            yAxisPaddingRatio,
            targetTickCount,
        );
        const grid = isMini
            ? { left: 4, right: 6, top: 4, bottom: 4, containLabel: true }
            : { left: 12, right: 12, top: 40, bottom: 28, containLabel: true };
        const legend = isMini
            ? {
                show: true,
                top: 0,
                left: "center",
                padding: [0, 0],
                itemGap: 8,
                itemWidth: 12,
                itemHeight: 10,
                textStyle: { color: "#FFFFFF", fontSize: 12, fontFamily: "JetBrains Mono" },
                selectedMode: true,
                inactiveColor: "#666666",
            }
            : {
                top: 0,
                textStyle: { color: "#FFFFFF" },
            };

        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            tooltip: {
                trigger: "axis",
                valueFormatter: (value) => formatFullAxisValue(Number(value)),
            },
            legend: legend,
            grid: grid,
            xAxis: {
                type: "time",
                axisLine: { lineStyle: { color: "#222222" } },
                axisLabel: {
                    color: "#8A8A8A",
                    fontSize: axisFontSize,
                    margin: isMini ? 4 : 8,
                    hideOverlap: true,
                    formatter: formatTimeAxisLabel,
                },
                splitLine: { lineStyle: { color: "#111111" } },
            },
            yAxis: {
                type: "value",
                scale: true,
                min: axisBounds.min,
                max: axisBounds.max,
                interval: axisBounds.interval,
                splitNumber: axisBounds.splitNumber,
                axisLine: { lineStyle: { color: "#222222" } },
                axisLabel: {
                    color: "#8A8A8A",
                    fontSize: axisFontSize,
                    margin: isMini ? 4 : 8,
                    hideOverlap: true,
                    formatter: (value) => formatCompactAxisValue(Number(value)),
                },
                splitLine: { lineStyle: { color: "#111111" } },
            },
            series: builtSeries.map((item) => ({
                ...item,
                markLine: markLineData.length > 0 ? { symbol: "none", lineStyle: { color: "#444444" }, data: markLineData } : undefined,
            })),
        });
        attachResize(element, chart);
    }

    function renderHeatmap(element, payload) {
        if (!payload.values || payload.values.length === 0) {
            const reason = payload.emptyReason ? " " + payload.emptyReason : "";
            const dropped = payload.droppedLabels && payload.droppedLabels.length > 0
                ? " Dropped: " + payload.droppedLabels.join(", ") + "."
                : "";
            element.innerHTML =
                '<div class="terminal-empty-state">No heatmap data available.' + reason + dropped + "</div>";
            return;
        }
        resetEchartInstance(element);
        const xLabels = payload.xLabels || [];
        const yLabels = payload.yLabels || [];
        const containerWidth = Math.max(320, element.clientWidth || 320);
        const containerHeight = Math.max(220, element.clientHeight || 220);
        const maxXLabelLength = xLabels.reduce((maxLen, label) => Math.max(maxLen, String(label || "").length), 0);
        const maxYLabelLength = yLabels.reduce((maxLen, label) => Math.max(maxLen, String(label || "").length), 0);

        const xAxisMaxChars = containerWidth <= 900 ? 10 : 16;
        const yAxisMaxChars = containerWidth <= 900 ? 12 : 18;
        const xAxisRotate = maxXLabelLength > xAxisMaxChars ? 26 : 0;

        const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
        const dynamicBottom = clamp(
            Math.round(containerHeight * 0.08 + (xAxisRotate > 0 ? 20 : 6)),
            28,
            56,
        );
        const dynamicLeft = clamp(
            Math.round(18 + Math.min(maxYLabelLength, yAxisMaxChars) * 7.2),
            76,
            148,
        );

        const compactLabel = (label, maxChars) => {
            const raw = String(label || "");
            if (raw.length <= maxChars) {
                return raw;
            }
            return raw.slice(0, Math.max(4, maxChars - 3)) + "...";
        };
        const chart = echarts.init(element);
        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            tooltip: {
                confine: true,
                formatter: ({ value }) => {
                    const xIndex = Number(value[0]);
                    const yIndex = Number(value[1]);
                    const corrValue = Number(value[2]);
                    const xLabel = xLabels[xIndex] || "";
                    const yLabel = yLabels[yIndex] || "";
                    return (
                        yLabel + " -> " + xLabel + ": " + corrValue.toFixed(2)
                    );
                },
            },
            grid: { left: dynamicLeft, right: 14, top: 44, bottom: dynamicBottom, containLabel: false },
            xAxis: {
                type: "category",
                data: xLabels,
                splitArea: { show: true },
                axisLabel: {
                    color: "#8A8A8A",
                    rotate: xAxisRotate,
                    margin: xAxisRotate > 0 ? 12 : 8,
                    formatter: (value) => compactLabel(value, xAxisMaxChars),
                },
                axisLine: { lineStyle: { color: "#222222" } },
            },
            yAxis: {
                type: "category",
                data: yLabels,
                splitArea: { show: true },
                axisLabel: {
                    color: "#8A8A8A",
                    margin: 6,
                    formatter: (value) => compactLabel(value, yAxisMaxChars),
                },
                axisLine: { lineStyle: { color: "#222222" } },
            },
            visualMap: {
                min: -1,
                max: 1,
                calculable: false,
                orient: "horizontal",
                left: "center",
                top: 2,
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
            const reason = payload.emptyReason ? " " + payload.emptyReason : "";
            element.innerHTML = '<div class="terminal-empty-state">No bar-chart data available.' + reason + '</div>';
            return;
        }
        resetEchartInstance(element);
        const chart = echarts.init(element);
        const yAxisIsPercent = payload.yAxisFormat === "percent";
        const yAxisLabelFormatter = yAxisIsPercent ? "{value}%" : undefined;
        const showAllCategoryLabels = payload.showAllCategoryLabels === true;
        const hasSecondAxis = (payload.series || []).some((s) => s.yAxisIndex === 1);
        const yAxis = [
            {
                type: "value",
                axisLabel: { color: "#8A8A8A", formatter: yAxisLabelFormatter },
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
            legend: payload.hideLegend ? { show: false } : { top: 0, textStyle: { color: "#FFFFFF" } },
            grid: { left: 56, right: hasSecondAxis ? 56 : 20, top: 36, bottom: showAllCategoryLabels ? 52 : 28 },
            xAxis: {
                type: "category",
                data: payload.categories,
                axisLabel: {
                    color: "#8A8A8A",
                    interval: showAllCategoryLabels ? 0 : "auto",
                    hideOverlap: !showAllCategoryLabels,
                    fontSize: showAllCategoryLabels ? 10 : 12,
                },
                axisLine: { lineStyle: { color: "#222222" } },
            },
            yAxis: yAxis,
            series: (payload.series || []).map((item, index) => {
                const isItemPercent = yAxisIsPercent || item.yAxisIndex === 1;
                return {
                    name: item.name,
                    type: "bar",
                    yAxisIndex: item.yAxisIndex || 0,
                    tooltip: {
                        valueFormatter: (value) => Number(value).toFixed(2) + (isItemPercent ? "%" : "")
                    },
                    data: item.itemColors
                        ? (item.values || []).map((v, i) => ({
                            value: v,
                            itemStyle: { color: (item.itemColors || [])[i] || "#8A8A8A" },
                        }))
                        : item.values,
                    itemStyle: item.itemColors
                        ? undefined
                        : { color: ["#22C55E", "#3B82F6", "#EAB308"][index % 3] },
                };
            }),
        });
        attachResize(element, chart);
    }

    function renderDistributionChart(element, payload) {
        if (!payload.bins || payload.bins.length === 0) {
            element.innerHTML = '<div class="terminal-empty-state">No distribution data available.</div>';
            return;
        }
        resetEchartInstance(element);
        const bins = payload.bins || [];
        const centers = bins.map((bin) => Number(bin.center));
        const markerStyles = {
            "VaR 95": { color: "#F59E0B", type: "solid" },
            "CVaR 95": { color: "#FB923C", type: "solid" },
            "VaR 99": { color: "#EF4444", type: "solid" },
            Mean: { color: "#9CA3AF", type: "dashed" },
        };
        const markerIndexHits = {};
        const markLineData = (payload.markers || [])
            .filter((marker) => marker && Number.isFinite(Number(marker.value)))
            .map((marker) => {
                const value = Number(marker.value);
                let closestIndex = 0;
                let closestDistance = Infinity;
                centers.forEach((center, index) => {
                    const distance = Math.abs(center - value);
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closestIndex = index;
                    }
                });
                const hitCount = markerIndexHits[closestIndex] || 0;
                markerIndexHits[closestIndex] = hitCount + 1;
                const style = markerStyles[marker.label] || { color: "#8A8A8A", type: "solid" };
                return {
                    xAxis: closestIndex,
                    lineStyle: { color: style.color, type: style.type, width: 1.5 },
                    label: {
                        show: true,
                        formatter: marker.label,
                        color: style.color,
                        position: "insideEndTop",
                        distance: 8 + hitCount * 12,
                        fontSize: 10,
                    },
                };
            });

        const chart = echarts.init(element);
        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            tooltip: { trigger: "axis" },
            grid: { left: 48, right: 20, top: 20, bottom: 28 },
            xAxis: {
                type: "category",
                data: bins.map((bin) => bin.label),
                axisLabel: { color: "#8A8A8A", interval: Math.max(0, Math.floor(bins.length / 12)) },
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
                    data: bins.map((bin) => ({
                        value: bin.value,
                        itemStyle: {
                            color: Number(bin.center) < 0 ? "#EF4444" : "#22C55E",
                        },
                    })),
                    markLine: markLineData.length > 0 ? { symbol: "none", data: markLineData } : undefined,
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
                scaleMargins: { top: 0, bottom: 0.87 },
            },
            rightPriceScale: {
                borderColor: "#222222",
            },
            timeScale: {
                borderColor: "#222222",
                timeVisible: true,
                minBarSpacing: 0,
            },
            crosshair: {
                vertLine: { color: "#444444" },
                horzLine: { color: "#444444" },
            },
        });

        let benchmarkSeries = null;
        let minTime = Infinity;
        let maxTime = -Infinity;

        payload.series.forEach((seriesItem) => {
            const isDrawdown = seriesItem.priceScaleId === "drawdown";
            const seriesData = (seriesItem.points || []).map((point) => ({
                time: Math.floor(new Date(point.time).getTime() / 1000),
                value: point.value,
            }));

            let activeSeries;
            if (isDrawdown) {
                activeSeries = chart.addBaselineSeries({
                    baseValue: { type: "price", price: 0 },
                    topLineColor: "transparent",
                    topFillColor1: "transparent",
                    topFillColor2: "transparent",
                    bottomLineColor: "rgba(239, 83, 80, 0.8)",
                    bottomFillColor1: "rgba(239, 83, 80, 0.35)",
                    bottomFillColor2: "rgba(239, 83, 80, 0.05)",
                    lineWidth: 1,
                    title: seriesItem.name,
                    priceScaleId: "left",
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false,
                });
            } else {
                activeSeries = chart.addLineSeries({
                    color: seriesItem.color || "#FFFFFF",
                    lineWidth: seriesItem.lineWidth || 2,
                    title: seriesItem.name,
                    priceScaleId: seriesItem.priceScaleId || "right",
                    priceLineVisible: false,
                    lastValueVisible: false,
                });
            }

            activeSeries.setData(seriesData);
            if (seriesData.length > 0) {
                minTime = Math.min(minTime, seriesData[0].time);
                maxTime = Math.max(maxTime, seriesData[seriesData.length - 1].time);
            }
            if (seriesItem.name === "Benchmark") {
                benchmarkSeries = activeSeries;
            }
        });

        if (hasDrawdown) {
            chart.priceScale("left").applyOptions({
                scaleMargins: { top: 0, bottom: 0.87 },
                borderColor: "#222222",
            });
        }

        attachResize(element, chart, true);

        setTimeout(() => {
            chart.timeScale().fitContent();
            if (minTime !== Infinity && maxTime !== -Infinity) {
                chart.timeScale().setVisibleRange({
                    from: minTime,
                    to: maxTime,
                });
            }
        }, 100);

        const toggleBenchmark = document.getElementById("toggle-benchmark");
        if (toggleBenchmark) {
            if (benchmarkSeries) {
                toggleBenchmark.parentElement.style.display = "";
                // Remove old listener if re-rendering
                const newToggle = toggleBenchmark.cloneNode(true);
                toggleBenchmark.parentNode.replaceChild(newToggle, toggleBenchmark);

                newToggle.addEventListener("change", (e) => {
                    benchmarkSeries.applyOptions({
                        visible: e.target.checked
                    });
                });
                benchmarkSeries.applyOptions({
                    visible: newToggle.checked
                });
            } else {
                toggleBenchmark.parentElement.style.display = "none";
            }
        }
    }

    function renderScatterChart(element, payload) {
        const series = (payload.series || []).filter(
            (item) => item.points && item.points.length > 0,
        );
        if (series.length === 0) {
            const reason = payload.emptyReason ? " " + payload.emptyReason : "";
            element.innerHTML =
                '<div class="terminal-empty-state">No scatter data available.' + reason + "</div>";
            return;
        }
        resetEchartInstance(element);
        const chart = echarts.init(element);
        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            legend: { top: 0, textStyle: { color: "#FFFFFF" } },
            tooltip: {
                trigger: "item",
                formatter: (params) => {
                    const xVal = params.data[0];
                    const yVal = params.data[1];
                    return (
                        params.seriesName +
                        "<br>MAE: $" +
                        formatFullAxisValue(xVal) +
                        "<br>MFE: $" +
                        formatFullAxisValue(yVal)
                    );
                },
            },
            grid: { left: 12, right: 12, top: 40, bottom: 36, containLabel: true },
            xAxis: {
                type: "value",
                name: payload.xAxisLabel || "",
                nameLocation: "middle",
                nameGap: 26,
                inverse: payload.xAxisReversed === true,
                axisLabel: {
                    color: "#8A8A8A",
                    formatter: (value) => formatCompactAxisValue(Number(value)),
                },
                axisLine: { lineStyle: { color: "#222222" } },
                splitLine: { lineStyle: { color: "#111111" } },
            },
            yAxis: {
                type: "value",
                name: payload.yAxisLabel || "",
                nameLocation: "middle",
                nameGap: 44,
                axisLabel: {
                    color: "#8A8A8A",
                    formatter: (value) => formatCompactAxisValue(Number(value)),
                },
                axisLine: { lineStyle: { color: "#222222" } },
                splitLine: { lineStyle: { color: "#111111" } },
            },
            series: [
                ...series.map((item) => ({
                    name: item.name,
                    type: "scatter",
                    symbolSize: 6,
                    itemStyle: { color: item.color || "#8A8A8A", opacity: 0.7 },
                    data: (item.points || []).map((p) => [p.x, p.y]),
                })),
                // Break-even boundary: y = -x (MFE equals adverse excursion).
                // Rendered as a separate line series so ECharts includes it in
                // axis scaling without requiring a secondary coordinate system.
                ...(payload.diagonal
                    ? [
                        {
                            name: "Break-even",
                            type: "line",
                            showSymbol: false,
                            data: [
                                [payload.diagonal.x1, payload.diagonal.y1],
                                [payload.diagonal.x2, payload.diagonal.y2],
                            ],
                            lineStyle: { type: "dashed", color: "#6B7280", width: 1 },
                            itemStyle: { color: "#6B7280" },
                        },
                    ]
                    : []),
            ],
        });
        attachResize(element, chart);
    }

    function renderCategoryLineChart(element, payload) {
        if (!payload.categories || payload.categories.length === 0) {
            const reason = payload.emptyReason ? " " + payload.emptyReason : "";
            element.innerHTML =
                '<div class="terminal-empty-state">No data available.' + reason + "</div>";
            return;
        }
        resetEchartInstance(element);
        const chart = echarts.init(element);
        const containerWidth = Math.max(320, element.clientWidth || 320);
        const axisFontSize = containerWidth <= 720 ? 10 : 11;
        const plotHeight = Math.max(120, (element.clientHeight || 0) - 88);
        const targetTickCount = clamp(Math.floor(plotHeight / 44), 4, 6);
        const markLineData = (payload.thresholds || [])
            .filter((t) => t && Number.isFinite(Number(t.value)))
            .map((t) => ({
                yAxis: Number(t.value),
                lineStyle: { type: "dashed", color: t.color || "#8A8A8A", width: 1.5 },
                label: { formatter: t.label || "", color: t.color || "#8A8A8A", fontSize: 10 },
            }));
        const verticalMarkLineData = (payload.verticalMarkers || [])
            .filter((marker) => marker && marker.category)
            .map((marker) => ({
                xAxis: marker.category,
                lineStyle: { type: "dashed", color: marker.color || "#F59E0B", width: 1.5 },
                label: {
                    formatter: marker.label || "Time Stop Hold",
                    color: marker.color || "#F59E0B",
                    fontSize: 10,
                },
            }));
        const yValues = [];
        (payload.series || []).forEach((item) => {
            (item.values || []).forEach((value) => {
                const parsed = Number(value);
                if (Number.isFinite(parsed)) {
                    yValues.push(parsed);
                }
            });
        });
        (payload.thresholds || []).forEach((threshold) => {
            const parsed = Number(threshold && threshold.value);
            if (Number.isFinite(parsed)) {
                yValues.push(parsed);
            }
        });
        const axisBounds = computeNiceAxisBounds(yValues, 0.06, targetTickCount);
        const mainSeries = (payload.series || []).map((item, index) => ({
            name: item.name,
            type: "line",
            data: item.values,
            symbol: "circle",
            symbolSize: 7,
            lineStyle: { color: item.color || "#FFFFFF", width: 2 },
            itemStyle: { color: item.color || "#FFFFFF" },
            markLine: undefined,
        }));
        const helperLegendSeries = [];
        (payload.thresholds || []).forEach((t, index) => {
            if (!t || !markLineData[index]) {
                return;
            }
            helperLegendSeries.push({
                name: t.legend || t.label || "Threshold",
                type: "line",
                data: (payload.categories || []).map(() => null),
                symbol: "none",
                lineStyle: { type: "dashed", color: t.color || "#8A8A8A", width: 1.5 },
                itemStyle: { color: t.color || "#8A8A8A" },
                tooltip: { show: false },
                markLine: { symbol: "none", data: [markLineData[index]] },
            });
        });
        (payload.verticalMarkers || []).forEach((marker, index) => {
            if (!marker || !verticalMarkLineData[index]) {
                return;
            }
            helperLegendSeries.push({
                name: marker.legend || marker.label || "Vertical Marker",
                type: "line",
                data: (payload.categories || []).map(() => null),
                symbol: "none",
                lineStyle: { type: "dashed", color: marker.color || "#F59E0B", width: 1.5 },
                itemStyle: { color: marker.color || "#F59E0B" },
                tooltip: { show: false },
                markLine: { symbol: "none", data: [verticalMarkLineData[index]] },
            });
        });

        const mainSeriesByLegendName = new Map();
        (payload.series || []).forEach((item) => {
            mainSeriesByLegendName.set(item.name, item);
        });
        const thresholdByLegendName = new Map();
        (payload.thresholds || []).forEach((threshold) => {
            if (!threshold) {
                return;
            }
            const legendName = threshold.legend || threshold.label || "Threshold";
            thresholdByLegendName.set(legendName, threshold);
        });

        function updateCategoryLineAxisFromLegend(selected) {
            const activeValues = [];

            mainSeriesByLegendName.forEach((seriesItem, legendName) => {
                if (!selected || selected[legendName] !== false) {
                    (seriesItem.values || []).forEach((value) => {
                        const parsed = Number(value);
                        if (Number.isFinite(parsed)) {
                            activeValues.push(parsed);
                        }
                    });
                }
            });

            thresholdByLegendName.forEach((threshold, legendName) => {
                if (!selected || selected[legendName] !== false) {
                    const parsed = Number(threshold && threshold.value);
                    if (Number.isFinite(parsed)) {
                        activeValues.push(parsed);
                    }
                }
            });

            const bounds = computeNiceAxisBounds(activeValues, 0.06, targetTickCount);
            chart.setOption({
                yAxis: {
                    min: bounds.min,
                    max: bounds.max,
                    interval: bounds.interval,
                    splitNumber: bounds.splitNumber,
                },
            });
        }

        chart.setOption({
            backgroundColor: "#0D0D0D",
            animation: false,
            textStyle: { color: "#FFFFFF", fontFamily: "JetBrains Mono" },
            tooltip: {
                trigger: "axis",
                valueFormatter: (value) => "$" + formatFullAxisValue(Number(value)),
            },
            legend: { top: 0, textStyle: { color: "#FFFFFF" } },
            grid: { left: 12, right: 12, top: 40, bottom: 28, containLabel: true },
            xAxis: {
                type: "category",
                data: payload.categories,
                axisLabel: { color: "#8A8A8A", fontSize: axisFontSize },
                axisLine: { lineStyle: { color: "#222222" } },
            },
            yAxis: {
                type: "value",
                scale: true,
                min: axisBounds.min,
                max: axisBounds.max,
                interval: axisBounds.interval,
                splitNumber: axisBounds.splitNumber,
                axisLabel: {
                    color: "#8A8A8A",
                    fontSize: axisFontSize,
                    hideOverlap: true,
                    formatter: (value) => formatCompactAxisValue(Number(value)),
                },
                axisLine: { lineStyle: { color: "#222222" } },
                splitLine: { lineStyle: { color: "#111111" } },
            },
            series: mainSeries.concat(helperLegendSeries),
        });
        chart.on("legendselectchanged", (event) => {
            updateCategoryLineAxisFromLegend(event.selected || {});
        });
        attachResize(element, chart);
    }

    function renderChart(element) {
        const endpoint = element.dataset.chartEndpoint;
        const renderer = element.dataset.chartRenderer;
        if (!endpoint || !renderer) {
            return;
        }
        let resolvedEndpoint = endpoint;
        if (element.dataset.decompositionLinked === "true") {
            const sortInput = document.getElementById("decomposition-sort-by-input");
            const sortBy = sortInput ? (sortInput.value || "").trim() : "";
            if (sortBy) {
                const separator = endpoint.includes("?") ? "&" : "?";
                resolvedEndpoint = endpoint + separator + "sort_by=" + encodeURIComponent(sortBy);
            }
        }
        const requestStartedAtMs = beginChartRequest();
        fetch(resolvedEndpoint, { headers: { Accept: "application/json" } })
            .then((response) => {
                if (!response.ok) {
                    throw new Error("HTTP " + String(response.status));
                }
                return response.json();
            })
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
                if (renderer === "scatter") {
                    renderScatterChart(element, payload);
                    return;
                }
                if (renderer === "category_line") {
                    renderCategoryLineChart(element, payload);
                    return;
                }
                renderLineChart(element, payload);
            })
            .catch(() => {
                element.innerHTML = '<div class="terminal-empty-state">Chart request failed.</div>';
            })
            .finally(() => {
                endChartRequest(requestStartedAtMs);
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
                let updated = current;
                try {
                    const absoluteUrl = new URL(current, window.location.origin);
                    absoluteUrl.searchParams.set("horizon", horizon);
                    updated = absoluteUrl.pathname + absoluteUrl.search;
                } catch (_) {
                    updated = current.includes("horizon=")
                        ? current.replace(/horizon=[^&]+/, "horizon=" + horizon)
                        : current + (current.includes("?") ? "&" : "?") + "horizon=" + horizon;
                }
                el.dataset.chartEndpoint = updated;
                el.innerHTML = "";
                renderChart(el);
            });
            const sidebarSelect = document.querySelector('#dashboard-filters [name="correlation_horizon"]');
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
