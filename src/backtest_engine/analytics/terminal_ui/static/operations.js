(function () {
    const TerminalUI = (window.TerminalUI = window.TerminalUI || {});
    const jobEventSources = {};
    let workerRefreshTimer = null;

    function syncInitialProgress(card) {
        const bar = card.querySelector('[data-job-field="progress_bar"]');
        if (!bar) {
            return;
        }
        const progressPct = Number(bar.dataset.progressPct || 0);
        bar.style.width = String(progressPct) + "%";
    }

    function updateJobCard(card, payload) {
        const status = card.querySelector('[data-job-field="status"]');
        const stage = card.querySelector('[data-job-field="progress_stage_label"]');
        const current = card.querySelector('[data-job-field="progress_current"]');
        const total = card.querySelector('[data-job-field="progress_total"]');
        const bar = card.querySelector('[data-job-field="progress_bar"]');
        const message = card.querySelector('[data-job-field="progress_message"]');
        const duration = card.querySelector('[data-job-field="duration_seconds"]');
        const output = card.querySelector('[data-job-field="output_artifact_path"]');
        const error = card.querySelector('[data-job-field="last_error"]');

        if (status) {
            status.textContent = payload.status || "unknown";
        }
        if (stage) {
            stage.textContent = payload.progress_stage_label || "Waiting for worker";
        }
        if (current) {
            current.textContent = String(payload.progress_current || 0);
        }
        if (total) {
            total.textContent = String(payload.progress_total || 0);
        }
        if (bar) {
            bar.style.width = String(payload.progress_pct || 0) + "%";
        }
        if (message) {
            message.textContent = payload.progress_message || "Waiting for worker.";
        }
        if (duration) {
            duration.textContent =
                payload.duration_seconds === null || payload.duration_seconds === undefined
                    ? "N/A"
                    : Number(payload.duration_seconds).toFixed(1) + "s";
        }
        if (output) {
            output.textContent = payload.output_artifact_path || "Pending";
        }
        if (error) {
            error.textContent = payload.last_error || "None";
        }
    }

    function closeJobStream(jobId) {
        const existing = jobEventSources[jobId];
        if (!existing) {
            return;
        }
        existing.source.close();
        delete jobEventSources[jobId];
    }

    const AUTO_REFRESH_STATES = ["worker_starting", "redis_starting", "install_running"];

    function scheduleAutoRefresh() {
        if (workerRefreshTimer) {
            window.clearTimeout(workerRefreshTimer);
            workerRefreshTimer = null;
        }
        const panel = document.querySelector('[data-worker-readiness-state]');
        if (!panel) {
            return;
        }
        const state = panel.dataset.workerReadinessState || "";
        if (!AUTO_REFRESH_STATES.includes(state)) {
            return;
        }
        const refreshMs = Number(panel.dataset.workerRefreshMs || 2000);
        const delayMs = Number.isFinite(refreshMs) && refreshMs > 0 ? refreshMs : 2000;
        workerRefreshTimer = window.setTimeout(function () {
            workerRefreshTimer = null;
            document.body.dispatchEvent(new Event("dashboard-tab-change"));
        }, delayMs);
    }

    TerminalUI.initOperations = function () {
        const cards = document.querySelectorAll(".terminal-job-card[data-job-stream-url]");
        cards.forEach((card) => {
            const jobId = card.dataset.jobId || "";
            const streamUrl = card.dataset.jobStreamUrl || "";
            syncInitialProgress(card);
            if (!jobId || !streamUrl || typeof window.EventSource === "undefined") {
                return;
            }

            const existing = jobEventSources[jobId];
            if (existing && existing.card === card && existing.url === streamUrl) {
                return;
            }
            closeJobStream(jobId);

            const source = new EventSource(streamUrl);
            jobEventSources[jobId] = { source, card, url: streamUrl };
            source.addEventListener("status", function (event) {
                const payload = JSON.parse(event.data);
                updateJobCard(card, payload);
                if (["completed", "failed", "timeout"].includes(payload.status || "")) {
                    closeJobStream(jobId);
                    document.body.dispatchEvent(new Event("dashboard-tab-change"));
                }
            });
            source.onerror = function () {
                closeJobStream(jobId);
            };
        });

        Object.keys(jobEventSources).forEach((jobId) => {
            const match = document.querySelector(
                '.terminal-job-card[data-job-id="' + jobId + '"]',
            );
            if (!match) {
                closeJobStream(jobId);
            }
        });
        scheduleAutoRefresh();
    };
})();
