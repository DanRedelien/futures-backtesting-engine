(function () {
    const TerminalUI = (window.TerminalUI = window.TerminalUI || {});
    const jobEventSources = {};

    function syncInitialProgress(card) {
        const bar = card.querySelector('[data-job-field="progress_bar"]');
        if (!bar) {
            return;
        }
        const progressPct = Number(bar.dataset.progressPct || 0);
        bar.style.width = String(progressPct) + "%";
    }

    function setSelectedJobId(jobId) {
        const input = document.getElementById("selected-job-id-input");
        if (input) {
            input.value = jobId || "";
        }
    }

    function updateJobCard(card, payload) {
        const status = card.querySelector('[data-job-field="status"]');
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

    TerminalUI.initOperations = function () {
        const cards = document.querySelectorAll(".terminal-job-card[data-job-stream-url]");
        cards.forEach((card) => {
            const jobId = card.dataset.selectedJobId || "";
            const streamUrl = card.dataset.jobStreamUrl || "";
            syncInitialProgress(card);
            if (!jobId || !streamUrl || typeof window.EventSource === "undefined") {
                return;
            }

            setSelectedJobId(jobId);

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
                '.terminal-job-card[data-selected-job-id="' + jobId + '"]',
            );
            if (!match) {
                closeJobStream(jobId);
            }
        });
    };
})();
