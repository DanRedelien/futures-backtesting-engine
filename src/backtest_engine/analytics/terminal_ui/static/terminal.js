(function () {
    const TerminalUI = (window.TerminalUI = window.TerminalUI || {});

    function setActiveTab(button) {
        document.querySelectorAll(".terminal-tab").forEach((item) => {
            item.classList.toggle("is-active", item === button);
        });
    }

    function activateTab(tabId) {
        const tabInput = document.getElementById("dashboard-tab-input");
        if (!tabInput) {
            return;
        }
        const button = document.querySelector('.terminal-tab[data-tab="' + tabId + '"]');
        tabInput.value = tabId;
        if (button) {
            setActiveTab(button);
        }
        document.body.dispatchEvent(new Event("dashboard-tab-change"));
    }

    function wireTabs(root) {
        const tabInput = document.getElementById("dashboard-tab-input");
        if (!tabInput) {
            return;
        }

        // Only bind buttons that carry an explicit data-tab attribute.
        // Secondary sub-view buttons inside panels use data-exit-detail-view
        // instead; including them here would call activateTab("") and navigate
        // the whole bottom panel away from Exit Analysis.
        root.querySelectorAll(".terminal-tab[data-tab]").forEach((button) => {
            if (button.dataset.terminalTabBound === "true") {
                return;
            }
            button.dataset.terminalTabBound = "true";
            button.addEventListener("click", function () {
                activateTab(button.dataset.tab || "");
            });
        });
    }

    function wireTabJumpButtons(root) {
        root.querySelectorAll("[data-tab-jump]").forEach((button) => {
            if (button.dataset.terminalTabJumpBound === "true") {
                return;
            }
            button.dataset.terminalTabJumpBound = "true";
            button.addEventListener("click", function () {
                activateTab(button.dataset.tabJump || "");
            });
        });
    }

    function wireTableSort(root) {
        root.querySelectorAll(".terminal-table-sort[data-sort-column]").forEach((button) => {
            if (button.dataset.terminalSortBound === "true") {
                return;
            }
            button.dataset.terminalSortBound = "true";
            button.addEventListener("click", function () {
                const sortColumn = button.dataset.sortColumn || "";
                const sortInputId = button.dataset.sortInputId || "";
                const sortTab = button.dataset.sortTab || "";
                if (!sortColumn || !sortInputId || !sortTab) {
                    return;
                }
                const sortInput = document.getElementById(sortInputId);
                if (!sortInput) {
                    return;
                }
                sortInput.value = sortColumn;
                activateTab(sortTab);
            });
        });
    }

    function wireRiskMetricControls(root) {
        void root;
    }

    function wireExitDetailSubViews(root) {
        root.querySelectorAll("[data-exit-detail-view]").forEach((button) => {
            if (button.dataset.terminalExitDetailBound === "true") {
                return;
            }
            button.dataset.terminalExitDetailBound = "true";
            button.addEventListener("click", function () {
                const view = button.dataset.exitDetailView || "trade-log";
                const viewInput = document.getElementById("exit-detail-view-input");
                if (viewInput) {
                    viewInput.value = view;
                }
                const nav = button.closest("nav");
                if (nav) {
                    nav.querySelectorAll(".terminal-tab").forEach((tab) => {
                        tab.classList.toggle("is-active", tab === button);
                    });
                }
                document.body.dispatchEvent(new Event("exit-detail-change"));
            });
        });
    }

    function wireExitSummaryRows(root) {
        root.querySelectorAll("tr[data-exit-strategy]").forEach((row) => {
            if (row.dataset.terminalExitStrategyBound === "true") {
                return;
            }
            row.dataset.terminalExitStrategyBound = "true";
            row.addEventListener("click", function () {
                const strategy = row.dataset.exitStrategy || "__all__";
                // Programmatic assignment does not fire a 'change' event, so the
                // full bottom-panel refresh is not triggered — only the workspace
                // detail fires via exit-detail-change below.
                const strategySelect = document.querySelector(
                    '#dashboard-filters [name="exit_strategy"]'
                );
                if (strategySelect) {
                    strategySelect.value = strategy;
                }
                // Mark selected row and clear others in the same tbody.
                const tbody = row.closest("tbody");
                if (tbody) {
                    tbody.querySelectorAll("tr.terminal-table__row--clickable").forEach((r) => {
                        r.classList.toggle("is-selected", r === row);
                    });
                }
                document.body.dispatchEvent(new Event("exit-detail-change"));
            });
        });
    }

    function initResize() {
        const root = document.documentElement;

        const sidebarHandle = document.getElementById("resize-sidebar");
        if (sidebarHandle) {
            let startX = 0;
            let startWidth = 0;
            sidebarHandle.addEventListener("mousedown", function (e) {
                startX = e.clientX;
                startWidth = parseInt(getComputedStyle(root).getPropertyValue("--sidebar-width")) || 320;
                sidebarHandle.classList.add("is-dragging");
                document.body.style.cursor = "col-resize";
                document.body.style.userSelect = "none";

                function onMove(ev) {
                    const newWidth = Math.max(180, Math.min(560, startWidth + ev.clientX - startX));
                    root.style.setProperty("--sidebar-width", newWidth + "px");
                }

                function onUp() {
                    sidebarHandle.classList.remove("is-dragging");
                    document.body.style.cursor = "";
                    document.body.style.userSelect = "";
                    document.removeEventListener("mousemove", onMove);
                    document.removeEventListener("mouseup", onUp);
                }

                document.addEventListener("mousemove", onMove);
                document.addEventListener("mouseup", onUp);
            });
        }

        const bottomHandle = document.getElementById("resize-bottom");
        const termMain = document.querySelector(".terminal-main");
        if (bottomHandle && termMain) {
            const COLLAPSE_SNAP_PX = 72;
            const MIN_EXPANDED_HEIGHT = 140;
            const TOP_HANDLE_CLEARANCE_PX = 4;
            let startY = 0;
            let startHeight = 0;
            let lastExpandedHeight = parseInt(getComputedStyle(root).getPropertyValue("--bottom-height")) || 280;

            function currentBottomHeight() {
                return parseInt(getComputedStyle(root).getPropertyValue("--bottom-height")) || 0;
            }

            function splitThresholdHeight() {
                const mainHeight = termMain.clientHeight || window.innerHeight || 0;
                return Math.max(0, Math.floor(mainHeight / 3));
            }

            function maxExpandedHeight() {
                const mainHeight = termMain.clientHeight || window.innerHeight || 0;
                return Math.max(
                    MIN_EXPANDED_HEIGHT,
                    Math.floor(mainHeight - TOP_HANDLE_CLEARANCE_PX),
                );
            }

            function applyBottomHeight(rawHeight) {
                const maxHeight = maxExpandedHeight();
                const height = Math.max(0, Math.min(maxHeight, Math.round(rawHeight)));
                const collapsed = height === 0;
                const reserved = Math.min(height, splitThresholdHeight());
                root.style.setProperty("--bottom-height", height + "px");
                root.style.setProperty("--bottom-reserved", reserved + "px");
                termMain.classList.toggle("is-bottom-collapsed", collapsed);
                bottomHandle.classList.toggle("is-collapsed", collapsed);
                if (!collapsed) {
                    lastExpandedHeight = height;
                }
            }

            function restoreBottomPanel() {
                applyBottomHeight(Math.max(MIN_EXPANDED_HEIGHT, lastExpandedHeight || 280));
            }

            function collapseBottomPanel() {
                applyBottomHeight(0);
            }

            function toggleBottomPanel() {
                if (currentBottomHeight() === 0) {
                    restoreBottomPanel();
                    return;
                }
                collapseBottomPanel();
            }

            applyBottomHeight(currentBottomHeight());

            bottomHandle.addEventListener("mousedown", function (e) {
                startY = e.clientY;
                startHeight = currentBottomHeight();
                bottomHandle.classList.add("is-dragging");
                document.body.style.cursor = "row-resize";
                document.body.style.userSelect = "none";

                function onMove(ev) {
                    const dy = startY - ev.clientY;
                    const rawHeight = startHeight + dy;
                    if (rawHeight <= COLLAPSE_SNAP_PX) {
                        applyBottomHeight(0);
                        return;
                    }
                    applyBottomHeight(rawHeight);
                }

                function onUp() {
                    const settledHeight = currentBottomHeight();
                    if (settledHeight > 0 && settledHeight < MIN_EXPANDED_HEIGHT) {
                        applyBottomHeight(MIN_EXPANDED_HEIGHT);
                    }
                    bottomHandle.classList.remove("is-dragging");
                    document.body.style.cursor = "";
                    document.body.style.userSelect = "";
                    document.removeEventListener("mousemove", onMove);
                    document.removeEventListener("mouseup", onUp);
                }

                document.addEventListener("mousemove", onMove);
                document.addEventListener("mouseup", onUp);
            });

            bottomHandle.addEventListener("dblclick", function () {
                toggleBottomPanel();
            });

            window.addEventListener("resize", function () {
                const height = currentBottomHeight();
                if (height === 0) {
                    collapseBottomPanel();
                    return;
                }
                const clamped = Math.min(height, maxExpandedHeight());
                applyBottomHeight(clamped);
            });
        }
    }

    function initRoot(root) {
        wireTabs(document);
        wireTabJumpButtons(document);
        wireTableSort(root);
        wireRiskMetricControls(root);
        wireExitDetailSubViews(root);
        wireExitSummaryRows(root);
        if (typeof TerminalUI.initCharts === "function") {
            TerminalUI.initCharts(root);
        }
        if (typeof TerminalUI.initOperations === "function") {
            TerminalUI.initOperations(root);
        }
    }

    TerminalUI.activateTab = activateTab;
    TerminalUI.wireTabs = wireTabs;
    TerminalUI.wireTabJumpButtons = wireTabJumpButtons;
    TerminalUI.wireTableSort = wireTableSort;
    TerminalUI.wireRiskMetricControls = wireRiskMetricControls;
    TerminalUI.wireExitDetailSubViews = wireExitDetailSubViews;
    TerminalUI.wireExitSummaryRows = wireExitSummaryRows;
    TerminalUI.initResize = initResize;

    document.addEventListener("DOMContentLoaded", function () {
        initRoot(document);
        initResize();
    });

    if (window.htmx) {
        window.htmx.onLoad(function (root) {
            initRoot(root);
        });
    }
})();
