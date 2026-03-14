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

        root.querySelectorAll(".terminal-tab").forEach((button) => {
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
            let startY = 0;
            let startHeight = 0;
            bottomHandle.addEventListener("mousedown", function (e) {
                startY = e.clientY;
                startHeight = parseInt(getComputedStyle(root).getPropertyValue("--bottom-height")) || 280;
                bottomHandle.classList.add("is-dragging");
                document.body.style.cursor = "row-resize";
                document.body.style.userSelect = "none";

                function onMove(ev) {
                    const dy = startY - ev.clientY;
                    const newHeight = Math.max(140, Math.min(window.innerHeight * 0.55, startHeight + dy));
                    root.style.setProperty("--bottom-height", newHeight + "px");
                    termMain.style.gridTemplateRows = "auto minmax(0, 1fr) 5px " + newHeight + "px";
                }

                function onUp() {
                    bottomHandle.classList.remove("is-dragging");
                    document.body.style.cursor = "";
                    document.body.style.userSelect = "";
                    document.removeEventListener("mousemove", onMove);
                    document.removeEventListener("mouseup", onUp);
                }

                document.addEventListener("mousemove", onMove);
                document.addEventListener("mouseup", onUp);
            });
        }
    }

    function initRoot(root) {
        wireTabs(document);
        wireTabJumpButtons(document);
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
