(() => {
  // Chainlit 2.11.1 constructs ClipboardItem before checking whether a browser
  // supports rich clipboard writes. Safari versions without ClipboardItem can
  // still copy plain text with writeText(), so make Chainlit take that path.
  if (typeof window.ClipboardItem === "undefined") {
    window.ClipboardItem = class ClipboardItemFallback {
      constructor(items) {
        this.items = items;
      }
    };

    const clipboard = navigator.clipboard;
    if (clipboard?.writeText) {
      try {
        Object.defineProperty(clipboard, "write", {
          configurable: true,
          value: undefined,
        });
      } catch {
        // Browsers that expose a non-configurable `write` will keep their
        // native behavior; browsers needing this workaround use writeText.
      }
    }
  }

  const dialogSelector = "#chat-settings";
  const label = "Restore config defaults and accept";

  function customizeResetButton() {
    const dialog = document.querySelector(dialogSelector);
    if (!dialog) return;

    const acceptButton = dialog.querySelector("#confirm, #confirm-sidebar");
    const resetButton = acceptButton?.parentElement?.querySelector("button");
    if (!resetButton || resetButton.dataset.configDefaultsCustomized) return;

    resetButton.dataset.configDefaultsCustomized = "true";
    resetButton.textContent = label;
    resetButton.setAttribute("aria-label", label);
    resetButton.title = label;

    resetButton.addEventListener("click", () => {
      // Let Chainlit reset its form to initial values, then accept those values.
      window.setTimeout(() => {
        const activeDialog = document.querySelector(dialogSelector);
        const activeAcceptButton = activeDialog?.querySelector(
          "#confirm, #confirm-sidebar"
        );
        activeAcceptButton?.click();
      }, 0);
    });
  }

  new MutationObserver(customizeResetButton).observe(document.documentElement, {
    childList: true,
    subtree: true,
  });
  customizeResetButton();
})();
