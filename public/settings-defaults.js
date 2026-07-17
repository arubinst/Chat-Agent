(() => {
  // Chainlit 2.11.1 constructs ClipboardItem before checking whether a browser
  // supports rich clipboard writes. Supply a plain-text fallback for browsers
  // without the modern Clipboard API (for example, an HTTP-served page).
  if (typeof window.ClipboardItem === "undefined") {
    window.ClipboardItem = class ClipboardItemFallback {
      constructor(items) {
        this.items = items;
      }
    };

    const copyTextWithSelection = (text) =>
      new Promise((resolve, reject) => {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.setAttribute("readonly", "");
        textarea.style.cssText =
          "position:fixed;top:0;left:0;opacity:0;pointer-events:none;";
        document.body.appendChild(textarea);
        textarea.select();

        try {
          const copied = document.execCommand("copy");
          textarea.remove();
          copied ? resolve() : reject(new Error("Clipboard access was denied"));
        } catch (error) {
          textarea.remove();
          reject(error);
        }
      });

    let clipboard = navigator.clipboard;
    if (!clipboard) {
      clipboard = { writeText: copyTextWithSelection };
      try {
        Object.defineProperty(navigator, "clipboard", {
          configurable: true,
          value: clipboard,
        });
      } catch {
        // Chainlit will report an error when the browser forbids both the
        // modern Clipboard API and this compatibility property.
        return;
      }
    }

    if (!clipboard.writeText) clipboard.writeText = copyTextWithSelection;

    try {
      Object.defineProperty(clipboard, "write", {
        configurable: true,
        value: undefined,
      });
    } catch {
      // If `write` cannot be replaced, it may still be safely unsupported.
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
