(() => {
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
