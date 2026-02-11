// Heartbeat mechanism to keep server alive while page is active
let heartbeatInterval;
let isRefreshing = false;

function sendHeartbeat() {
  fetch("/heartbeat", { method: "POST", keepalive: true }).catch(() => {});
}

function startHeartbeat() {
  sendHeartbeat();
  heartbeatInterval = setInterval(sendHeartbeat, 5000);
}

function stopHeartbeat() {
  if (heartbeatInterval) {
    clearInterval(heartbeatInterval);
  }
}

window.addEventListener("load", startHeartbeat);

window.addEventListener("beforeunload", () => {
  if (performance.navigation.type === 1) {
    isRefreshing = true;
  }

  if (!isRefreshing) {
    navigator.sendBeacon("/goodbye");
  }
});

const navEntry = performance.getEntriesByType("navigation")[0];
if (navEntry && navEntry.type === "reload") {
  isRefreshing = true;
}
