/* This PC or another: controls only a browser on the PC that renders can use, sharing this PC, and rendering on another. */
"use strict";

/* Some controls open a Windows dialog or folder on the PC that renders, so they only show in a browser on that PC. */
let local = true;
function isLocal() { return local; }

function showLocal(here) {
  if (here === local && document.body.dataset.place) return;
  local = here;
  document.body.dataset.place = here ? "local" : "remote";
  document.querySelectorAll(".local-only").forEach((el) => { el.hidden = !here; });
  // Another PC can choose model folders over there, through the page's own browser. Where images are saved is
  // set on the PC that renders.
  $("outputDir").readOnly = !here;
  for (const id of ["outputBrowse", "openFolder"]) $(id).hidden = !here;
  showCheckpoints();
  loadNetwork();
}

// ---- network ----
/* Share this PC lets a browser on another PC render here. That browser gives a code once. Render on another PC
   opens the other PC's Nano in this browser. */
async function loadNetwork() {
  if (!isLocal()) return;
  try { showNetwork(await api("/api/network")); } catch { /* the server is restarting */ }
}

function showNetwork(n) {
  $("shareToggle").checked = n.on;
  $("shareInfo").hidden = !n.on || !n.addresses.length;
  $("shareAddress").textContent = n.addresses.map((a) => `http://${a}:${n.port}`).join("  or  ");
  $("shareCode").textContent = n.code ? `${n.code.slice(0, 4)}-${n.code.slice(4)}` : "";
  $("shareError").textContent = n.error || "";
  $("shareError").hidden = !n.error;
}

async function setSharing(body) {
  $("shareToggle").disabled = true;
  try { showNetwork(await post("/api/network", body)); } catch (e) { $("shareError").textContent = e.message; $("shareError").hidden = false; }
  $("shareToggle").disabled = false;
}

function openRemote(e) {
  e.preventDefault();
  let address = $("remoteAddress").value.trim();
  if (!address) { $("remoteAddress").focus(); return; }
  if (!/^https?:\/\//i.test(address)) address = `http://${address}`;
  const url = new URL(address);
  if (!url.port) url.port = "7861";
  state.remote_address = $("remoteAddress").value.trim();
  save();
  location.href = url.origin + "/";
}

/* In a browser on another PC, the menu bar says whose GPU it is using. */
function showRemote(s) {
  if (s.version) $("aboutVersion").textContent = `Version ${s.version}.`;
  const remote = s.local === false;
  $("remoteBadge").hidden = !remote;
  $("backHome").hidden = !remote;
  if (remote) {
    $("remoteBadge").textContent = `Rendering on ${s.machine || location.hostname}`;
    $("remoteBadge").title = `This page runs on ${s.machine || location.hostname} at ${location.host}. Renders use its GPU, and images are saved there.`;
  }
}

function setUpNetwork() {
  $("shareToggle").addEventListener("change", () => setSharing({ on: $("shareToggle").checked }));
  $("shareNewCode").addEventListener("click", () => setSharing({ new_code: true }));
  $("remoteForm").addEventListener("submit", openRemote);
  $("remoteAddress").value = state.remote_address || "";
}
