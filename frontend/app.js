(() => {
  // API base:
  // - if served by FastAPI (http://127.0.0.1:8000/frontend/index.html) => use same origin
  // - if opened via file:// => fall back to local API
  const origin = window.location.origin || "";
  const API_BASE = (origin.startsWith("http")) ? origin : "http://127.0.0.1:8000";

  // Safe set API label (avoid Firefox null errors)
  const apiLabel = document.getElementById("apiLabel");
  if (apiLabel) apiLabel.textContent = API_BASE;

  // Helpers
  const $ = (id) => document.getElementById(id);

  function pct(x) {
    if (typeof x !== "number" || Number.isNaN(x)) return 0;
    return Math.max(0, Math.min(100, Math.round(x * 100)));
  }

  function verdictClass(verdict) {
    const v = (verdict || "").toLowerCase();
    if (v.includes("legit") || v.includes("safe")) return "ok";
    if (v.includes("phish") || v.includes("mal") || v.includes("susp")) return "bad";
    return "";
  }

  function showAlert(el, verdict, confidence) {
    const c = verdictClass(verdict);
    el.className = `alert ${c}`.trim();
    const confPct = (typeof confidence === "number") ? `${Math.round(confidence * 100)}%` : "—";
    el.innerHTML = `<b>${String(verdict || "Result").toUpperCase()}</b> • Confidence: <b>${confPct}</b>`;
    el.classList.remove("hidden");
  }

  function hideAlert(el) {
    el.classList.add("hidden");
    el.textContent = "";
  }

  function setMeters(metersEl, legitBar, phishBar, legitVal, phishVal, legitProb, phishProb) {
    const l = pct(legitProb);
    const p = pct(phishProb);

    legitBar.style.width = `${l}%`;
    phishBar.style.width = `${p}%`;

    legitVal.textContent = `${l}%`;
    phishVal.textContent = `${p}%`;

    metersEl.classList.remove("hidden");
  }

  function pretty(obj) {
    try {
      return JSON.stringify(obj, null, 2);
    } catch {
      return String(obj);
    }
  }

  async function postJSON(path, payload) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    // If backend returns HTML error page, this prevents JSON.parse crash
    const text = await res.text();
    try {
      const json = JSON.parse(text);
      if (!res.ok) throw new Error(json.detail || `HTTP ${res.status}`);
      return json;
    } catch (e) {
      // Not JSON
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${text.slice(0, 120)}`);
      throw new Error(`Unexpected response: ${text.slice(0, 120)}`);
    }
  }

  async function getJSON(path) {
    const res = await fetch(`${API_BASE}${path}`);
    const text = await res.text();
    try {
      const json = JSON.parse(text);
      if (!res.ok) throw new Error(json.detail || `HTTP ${res.status}`);
      return json;
    } catch (e) {
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${text.slice(0, 120)}`);
      throw new Error(`Unexpected response: ${text.slice(0, 120)}`);
    }
  }

  // URL Scan
  const urlBtn = $("urlBtn");
  urlBtn?.addEventListener("click", async () => {
    const url = ($("urlInput")?.value || "").trim();
    const alertEl = $("urlAlert");
    const metersEl = $("urlMeters");
    const indicatorsEl = $("urlIndicators");

    hideAlert(alertEl);
    metersEl?.classList.add("hidden");
    if (indicatorsEl) indicatorsEl.textContent = "";

    if (!url) {
      alertEl.className = "alert bad";
      alertEl.textContent = "Please enter a URL.";
      alertEl.classList.remove("hidden");
      return;
    }

    urlBtn.disabled = true;
    urlBtn.textContent = "SCANNING...";

    try {
      const data = await postJSON("/predict/url", { url });

      showAlert(alertEl, data.verdict, data.confidence);

      setMeters(
        metersEl,
        $("urlLegitBar"),
        $("urlPhishBar"),
        $("urlLegitVal"),
        $("urlPhishVal"),
        data.legit_probability ?? (1 - (data.phishing_probability ?? 0)),
        data.phishing_probability ?? (1 - (data.legit_probability ?? 0))
      );

      if (indicatorsEl) {
        indicatorsEl.textContent = pretty({
          normalized_url: data.normalized_url,
          indicators: data.indicators,
          model: data.model
        });
      }
    } catch (err) {
      alertEl.className = "alert bad";
      alertEl.textContent = `Connection failed: ${err.message}`;
      alertEl.classList.remove("hidden");
    } finally {
      urlBtn.disabled = false;
      urlBtn.textContent = "SCAN URL";
    }
  });

  // Email Scan
  const emailBtn = $("emailBtn");
  emailBtn?.addEventListener("click", async () => {
    const subject = ($("emailSubject")?.value || "").trim();
    const body = ($("emailBody")?.value || "").trim();

    const alertEl = $("emailAlert");
    const metersEl = $("emailMeters");
    const indicatorsEl = $("emailIndicators");

    hideAlert(alertEl);
    metersEl?.classList.add("hidden");
    if (indicatorsEl) indicatorsEl.textContent = "";

    if (!subject && !body) {
      alertEl.className = "alert bad";
      alertEl.textContent = "Please enter email subject/body.";
      alertEl.classList.remove("hidden");
      return;
    }

    emailBtn.disabled = true;
    emailBtn.textContent = "ANALYZING...";

    try {
      const data = await postJSON("/predict/email", { subject, body });

      showAlert(alertEl, data.verdict, data.confidence);

      setMeters(
        metersEl,
        $("emailLegitBar"),
        $("emailPhishBar"),
        $("emailLegitVal"),
        $("emailPhishVal"),
        data.legit_probability ?? (1 - (data.phishing_probability ?? 0)),
        data.phishing_probability ?? (1 - (data.legit_probability ?? 0))
      );

      if (indicatorsEl) {
        indicatorsEl.textContent = pretty({
          indicators: data.indicators,
          model: data.model
        });
      }
    } catch (err) {
      alertEl.className = "alert bad";
      alertEl.textContent = `Connection failed: ${err.message}`;
      alertEl.classList.remove("hidden");
    } finally {
      emailBtn.disabled = false;
      emailBtn.textContent = "ANALYZE EMAIL";
    }
  });

  // SMS Scan (tries /predict/sms, falls back to /predict/email)
  const smsBtn = $("smsBtn");
  smsBtn?.addEventListener("click", async () => {
    const msg = ($("smsBody")?.value || "").trim();

    const alertEl = $("smsAlert");
    const metersEl = $("smsMeters");
    const indicatorsEl = $("smsIndicators");

    hideAlert(alertEl);
    metersEl?.classList.add("hidden");
    if (indicatorsEl) indicatorsEl.textContent = "";

    if (!msg) {
      alertEl.className = "alert bad";
      alertEl.textContent = "Please paste an SMS message.";
      alertEl.classList.remove("hidden");
      return;
    }

    smsBtn.disabled = true;
    smsBtn.textContent = "ANALYZING...";

    try {
      let data;
      try {
        // If you add /predict/sms later, it will auto-use it.
        data = await postJSON("/predict/sms", { text: msg });
      } catch (e) {
        // Fallback: reuse email model
        data = await postJSON("/predict/email", { subject: "[SMS]", body: msg });
      }

      showAlert(alertEl, data.verdict, data.confidence);

      setMeters(
        metersEl,
        $("smsLegitBar"),
        $("smsPhishBar"),
        $("smsLegitVal"),
        $("smsPhishVal"),
        data.legit_probability ?? (1 - (data.phishing_probability ?? 0)),
        data.phishing_probability ?? (1 - (data.legit_probability ?? 0))
      );

      if (indicatorsEl) {
        indicatorsEl.textContent = pretty({
          indicators: data.indicators,
          model: data.model
        });
      }
    } catch (err) {
      alertEl.className = "alert bad";
      alertEl.textContent = `Connection failed: ${err.message}`;
      alertEl.classList.remove("hidden");
    } finally {
      smsBtn.disabled = false;
      smsBtn.textContent = "ANALYZE SMS";
    }
  });

  // History
  const historyBtn = $("historyBtn");
  const historyList = $("historyList");

  function parseMeta(meta_json) {
    if (!meta_json) return null;
    if (typeof meta_json === "object") return meta_json;
    if (typeof meta_json === "string") {
      try { return JSON.parse(meta_json); } catch { return { raw: meta_json }; }
    }
    return { raw: String(meta_json) };
  }

  function renderHistory(items = []) {
    if (!historyList) return;

    if (!items.length) {
      historyList.innerHTML = `<div class="muted">No scans found yet.</div>`;
      return;
    }

    historyList.innerHTML = items.map((it, idx) => {
      const type = (it.scan_type || "unknown").toLowerCase();
      const badgeClass = type.includes("url") ? "url" : type.includes("sms") ? "sms" : "email";
      const verdict = it.verdict || "unknown";
      const vClass = verdictClass(verdict);
      const conf = (typeof it.confidence === "number") ? Math.round(it.confidence * 100) : null;
      const confLabel = conf !== null ? `${conf}%` : "—";
      const created = it.created_at ? new Date(it.created_at).toLocaleString() : "";

      const preview = (it.input_value || "").toString().replace(/\s+/g, " ").trim();
      const safePreview = preview.length > 140 ? preview.slice(0, 140) + "…" : preview;

      const meta = parseMeta(it.meta_json);
      const metaText = pretty(meta);

      return `
        <div class="hitem">
          <div class="hrow">
            <div class="hleft">
              <div class="htitle">
                <span class="badge ${badgeClass}">${type.toUpperCase()}</span>
                <span>${created}</span>
              </div>
              <div class="hpreview">${escapeHtml(safePreview || "(no input)")}</div>
            </div>
            <div class="hright">
              <div class="score ${vClass}">${confLabel}</div>
              <button class="chev" data-idx="${idx}">Details</button>
            </div>
          </div>
          <div id="meta_${idx}" class="hmeta">
            <div class="muted" style="margin-top:8px;">Verdict: <b>${escapeHtml(verdict)}</b></div>
            <pre class="code" style="margin-top:8px;">${escapeHtml(metaText)}</pre>
          </div>
        </div>
      `;
    }).join("");

    // Wire up detail buttons
    historyList.querySelectorAll(".chev").forEach(btn => {
      btn.addEventListener("click", () => {
        const idx = btn.getAttribute("data-idx");
        const metaEl = document.getElementById(`meta_${idx}`);
        if (!metaEl) return;
        metaEl.classList.toggle("open");
        metaEl.classList.toggle("hmeta");
        // Keep class structure stable:
        metaEl.classList.add("hmeta");
        if (metaEl.classList.contains("open")) metaEl.classList.remove("open");
        else metaEl.classList.add("open");
      });
    });
  }

  function escapeHtml(str) {
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  historyBtn?.addEventListener("click", async () => {
    if (!historyList) return;
    historyBtn.disabled = true;
    historyBtn.textContent = "LOADING...";

    historyList.innerHTML = `<div class="muted">Loading history…</div>`;

    try {
      const data = await getJSON("/history/recent");
      // Supports both: {items:[...]} or [...]
      const items = Array.isArray(data) ? data : (data.items || []);
      renderHistory(items);

      // Optional: If API returns model meta, show it. If not, keep placeholder.
      // If you store test accuracy in email_meta.json/url_meta.json, we can wire it later.
      const modelAcc = document.getElementById("modelAcc");
      if (modelAcc && items.length > 0) {
        // If your meta_json has model/test_accuracy, try to display something meaningful
        const meta0 = parseMeta(items[0].meta_json);
        const acc = meta0?.model?.test_accuracy;
        if (typeof acc === "number") modelAcc.textContent = `${Math.round(acc * 100)}%`;
        else modelAcc.textContent = "—";
      }
    } catch (err) {
      historyList.innerHTML = `<div class="alert bad">History failed: ${escapeHtml(err.message)}</div>`;
    } finally {
      historyBtn.disabled = false;
      historyBtn.textContent = "REFRESH HISTORY";
    }
  });

  // Auto-load history once on page load (nice UX)
  window.addEventListener("DOMContentLoaded", () => {
    // Don’t auto-call if user doesn’t want; but it looks better for demo.
    historyBtn?.click();
  });
})();