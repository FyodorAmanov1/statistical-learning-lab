const $ = (id) => document.getElementById(id),
  esc = (s) =>
    String(s ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
const fmt = (v) =>
  typeof v === "boolean"
    ? v
      ? "yes"
      : "no"
    : typeof v === "number"
      ? !v
        ? "0"
        : Math.abs(v) < 0.001
          ? v.toExponential(3)
          : v.toLocaleString(undefined, { maximumFractionDigits: 4 })
      : String(v);
let study,
  current,
  registry,
  timer = null;
const palette = ["#65d6b1", "#b3bbf9", "#f4bf76", "#f596aa", "#7cb8e9"];
async function load() {
  try {
    const response = await fetch("study.json");
    if (!response.ok) throw Error("Study data could not be loaded.");
    study = await response.json();
    document.title = study.title + " · Fyodor Amanov";
    document.documentElement.style.setProperty("--accent", study.accent);
    $("title").textContent = study.title;
    $("subtitle").textContent = study.subtitle;
    $("source").href = "https://github.com/FyodorAmanov1/" + study.slug;
    $("series").textContent = String(
      [
        "quantum-estimation-lab",
        "quantum-optimization-lab",
        "market-microstructure-lab",
        "statistical-learning-lab",
        "ai-research-workbench",
      ].indexOf(study.slug) + 1,
    ).padStart(2, "0");
    $("provenance").textContent = study.provenance;
    $("references").innerHTML = study.references
      .map(
        (r) =>
          `<li><a href="${esc(r.url)}" target="_blank" rel="noreferrer">${esc(r.title)} ↗</a></li>`,
      )
      .join("");
    $("cases").innerHTML = study.cases
      .map(
        (c, i) =>
          `<button data-case="${esc(c.id)}"><small>EXPERIMENT ${String(i + 1).padStart(2, "0")}</small>${esc(c.label)}</button>`,
      )
      .join("");
    document.querySelectorAll("[data-case]").forEach(
      (b) =>
        (b.onclick = () => {
          location.hash = b.dataset.case;
        }),
    );
    window.addEventListener("hashchange", selectHash);
    selectHash();
    if (study.slug === "ai-research-workbench") {
      const r = await fetch("registry.json");
      if (!r.ok) throw Error("Evidence registry could not be loaded.");
      registry = await r.json();
      $("evidence").hidden = false;
      initializeEvidence();
    }
  } catch (e) {
    $("error").hidden = false;
    $("error").textContent =
      e.message + " Use an HTTP server or the published GitHub Pages site.";
  }
}
function selectHash() {
  const requested = decodeURIComponent(location.hash.slice(1));
  render(study.cases.find((c) => c.id === requested) || study.cases[0]);
}
function render(c) {
  stop();
  current = c;
  document.querySelectorAll("[data-case]").forEach((b) => {
    b.classList.toggle("active", b.dataset.case === c.id);
    b.setAttribute("aria-current", b.dataset.case === c.id ? "true" : "false");
  });
  $("kicker").textContent = c.kicker;
  $("case-title").textContent = c.title;
  $("description").textContent = c.description;
  $("metrics").innerHTML = c.metrics
    .map(
      (m) =>
        `<article class="metric"><span>${esc(m.label)}</span><strong>${esc(fmt(m.value))}<small>${esc(m.unit || "")}</small></strong>${m.note ? `<p>${esc(m.note)}</p>` : ""}</article>`,
    )
    .join("");
  $("charts").replaceChildren();
  for (const chart of c.charts || []) renderChart(chart);
  $("table").innerHTML =
    `<thead><tr>${c.table.columns.map((s) => `<th scope="col">${esc(s)}</th>`).join("")}</tr></thead><tbody>${c.table.rows.map((row) => `<tr>${row.map((v) => `<td>${esc(fmt(v))}</td>`).join("")}</tr>`).join("")}</tbody>`;
  $("method").innerHTML = c.method.map((s) => `<li>${esc(s)}</li>`).join("");
  $("limitations").innerHTML = c.limitations
    .map((s) => `<li>${esc(s)}</li>`)
    .join("");
  $("command").textContent = c.reproduce;
  $("replay").hidden = !c.replay;
  if (c.replay) {
    $("replay-range").max = c.replay.length - 1;
    $("replay-range").value = 0;
    replayFrame();
  }
}
function renderChart(data) {
  const section = document.createElement("section");
  section.className = "panel";
  section.innerHTML = `<span class="eyebrow">RECORDED COMPARISON</span><h3>${esc(data.title)}</h3><div class="legend">${data.series.map((s, i) => `<label><input type="checkbox" checked data-series="${i}" aria-label="Show ${esc(s.name)}"><i style="background:${palette[i % palette.length]}"></i>${esc(s.name)}</label>`).join("")}</div><div class="chart" role="img" aria-label="${esc(data.title)}"></div><p class="chart-note">${esc(data.xlabel)} · hover points for values · toggle series above</p>`;
  $("charts").append(section);
  const draw = () => {
    const enabled = [...section.querySelectorAll("input")].map(
        (b) => b.checked,
      ),
      active = data.series.filter((_, i) => enabled[i]);
    let values = active.flatMap((s) => s.values).filter(Number.isFinite);
    if (!values.length) values = [0, 1];
    let low = Math.min(...values),
      high = Math.max(...values);
    if (low === high) {
      low -= 0.5;
      high += 0.5;
    }
    const pad = (high - low) * 0.08;
    low -= pad;
    high += pad;
    const W = 650,
      H = 275,
      L = 62,
      R = 20,
      T = 20,
      B = 42,
      x = (i) => L + (i / Math.max(1, data.x.length - 1)) * (W - L - R),
      y = (v) => T + ((high - v) / (high - low)) * (H - T - B);
    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg"><title>${esc(data.title)}</title>`;
    for (let i = 0; i < 5; i++) {
      const v = low + ((high - low) * i) / 4;
      svg += `<path class="gridline" d="M${L} ${y(v)}H${W - R}"/><text x="${L - 10}" y="${y(v) + 3}" text-anchor="end">${esc(Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : Number(v.toPrecision(3)))}</text>`;
    }
    const ticks = [
      ...new Set([
        0,
        Math.floor((data.x.length - 1) / 4),
        Math.floor((data.x.length - 1) / 2),
        Math.floor((3 * (data.x.length - 1)) / 4),
        data.x.length - 1,
      ]),
    ];
    for (const i of ticks)
      svg += `<text x="${x(i)}" y="${H - 15}" text-anchor="middle">${esc(fmt(data.x[i]))}</text>`;
    data.series.forEach((s, j) => {
      if (!enabled[j]) return;
      const color = palette[j % palette.length];
      const points = s.values
        .map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(2)},${y(v).toFixed(2)}`)
        .join(" ");
      svg += `<path d="${points}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`;
      s.values.forEach((v, i) => {
        svg += `<circle cx="${x(i)}" cy="${y(v)}" r="${s.values.length > 100 ? 2 : 3}" fill="${color}" fill-opacity="${s.values.length > 100 ? 0.15 : 0.75}"><title>${esc(s.name)} · ${esc(data.x[i])}: ${esc(fmt(v))}</title></circle>`;
      });
    });
    section.querySelector(".chart").innerHTML = svg + "</svg>";
  };
  section.querySelectorAll("input").forEach((b) => (b.onchange = draw));
  draw();
}
function download(name, text, type) {
  const blob = new Blob([text], { type }),
    url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("csv").onclick = () =>
  download(
    current.id + ".csv",
    [current.table.columns, ...current.table.rows]
      .map((row) =>
        row.map((v) => '"' + String(v).replaceAll('"', '""') + '"').join(","),
      )
      .join("\n"),
    "text/csv",
  );
$("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText(current.reproduce);
    $("copy").textContent = "Copied";
  } catch {
    $("copy").textContent = "Select command";
  }
  setTimeout(() => ($("copy").textContent = "Copy"), 1800);
};
function stop() {
  clearInterval(timer);
  timer = null;
  $("play").textContent = "Play replay ▶";
}
function replayFrame() {
  if (!current?.replay) return;
  const i = Number($("replay-range").value),
    f = current.replay[i];
  $("replay-step").textContent =
    `BATCH ${String(i).padStart(3, "0")} / ${current.replay.length - 1}`;
  const max = Math.max(
    1,
    ...f.bid.map((l) => l.size),
    ...f.ask.map((l) => l.size),
  );
  const depth = (rows, side) =>
    rows
      .map(
        (l) =>
          `<div class="depth ${side}" style="--depth:${(l.size / max) * 100}%"><span>${l.price.toLocaleString()}</span><span>${l.size}</span></div>`,
      )
      .join("");
  $("asks").innerHTML = depth([...f.ask].reverse(), "ask");
  $("bids").innerHTML = depth(f.bid, "bid");
  $("mid").textContent =
    `REFERENCE ${f.mid.toLocaleString()} ticks · ${f.flow.toUpperCase()} flow${f.informed ? " · informed" : ""}`;
  $("replay-values").innerHTML =
    `<div><span>INVENTORY</span><strong>${f.inventory}</strong></div><div><span>MARKED EQUITY / TICKS</span><strong>${fmt(f.equity)}</strong></div>`;
  $("replay-fills").textContent = f.fills.length
    ? f.fills
        .map((v) => `${v.side.toUpperCase()} ${v.quantity} @ ${v.price}`)
        .join(" · ")
    : "No agent fills in this batch.";
}
$("replay-range").oninput = () => {
  stop();
  replayFrame();
};
$("play").onclick = () => {
  if (timer) {
    stop();
    return;
  }
  if (Number($("replay-range").value) >= current.replay.length - 1)
    $("replay-range").value = 0;
  $("play").textContent = "Pause replay Ⅱ";
  timer = setInterval(() => {
    const i = Number($("replay-range").value);
    if (i >= current.replay.length - 1) {
      stop();
      return;
    }
    $("replay-range").value = i + 1;
    replayFrame();
  }, 220);
};
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stop();
});
const STOP = new Set(
    "a an the and or of to for in on is are was with what how why does do from this that at as it its".split(
      " ",
    ),
  ),
  tokens = (s) =>
    (s.toLowerCase().match(/[a-z0-9]+/g) || []).filter((t) => !STOP.has(t));
function retrieve(query) {
  const terms = new Set(tokens(query)),
    docs = registry.documents,
    counts = docs.map((d) => {
      const c = {};
      for (const t of tokens(d.title + " " + d.text)) c[t] = (c[t] || 0) + 1;
      return c;
    }),
    lengths = counts.map((c) => Object.values(c).reduce((a, b) => a + b, 0)),
    average = lengths.reduce((a, b) => a + b, 0) / docs.length;
  return docs
    .map((d, i) => {
      let score = 0;
      for (const term of terms) {
        const f = counts[i][term] || 0;
        if (!f) continue;
        const df = counts.filter((c) => c[term]).length,
          idf = Math.log(1 + (docs.length - df + 0.5) / (df + 0.5));
        score +=
          (idf * f * 2.2) / (f + 1.2 * (0.25 + (0.75 * lengths[i]) / average));
      }
      return { ...d, score };
    })
    .filter((d) => d.score > 0)
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
    .slice(0, 3);
}
function search() {
  const hits = retrieve($("query").value);
  $("evidence-results").innerHTML = hits.length
    ? hits
        .map(
          (d) =>
            `<article class="evidence-card"><span class="eyebrow">${esc(d.id)} · RELEVANCE ${d.score.toFixed(2)}</span><h4>${esc(d.title)}</h4><p>${esc(d.text.slice(0, 520))}${d.text.length > 520 ? "…" : ""}</p><details><summary>Full evidence, methods & limitations</summary><p>${esc(d.text)}</p></details><p class="digest">CONTENT SHA-256: ${esc(d.sha256)}<br>STUDY ARTIFACT SHA-256: ${esc(d.artifact_sha256)}</p><a href="${esc(d.url)}" target="_blank" rel="noreferrer">Source artifact ↗</a> &nbsp; <a href="${esc(d.demo)}" target="_blank" rel="noreferrer">Open experiment ↗</a></article>`,
        )
        .join("")
    : '<p class="subtle">No supporting evidence in this registry. The explorer abstains.</p>';
}
function initializeEvidence() {
  $("search-form").onsubmit = (e) => {
    e.preventDefault();
    search();
  };
  document.querySelectorAll("[data-query]").forEach(
    (b) =>
      (b.onclick = () => {
        $("query").value = b.dataset.query;
        search();
      }),
  );
  $("claim-source").innerHTML = registry.documents
    .map((d) => `<option value="${esc(d.id)}">${esc(d.id)}</option>`)
    .join("");
  const metrics = () =>
    registry.documents.find((d) => d.id === $("claim-source").value).metrics;
  const resetValue = () => {
    const m = metrics()[Number($("claim-metric").value)];
    $("claim-value").value = m.value;
    $("claim-unit").value = m.unit || "";
    $("claim-status").textContent = "";
  };
  const resetMetrics = () => {
    $("claim-metric").innerHTML = metrics()
      .map((m, i) => `<option value="${i}">${esc(m.label)}</option>`)
      .join("");
    resetValue();
  };
  $("claim-source").onchange = resetMetrics;
  $("claim-metric").onchange = resetValue;
  $("claim-check").onclick = () => {
    const m = metrics()[Number($("claim-metric").value)],
      value = Number($("claim-value").value),
      match =
        $("claim-value").value !== "" &&
        Number.isFinite(value) &&
        Math.abs(value - m.value) <=
          Math.max(1e-8, 1e-6 * Math.max(Math.abs(value), Math.abs(m.value)));
    $("claim-status").textContent =
      $("claim-unit").value !== (m.unit || "")
        ? "Rejected: unit mismatch."
        : !match
          ? "Rejected: numeric value does not match the source."
          : "Accepted: matches the recorded metric and exact unit. This checks transcription, not scientific validity.";
  };
  resetMetrics();
  search();
}
load();
