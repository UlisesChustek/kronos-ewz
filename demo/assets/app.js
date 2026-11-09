/* Kronos × EWZ — static interactive forecast explorer
   Reads window.DEMO_DATA (precomputed) and renders with Plotly. */

(function () {
  "use strict";

  var D = window.DEMO_DATA;
  var root = {
    metrics: document.getElementById("metrics"),
    chart: document.getElementById("chart"),
    title: document.getElementById("chart-title"),
    sub: document.getElementById("chart-sub"),
    outcome: document.getElementById("outcome"),
    modelBtns: document.getElementById("model-buttons"),
    dateSelect: document.getElementById("date-select"),
    ctxBtns: document.getElementById("context-buttons"),
    playBtn: document.getElementById("play-btn"),
    samples: document.getElementById("show-samples"),
    legendActual: document.getElementById("legend-actual"),
    updated: document.getElementById("meta-updated")
  };

  if (!D) {
    if (root.chart) {
      root.chart.innerHTML =
        '<div class="empty">Dataset not found. Run <code>scripts/make_demo_data.py</code> first.</div>';
    }
    return;
  }

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var CONTEXTS = [64, 128, 256, 512];
  var state = {
    model: Object.keys(D.models)[0],
    dateId: D.dates[0].id,
    ctx: 128,
    samples: false
  };
  var playTimer = null;

  /* ---------- formatting ---------- */
  function pct(x, d) {
    d = d == null ? 1 : d;
    return (x * 100).toFixed(d) + "%";
  }
  function signed(x, d) {
    d = d == null ? 1 : d;
    var s = x < 0 ? "\u2212" : "+";
    return s + Math.abs(x * 100).toFixed(d) + "%";
  }

  /* Wilson score interval for a binomial proportion. */
  function wilson(k, n, z) {
    if (!n) return [0, 1];
    z = z || 1.96;
    var p = k / n;
    var den = 1 + (z * z) / n;
    var centre = p + (z * z) / (2 * n);
    var margin = z * Math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n);
    return [Math.max(0, (centre - margin) / den), Math.min(1, (centre + margin) / den)];
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function shortTime(iso, compact) {
    var d = iso.slice(0, 10).split("-");
    var hh = iso.slice(11, 13);
    var label = MONTHS[+d[1] - 1] + " " + +d[2];
    return compact ? label : label + " \u00b7 " + hh + ":00";
  }
  function dateById(id) {
    for (var i = 0; i < D.dates.length; i++) if (D.dates[i].id === id) return D.dates[i];
    return D.dates[0];
  }
  function scenario() {
    return D.scenarios[state.model + "|" + state.dateId] || null;
  }
  function horizonLabel(h) {
    return h === 7 ? "~1 trading day" : "~5 calendar days";
  }

  /* ---------- controls ---------- */
  function buildModelButtons() {
    Object.keys(D.models).forEach(function (key) {
      var m = D.models[key];
      var b = document.createElement("button");
      b.type = "button";
      b.dataset.model = key;
      b.innerHTML = "<span>" + esc(m.label) + "</span><small>" + esc(m.sublabel) + "</small>";
      b.addEventListener("click", function () {
        if (state.model === key) return;
        stopPlay();
        state.model = key;
        render();
      });
      root.modelBtns.appendChild(b);
    });
  }

  function buildContextButtons() {
    CONTEXTS.forEach(function (n) {
      var b = document.createElement("button");
      b.type = "button";
      b.dataset.ctx = String(n);
      b.textContent = String(n);
      b.title = n + " bars of history shown";
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", function () {
        if (state.ctx === n) return;
        state.ctx = n;
        render();
      });
      root.ctxBtns.appendChild(b);
    });
  }

  function buildDateSelect() {
    D.dates.forEach(function (d) {
      var o = document.createElement("option");
      o.value = d.id;
      o.textContent = d.label;
      root.dateSelect.appendChild(o);
    });
    root.dateSelect.addEventListener("change", function () {
      stopPlay();
      state.dateId = root.dateSelect.value;
      render();
    });
  }

  function updateControls() {
    Array.prototype.forEach.call(root.modelBtns.children, function (b) {
      var on = b.dataset.model === state.model;
      b.classList.toggle("active", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
    Array.prototype.forEach.call(root.ctxBtns.children, function (b) {
      var on = +b.dataset.ctx === state.ctx;
      b.classList.toggle("active", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
    root.dateSelect.value = state.dateId;
    root.samples.checked = state.samples;
  }

  /* ---------- metrics ---------- */
  function metric(k, v, s, cls) {
    return (
      '<div class="metric"><div class="k">' + k + "</div>" +
      '<div class="v' + (cls ? " " + cls : "") + '">' + v + "</div>" +
      '<div class="s">' + s + "</div></div>"
    );
  }

  function renderMetrics(sc) {
    var m = sc.metrics;
    var close0 = m.close0;
    var width = (m.p95_r - m.p5_r);
    var upCls = m.p_up >= 0.6 ? "up" : m.p_up <= 0.4 ? "down" : "plain";
    var medCls = m.median_r > 0.0005 ? "up" : m.median_r < -0.0005 ? "down" : "plain";
    var ampCls = m.amp > 1 ? "plain" : "plain";
    var nPaths = (D.meta && D.meta.samples) || 64;
    var kUp = Math.round(m.p_up * nPaths);
    var ci = wilson(kUp, nPaths);
    var upSub = kUp + "/" + nPaths + " paths \u00b7 MC 95% CI " +
      pct(ci[0], 0) + "\u2013" + pct(ci[1], 0) + "<br>terminal close &gt; $" + close0.toFixed(2);

    root.metrics.innerHTML =
      metric("P(up)", pct(m.p_up), upSub, upCls) +
      metric("Median move", signed(m.median_r), sc.horizon + "-bar horizon \u00b7 median path", medCls) +
      metric("90% band", pct(width), signed(m.p5_r) + " to " + signed(m.p95_r), "plain range") +
      metric("P(+2%)", pct(m.p_up_2), "terminal return \u2265 +2%", "plain") +
      metric("P(\u22122%)", pct(m.p_down_2), "terminal return \u2264 \u22122%", "plain") +
      metric(
        "Volatility amp",
        m.amp.toFixed(2) + "\u00d7",
        "forecast \u03c3 " + pct(m.vol_pred, 2) + " vs trailing " + pct(m.trail_vol, 2),
        ampCls
      );
  }

  function renderOutcome(sc, d) {
    var m = sc.metrics;
    var parts = [];
    if (m.r_actual != null) {
      var good = m.r_actual >= 0;
      var hit = !!m.dir_correct;
      var inBand = Math.round(m.coverage_90 * sc.horizon);
      parts.push(
        '<span class="pill ' + (good ? "good" : "bad") + '">Actual ' + signed(m.r_actual) + "</span>"
      );
      parts.push("<span>Median path <b>" + signed(m.median_r) + "</b></span>");
      parts.push('<span class="sep"></span>');
      parts.push(
        '<span>Direction <b class="' + (hit ? "ok" : "miss") + '">' + (hit ? "\u2713 hit" : "\u2717 miss") + "</b></span>"
      );
      parts.push("<span>MAE <b>" + m.mae.toFixed(4) + "</b> vs RW <b>" + m.rw_mae.toFixed(4) + "</b></span>");
      parts.push("<span>" + inBand + "/" + sc.horizon + " closes in 90% band</span>");
      parts.push('<span class="sep"></span>');
      parts.push("<span>Raw model frequencies \u2014 uncalibrated</span>");
    } else {
      parts.push('<span class="pill info">Live</span>');
      parts.push("<span>Newest available bar. No realized outcome yet.</span>");
      parts.push('<span class="sep"></span>');
      parts.push(
        "<span>" + sc.horizon + " bars forecast \u00b7 " + horizonLabel(sc.horizon) + " of EWZ session data</span>"
      );
      parts.push('<span class="sep"></span>');
      parts.push("<span>Raw model frequencies \u2014 uncalibrated</span>");
    }
    root.outcome.innerHTML = parts.join("");
  }

  /* ---------- chart ---------- */
  function buildTraces(d, sc, m) {
    var h = d.history;
    var n = Math.min(state.ctx, h.t.length);
    var start = h.t.length - n;
    var histT = h.t.slice(start);
    var histO = h.o.slice(start);
    var histH = h.h.slice(start);
    var histL = h.l.slice(start);
    var histC = h.c.slice(start);

    var f = sc.forecast;
    var fT = f.t;
    var H = fT.length;

    var xBand = fT.concat(fT.slice().reverse());
    var y90 = f.p5.concat(f.p95.slice().reverse());
    var y50 = f.p25.concat(f.p75.slice().reverse());

    var traces = [
      {
        type: "scatter",
        x: xBand,
        y: y90,
        fill: "toself",
        fillcolor: "rgba(106,169,255,0.13)",
        line: { color: "rgba(106,169,255,0.0)", width: 0 },
        hoverinfo: "skip",
        name: "90% band",
        showlegend: false
      },
      {
        type: "scatter",
        x: xBand,
        y: y50,
        fill: "toself",
        fillcolor: "rgba(106,169,255,0.20)",
        line: { color: "rgba(106,169,255,0.0)", width: 0 },
        hoverinfo: "skip",
        name: "50% band",
        showlegend: false
      }
    ];

    if (state.samples && f.samples) {
      f.samples.forEach(function (path) {
        traces.push({
          type: "scatter",
          x: fT,
          y: path,
          mode: "lines",
          line: { color: "rgba(140,190,255,0.30)", width: 1 },
          hoverinfo: "skip",
          showlegend: false
        });
      });
    }

    traces.push({
      type: "candlestick",
      x: histT,
      open: histO,
      high: histH,
      low: histL,
      close: histC,
      name: "EWZ",
      increasing: {
        line: { color: "#6ef2b0", width: 0.6 },
        fillcolor: "rgba(110,242,176,0.55)"
      },
      decreasing: {
        line: { color: "#ff7a7a", width: 0.6 },
        fillcolor: "rgba(255,122,122,0.5)"
      },
      whiskerwidth: 0,
      hoverlabel: { bgcolor: "#0d131c", bordercolor: "#2b3d55" }
    });

    traces.push({
      type: "scatter",
      x: fT,
      y: f.p50,
      mode: "lines",
      line: { color: "#6ef2b0", width: 2.4 },
      name: "median",
      hovertemplate: "%{x}<br>median $%{y:.2f}<extra></extra>",
      showlegend: false
    });

    if (d.actual) {
      traces.push({
        type: "scatter",
        x: d.actual.t.slice(0, H),
        y: d.actual.c.slice(0, H),
        mode: "lines",
        line: { color: "#ffffff", width: 1.7, dash: "dot" },
        name: "actual",
        hovertemplate: "%{x}<br>actual $%{y:.2f}<extra></extra>",
        showlegend: false
      });
    }

    return { traces: traces, histT: histT, fT: fT, allT: histT.concat(fT) };
  }

  function tickConfig(histT, fT) {
    var all = histT.concat(fT);
    var L = all.length;
    var compact = window.innerWidth < 640;
    var count = compact ? 5 : 8;
    var step = (L - 1) / (count - 1);
    var idx = [];
    for (var i = 0; i < count; i++) idx.push(Math.round(i * step));

    // Mark the history/forecast boundary only when it will not collide.
    var boundary = histT.length - 1;
    var tooClose = idx.some(function (v) { return Math.abs(v - boundary) < step * 0.42; });
    if (!tooClose) idx.push(boundary);

    idx = idx.filter(function (v, i, a) {
      return v >= 0 && v < L && a.indexOf(v) === i;
    }).sort(function (a, b) { return a - b; });

    var vals = [];
    var text = [];
    idx.forEach(function (i) {
      vals.push(all[i]);
      text.push(shortTime(all[i], compact));
    });
    return { vals: vals, text: text };
  }

  function renderChart(d, sc, m) {
    if (!window.Plotly) {
      root.chart.innerHTML = '<div class="empty">Plotly failed to load.</div>';
      return;
    }

    var built = buildTraces(d, sc, m);
    var ticks = tickConfig(built.histT, built.fT);
    var lastHist = built.histT[built.histT.length - 1];
    var lastForecast = built.fT[built.fT.length - 1];

    var shapes = [
      {
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: lastHist,
        x1: lastForecast,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(106,169,255,0.035)",
        line: { width: 0 },
        layer: "below"
      },
      {
        type: "line",
        xref: "x",
        yref: "paper",
        x0: lastHist,
        x1: lastHist,
        y0: 0,
        y1: 1,
        line: { color: "rgba(233,239,248,0.28)", width: 1, dash: "dot" }
      }
    ];

    var layout = {
      autosize: true,
      margin: { l: 40, r: 58, t: 22, b: 34 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size: 11, color: "#8ea0b8" },
      showlegend: false,
      hovermode: "x",
      hoverlabel: {
        bgcolor: "#0d131c",
        bordercolor: "#2b3d55",
        font: { color: "#e9eff8", size: 11 }
      },
      dragmode: "zoom",
      uirevision: state.model + "|" + state.dateId + "|" + state.ctx,
      xaxis: {
        type: "category",
        categoryorder: "array",
        categoryarray: built.allT,
        showgrid: false,
        zeroline: false,
        showline: true,
        linecolor: "#1d2938",
        tickmode: "array",
        tickvals: ticks.vals,
        ticktext: ticks.text,
        tickfont: { size: 10, color: "#6f829a" },
        ticks: "",
        showspikes: false,
        rangeslider: { visible: false }
      },
      yaxis: {
        side: "right",
        showgrid: true,
        gridcolor: "#16202e",
        gridwidth: 1,
        zeroline: false,
        showline: false,
        tickprefix: "$",
        tickfont: { size: 10, color: "#6f829a" },
        fixedrange: false
      },
      shapes: shapes
    };

    Plotly.react(root.chart, built.traces, layout, {
      displayModeBar: false,
      responsive: true,
      scrollZoom: false,
      doubleClick: "reset"
    });
  }

  /* ---------- render ---------- */
  function render() {
    var d = dateById(state.dateId);
    var sc = scenario();
    var m = D.models[state.model];

    updateControls();

    if (!sc) {
      root.metrics.innerHTML = "";
      root.title.textContent = "No scenario";
      root.sub.textContent = "This model/date combination was not precomputed.";
      root.chart.innerHTML = '<div class="empty">Scenario unavailable.</div>';
      root.outcome.innerHTML = "";
      return;
    }

    root.title.textContent = d.label + " \u00b7 " + m.label;
    root.sub.textContent =
      sc.horizon + "-bar horizon (" + horizonLabel(sc.horizon) + ") \u00b7 " +
      D.meta.samples + " sampled paths \u00b7 " + D.meta.lookback + "-bar context \u00b7 T=" +
      m.temperature.toFixed(1) + " \u00b7 top-p=" + m.top_p.toFixed(2);

    renderMetrics(sc);
    renderOutcome(sc, d);
    if (root.legendActual) root.legendActual.style.display = d.actual ? "" : "none";
    renderChart(d, sc, m);
    syncURL();
  }

  /* ---------- play ---------- */
  function chronoDates() {
    return D.dates.slice().sort(function (a, b) { return +a.id - +b.id; });
  }

  function startPlay() {
    var order = chronoDates();
    var i = 0;
    for (var k = 0; k < order.length; k++) {
      if (order[k].id === state.dateId) { i = k; break; }
    }
    if (i >= order.length - 1) i = -1; // restart from oldest

    root.playBtn.setAttribute("aria-pressed", "true");
    root.playBtn.textContent = "\u23f8 Pause";

    function step() {
      i += 1;
      if (i >= order.length) {
        stopPlay();
        return;
      }
      state.dateId = order[i].id;
      render();
    }
    step();
    playTimer = window.setInterval(step, 1500);
  }

  function stopPlay() {
    if (playTimer) {
      window.clearInterval(playTimer);
      playTimer = null;
    }
    root.playBtn.setAttribute("aria-pressed", "false");
    root.playBtn.textContent = "\u25b6 Play";
  }

  root.playBtn.addEventListener("click", function () {
    if (playTimer) stopPlay();
    else startPlay();
  });

  root.samples.addEventListener("change", function () {
    state.samples = root.samples.checked;
    render();
  });

  /* ---------- URL ---------- */
  function syncURL() {
    try {
      var p = new URLSearchParams();
      p.set("model", state.model);
      p.set("date", state.dateId);
      p.set("ctx", String(state.ctx));
      if (state.samples) p.set("samples", "1");
      history.replaceState(null, "", location.pathname + "?" + p.toString());
    } catch (e) {
      /* file:// or restricted context */
    }
  }

  function readURL() {
    var p = new URLSearchParams(location.search);
    var model = p.get("model");
    var date = p.get("date");
    var ctx = parseInt(p.get("ctx"), 10);
    if (model && D.models[model]) state.model = model;
    if (date && dateById(date) && dateById(date).id === date) state.dateId = date;
    if (CONTEXTS.indexOf(ctx) !== -1) state.ctx = ctx;
    state.samples = p.get("samples") === "1";
  }

  /* ---------- init ---------- */
  if (root.updated) {
    root.updated.textContent = "dataset \u00b7 " + (D.meta.generated_at || "precomputed");
  }
  buildModelButtons();
  buildContextButtons();
  buildDateSelect();
  readURL();

  var resizeTimer = null;
  window.addEventListener("resize", function () {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(function () {
      var d = dateById(state.dateId);
      var sc = scenario();
      if (sc) renderChart(d, sc, D.models[state.model]);
    }, 180);
  });

  render();
})();
