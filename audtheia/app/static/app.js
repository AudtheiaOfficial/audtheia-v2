/* Audtheia local interface behavior.

   This script binds the static shell to the local backend. It reads the
   record through the /api endpoints, renders each panel, keeps the live views
   current on a gentle poll, and wires the two actions the backend supports
   today: pausing or resuming a longitudinal pass, and asking for a report.

   Everything runs against the same local origin that served the page, so there
   is no outside request at any point. Interactive features that would change
   stored data (editing memory, adding models or skills, running training) need
   backend write paths that are added in a later step; where the interface
   offers them, it says clearly that they are not active yet, rather than
   pretending to act.

   Structure:
     1. Constants and small helpers
     2. Preferences (theme, display timezone, last panel) held in the browser
     3. Connection status
     4. Router: primary panels and the Brain sub-panels
     5. Rendering helpers (badges, tables, simple charts, a coordinate plot)
     6. One loader per panel
     7. Startup
*/

(function () {
  "use strict";

  // ----------------------------------------------------------------------
  // 1. Constants and small helpers
  // ----------------------------------------------------------------------

  var API = "/api";

  // How often the active live view re-reads the record. The backend keeps
  // detection event-driven; this only controls how quickly the screen catches
  // up, and a manual refresh is always available.
  var POLL_INTERVAL_MS = 10000;
  var HEALTH_INTERVAL_MS = 15000;

  // Keys under which the browser remembers a user's display choices.
  var STORE = {
    theme: "audtheia.theme",
    lastDark: "audtheia.theme.lastDark",
    lastLight: "audtheia.theme.lastLight",
    panel: "audtheia.panel",
    subpanel: "audtheia.subpanel",
    settingsGroup: "audtheia.settingsGroup",
    timezone: "audtheia.timezone",
    station: "audtheia.station"
  };

  // The theme names the stylesheet defines, and which mode each one is, so the
  // top-bar control can flip between a dark and a light choice.
  var THEMES = {
    ocean: { label: "Ocean", mode: "dark" },
    forest: { label: "Forest", mode: "light" },
    flat: { label: "Flat", mode: "light" },
    cyberpunk: { label: "Cyberpunk", mode: "dark" },
    retro: { label: "Retro-Futurism", mode: "dark" },
    glass: { label: "Glassmorphism", mode: "dark" },
    neumorph: { label: "Neumorphism", mode: "light" }
  };
  var DEFAULT_THEME = "ocean";

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  function store(key, value) {
    try {
      if (value === undefined) { return window.localStorage.getItem(key); }
      window.localStorage.setItem(key, value);
    } catch (e) { return null; }
  }

  // Build a DOM element from a tag, a set of properties, and children.
  function el(tag, props, children) {
    var node = document.createElement(tag);
    if (props) {
      Object.keys(props).forEach(function (k) {
        if (k === "class") { node.className = props[k]; }
        else if (k === "text") { node.textContent = props[k]; }
        else if (k === "html") { node.innerHTML = props[k]; }
        else if (k.indexOf("data-") === 0 || k.indexOf("aria-") === 0) { node.setAttribute(k, props[k]); }
        else if (k === "onclick") { node.addEventListener("click", props[k]); }
        else { node[k] = props[k]; }
      });
    }
    (children || []).forEach(function (c) {
      if (c === null || c === undefined) { return; }
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  }

  function clear(node) { while (node && node.firstChild) { node.removeChild(node.firstChild); } }

  function region(name) { return $('[data-region="' + name + '"]'); }

  function setState(node, kind, message) {
    if (!node) { return; }
    clear(node);
    node.appendChild(el("p", { class: kind, text: message }));
  }

  // Read one endpoint as JSON. Throws on a non-success response so callers can
  // show a clear message instead of rendering half a view.
  function apiGet(path) {
    return fetch(API + path, { headers: { "Accept": "application/json" } }).then(function (r) {
      if (!r.ok) { throw new Error("request failed (" + r.status + ")"); }
      return r.json();
    });
  }

  function apiSend(path, method, body) {
    var opts = { method: method, headers: { "Accept": "application/json" } };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(API + path, opts).then(function (r) {
      if (!r.ok) { throw new Error("request failed (" + r.status + ")"); }
      return r.json();
    });
  }

  function query(params) {
    var parts = [];
    Object.keys(params).forEach(function (k) {
      var v = params[k];
      if (v !== undefined && v !== null && v !== "") {
        parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(v));
      }
    });
    return parts.length ? "?" + parts.join("&") : "";
  }

  // ----------------------------------------------------------------------
  // 2. Formatting
  // ----------------------------------------------------------------------

  function displayTimezone() {
    return store(STORE.timezone) || state.serverTimezone || Intl.DateTimeFormat().resolvedOptions().timeZone;
  }

  // Timestamps are stored in UTC; localize only for display, per the record's
  // time rule.
  function fmtTime(iso) {
    if (!iso) { return "not recorded"; }
    var d = new Date(iso);
    if (isNaN(d.getTime())) { return String(iso); }
    try {
      return new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium", timeStyle: "short", timeZone: displayTimezone()
      }).format(d);
    } catch (e) {
      return d.toISOString();
    }
  }

  function fmtNum(v, digits) {
    if (v === null || v === undefined || v === "") { return "-"; }
    var n = Number(v);
    if (isNaN(n)) { return String(v); }
    return digits ? n.toFixed(digits) : String(Math.round(n));
  }

  function fmtPct(fraction) {
    if (fraction === null || fraction === undefined) { return "-"; }
    return (Number(fraction) * 100).toFixed(1) + "%";
  }

  function fmtConfidence(v) {
    if (v === null || v === undefined) { return "-"; }
    return (Number(v) * 100).toFixed(0) + "%";
  }

  function taxonName(det) {
    return det.common_name || det.scientific_name || det.gbif_usage_key || "unresolved taxon";
  }

  // ----------------------------------------------------------------------
  // 3. Reusable render pieces
  // ----------------------------------------------------------------------

  function badge(text, kind) {
    return el("span", { class: "badge badge-" + (kind || "neutral"), text: text });
  }

  // Every value shown carries its origin and status where the record has them,
  // so measured and inferred data stay visibly distinct.
  function provenanceBadges(source, status) {
    var out = [];
    if (source) { out.push(badge(source, "source")); }
    if (status) { out.push(badge(status, "status")); }
    return out;
  }

  function metricCard(label, value) {
    return el("div", { class: "metric-card" }, [
      el("div", { class: "metric-label", text: label }),
      el("div", { class: "metric-value", text: value })
    ]);
  }

  function metricRow(cards) {
    return el("div", { class: "metric-row" }, cards);
  }

  // A note for a control that is intentionally not active until a later
  // backend step. It is shown, not hidden, and it never looks operational.
  function deferredNote(text) {
    return el("div", { class: "deferred" }, [
      el("span", { class: "deferred-tag", text: "Not active yet" }),
      el("span", { class: "deferred-text", text: text })
    ]);
  }

  function svgEl(name, attrs) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }

  // A simple horizontal bar chart drawn as inline SVG, themed through the CSS
  // variables so it matches every palette. Values is an array of {label,value}.
  function barChart(values, opts) {
    opts = opts || {};
    var width = 640, rowH = 30, pad = 8, labelW = 170;
    var max = values.reduce(function (m, d) { return Math.max(m, Number(d.value) || 0); }, 0) || 1;
    var height = pad * 2 + values.length * rowH;
    var svg = svgEl("svg", {
      viewBox: "0 0 " + width + " " + height, width: "100%", height: height,
      role: "img", "aria-label": opts.title || "bar chart", class: "chart"
    });
    values.forEach(function (d, i) {
      var y = pad + i * rowH;
      var barMax = width - labelW - 60;
      var w = Math.max(2, Math.round((Number(d.value) || 0) / max * barMax));
      svg.appendChild(svgEl("text", { x: 0, y: y + 19, class: "chart-label" })).appendChild(document.createTextNode(d.label));
      svg.appendChild(svgEl("rect", { x: labelW, y: y + 6, width: w, height: rowH - 14, rx: 4, class: "chart-bar" }));
      var val = svgEl("text", { x: labelW + w + 8, y: y + 19, class: "chart-value" });
      val.appendChild(document.createTextNode(String(d.value)));
      svg.appendChild(val);
    });
    return svg;
  }

  // A coordinate plot of detection locations, drawn as inline SVG with no map
  // tiles so it works fully offline. Points is an array of {lat, lon, label}.
  function coordinatePlot(points) {
    var width = 640, height = 380, pad = 46;
    var svg = svgEl("svg", {
      viewBox: "0 0 " + width + " " + height, width: "100%", height: height,
      role: "img", "aria-label": "detection locations", class: "chart map-plot"
    });
    var lats = points.map(function (p) { return p.lat; });
    var lons = points.map(function (p) { return p.lon; });
    var minLat = Math.min.apply(null, lats), maxLat = Math.max.apply(null, lats);
    var minLon = Math.min.apply(null, lons), maxLon = Math.max.apply(null, lons);
    var latSpan = (maxLat - minLat) || 0.001, lonSpan = (maxLon - minLon) || 0.001;

    function px(lon) { return pad + (lon - minLon) / lonSpan * (width - pad * 2); }
    function py(lat) { return height - pad - (lat - minLat) / latSpan * (height - pad * 2); }

    svg.appendChild(svgEl("rect", { x: pad, y: pad, width: width - pad * 2, height: height - pad * 2, class: "map-frame", fill: "none" }));
    var xl = svgEl("text", { x: width / 2, y: height - 14, class: "chart-axis", "text-anchor": "middle" });
    xl.appendChild(document.createTextNode("longitude " + minLon.toFixed(4) + " to " + maxLon.toFixed(4)));
    svg.appendChild(xl);
    var yl = svgEl("text", { x: 16, y: height / 2, class: "chart-axis", transform: "rotate(-90 16 " + (height / 2) + ")", "text-anchor": "middle" });
    yl.appendChild(document.createTextNode("latitude " + minLat.toFixed(4) + " to " + maxLat.toFixed(4)));
    svg.appendChild(yl);

    points.forEach(function (p) {
      var c = svgEl("circle", { cx: px(p.lon), cy: py(p.lat), r: 5, class: "map-point" });
      var title = svgEl("title", {});
      title.appendChild(document.createTextNode((p.label || "") + " (" + p.lat.toFixed(5) + ", " + p.lon.toFixed(5) + ")"));
      c.appendChild(title);
      svg.appendChild(c);
    });
    return svg;
  }

  function filterBar(onChange, opts) {
    opts = opts || {};
    var bar = el("div", { class: "filter-bar" });
    var select = el("select", { class: "filter-station", "aria-label": "Station" });
    select.appendChild(el("option", { value: "", text: "All stations" }));
    (state.stations || []).forEach(function (s) {
      var o = el("option", { value: s.id, text: s.station_name || s.id });
      if (s.id === state.stationId) { o.selected = true; }
      select.appendChild(o);
    });
    select.addEventListener("change", function () {
      state.stationId = select.value;
      store(STORE.station, select.value);
      onChange();
    });
    bar.appendChild(el("label", { class: "filter-field" }, [el("span", { text: "Station" }), select]));

    var refresh = el("button", { type: "button", class: "btn", "aria-label": "Refresh" }, [
      el("span", { class: "nav-icon", "aria-hidden": "true", html: iconRefresh() }),
      el("span", { text: "Refresh" })
    ]);
    refresh.addEventListener("click", onChange);
    bar.appendChild(refresh);
    return bar;
  }

  function iconRefresh() {
    return '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' +
      '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"></path><path d="M21 3v5h-5"></path></svg>';
  }

  // ----------------------------------------------------------------------
  // 4. Application state
  // ----------------------------------------------------------------------

  var state = {
    stations: [],
    stationId: store(STORE.station) || "",
    serverTimezone: null,
    activePanel: null,
    pollTimer: null
  };

  // ----------------------------------------------------------------------
  // 5. Theme control
  // ----------------------------------------------------------------------

  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") || DEFAULT_THEME;
  }

  function applyTheme(name) {
    if (!THEMES[name]) { name = DEFAULT_THEME; }
    document.documentElement.setAttribute("data-theme", name);
    store(STORE.theme, name);
    if (THEMES[name].mode === "dark") { store(STORE.lastDark, name); }
    else { store(STORE.lastLight, name); }
    var picker = $("#appearance-theme");
    if (picker) { picker.value = name; }
  }

  function toggleMode() {
    var mode = THEMES[currentTheme()].mode;
    if (mode === "dark") { applyTheme(store(STORE.lastLight) || "forest"); }
    else { applyTheme(store(STORE.lastDark) || "ocean"); }
  }

  function initTheme() {
    applyTheme(store(STORE.theme) || currentTheme() || DEFAULT_THEME);
    var toggle = $("[data-theme-toggle]");
    if (toggle) { toggle.addEventListener("click", toggleMode); }
  }

  // ----------------------------------------------------------------------
  // 6. Connection status
  // ----------------------------------------------------------------------

  function refreshHealth() {
    var pill = $("[data-connection-status]");
    apiGet("/health").then(function (h) {
      state.serverTimezone = h.timezone_display || state.serverTimezone;
      if (pill) {
        pill.setAttribute("data-connection-status", "online");
        $(".status-text", pill).textContent = "Online";
      }
    }).catch(function () {
      if (pill) {
        pill.setAttribute("data-connection-status", "offline");
        $(".status-text", pill).textContent = "Offline";
      }
    });
  }

  // ----------------------------------------------------------------------
  // 7. Router
  // ----------------------------------------------------------------------

  var loaders = {};

  function stopPolling() {
    if (state.pollTimer) { window.clearInterval(state.pollTimer); state.pollTimer = null; }
  }

  // The live views (detections, the longitudinal status) re-read on the poll;
  // the reference views load once when opened.
  var LIVE_PANELS = { detections: true, audio: true, brain: true, analytics: true };

  function activatePanel(name) {
    if (!loaders[name]) { name = "detections"; }
    state.activePanel = name;
    store(STORE.panel, name);

    $all(".panel").forEach(function (p) { p.hidden = p.getAttribute("data-panel") !== name; });
    $all(".nav-item").forEach(function (b) {
      if (b.getAttribute("data-panel") === name) { b.setAttribute("aria-current", "page"); }
      else { b.removeAttribute("aria-current"); }
    });
    var titleEl = $("[data-view-title]");
    var btn = $('.nav-item[data-panel="' + name + '"]');
    if (titleEl && btn) { titleEl.textContent = $(".nav-label", btn).textContent; }

    stopPolling();
    loaders[name]();
    if (LIVE_PANELS[name]) {
      state.pollTimer = window.setInterval(loaders[name], POLL_INTERVAL_MS);
    }
  }

  function activateSubpanel(name) {
    $all(".subpanel").forEach(function (p) { p.hidden = p.getAttribute("data-subpanel") !== name; });
    $all(".subnav-item").forEach(function (b) {
      if (b.getAttribute("data-subpanel") === name) { b.setAttribute("aria-current", "page"); }
      else { b.removeAttribute("aria-current"); }
    });
    store(STORE.subpanel, name);
  }

  function initRouter() {
    $all(".nav-item").forEach(function (b) {
      b.addEventListener("click", function () { activatePanel(b.getAttribute("data-panel")); });
    });
    $all(".subnav-item").forEach(function (b) {
      b.addEventListener("click", function () { activateSubpanel(b.getAttribute("data-subpanel")); });
    });
  }

  // ----------------------------------------------------------------------
  // 8. Panel loaders
  // ----------------------------------------------------------------------

  // Detections: live event cards, each showing provenance, quality state, and
  // the desktop verification verdict when present.
  loaders.detections = function () {
    var host = region("detections-list");
    var filters = region("detections-filters");
    if (filters && !filters.hasChildNodes()) { filters.appendChild(filterBar(loaders.detections)); }
    apiGet("/detections" + query({ station_id: state.stationId, limit: 100 })).then(function (rows) {
      clear(host);
      if (!rows.length) { setState(host, "empty-state", "No detections recorded yet."); return; }
      var grid = el("div", { class: "card-grid" });
      rows.forEach(function (obs) {
        var v = obs.verification;
        var species = (obs.vision_detections || []).map(taxonName).join(", ") || "no resolved taxon";
        var badges = provenanceBadges(obs.data_source, obs.qc_state);
        if (v && v.verified) { badges.push(badge("verified", "ok")); }
        else { badges.push(badge("not verified", "muted")); }
        var meta = [
          obs.event_name || obs.id,
          fmtTime(obs.first_seen),
          "trigger: " + (obs.trigger_source || "unknown")
        ].join(" · ");
        grid.appendChild(el("article", { class: "card" }, [
          el("div", { class: "card-meta", text: meta }),
          el("div", { class: "card-title", text: species }),
          el("div", { class: "card-stats", text:
            "frames: " + fmtNum(obs.frame_count) +
            "  duration: " + fmtNum(obs.duration, 1) + "s" +
            "  confidence: " + fmtConfidence(obs.screening_confidence) +
            "  salience: " + fmtNum(obs.salience_provisional, 2) }),
          el("div", { class: "badge-row" }, badges)
        ]));
      });
      host.appendChild(grid);
    }).catch(function (e) { setState(host, "empty-state", "Could not load detections: " + e.message); });
  };

  // Audio: acoustic detections tied to events, with the true clip duration.
  loaders.audio = function () {
    var host = region("audio-list");
    var filters = region("audio-filters");
    if (filters && !filters.hasChildNodes()) { filters.appendChild(filterBar(loaders.audio)); }
    apiGet("/audio" + query({ station_id: state.stationId, limit: 100 })).then(function (rows) {
      clear(host);
      if (!rows.length) { setState(host, "empty-state", "No acoustic detections yet."); return; }
      var grid = el("div", { class: "card-grid" });
      rows.forEach(function (a) {
        var species = (a.audio_detections || []).map(taxonName).join(", ") || "unclassified sound";
        grid.appendChild(el("article", { class: "card" }, [
          el("div", { class: "card-meta", text: (a.event_name || a.observation_id) + " · " + fmtTime(a.first_seen) }),
          el("div", { class: "card-title", text: species }),
          el("div", { class: "card-stats", text:
            "true duration: " + fmtNum(a.audio_true_duration_seconds, 1) + "s" +
            (a.audio_capped ? " (stored clip capped)" : "") +
            "  model: " + (a.acoustic_model_version || "unstated") }),
          el("div", { class: "card-note", text: a.audio_clip_path
            ? "Clip is stored with the event. In-browser playback arrives with the media serving path."
            : "No stored clip for this event." })
        ]));
      });
      host.appendChild(grid);
    }).catch(function (e) { setState(host, "empty-state", "Could not load audio: " + e.message); });
  };

  // GPS: a self-contained coordinate plot, no external map tiles.
  loaders.gps = function () {
    var host = region("gps-map");
    var filters = region("gps-filters");
    if (filters && !filters.hasChildNodes()) { filters.appendChild(filterBar(loaders.gps)); }
    apiGet("/gps" + query({ station_id: state.stationId, limit: 1000 })).then(function (rows) {
      clear(host);
      var points = rows.filter(function (r) {
        return r.gps_latitude !== null && r.gps_longitude !== null;
      }).map(function (r) {
        return { lat: Number(r.gps_latitude), lon: Number(r.gps_longitude), label: r.event_name || r.observation_id };
      });
      if (!points.length) { setState(host, "empty-state", "No located detections yet."); return; }
      host.appendChild(coordinatePlot(points));
      host.appendChild(el("p", { class: "card-note", text: fmtNum(points.length) + " located detections. Coordinates are measured GPS positions." }));
    }).catch(function (e) { setState(host, "empty-state", "Could not load spatial data: " + e.message); });
  };

  // Analytics: biodiversity summaries computed from the record, drawn as SVG.
  loaders.analytics = function () {
    var host = region("analytics-charts");
    var filters = region("analytics-filters");
    if (filters && !filters.hasChildNodes()) { filters.appendChild(filterBar(loaders.analytics)); }
    apiGet("/analytics" + query({ station_id: state.stationId })).then(function (a) {
      clear(host);
      host.appendChild(metricRow([
        metricCard("Events", fmtNum(a.total_events)),
        metricCard("Species richness", fmtNum(a.species_richness)),
        metricCard("Verified", fmtNum(a.verified_count)),
        metricCard("Verified share", fmtPct(a.verified_fraction))
      ]));

      var taxa = Object.keys(a.taxon_event_counts || {}).map(function (k) {
        return { label: k, value: a.taxon_event_counts[k] };
      }).sort(function (x, y) { return y.value - x.value; }).slice(0, 12);
      if (taxa.length) {
        host.appendChild(el("h3", { text: "Detections per taxon" }));
        host.appendChild(barChart(taxa, { title: "detections per taxon" }));
      }

      var qc = Object.keys(a.events_by_qc_state || {}).map(function (k) {
        return { label: k, value: a.events_by_qc_state[k] };
      });
      if (qc.length) {
        host.appendChild(el("h3", { text: "Events by quality state" }));
        host.appendChild(barChart(qc, { title: "events by quality state" }));
      }
      host.appendChild(el("p", { class: "card-note", text: a.note || "" }));
    }).catch(function (e) { setState(host, "empty-state", "Could not load analytics: " + e.message); });
  };

  // Brain, Models and Memory: the loaded models, the site baselines, and
  // species profiles derived from the detection record.
  loaders.brain = function () {
    loadBrainModels();
    loadBrainLearning();
    loadBrainSkills();
  };

  function loadBrainModels() {
    var host = region("brain-models");
    clear(host);
    host.appendChild(el("p", { class: "empty-state", text: "Loading models and memory." }));
    Promise.all([
      apiGet("/brain/models"),
      apiGet("/brain/memory" + query({ station_id: state.stationId })),
      apiGet("/detections" + query({ station_id: state.stationId, limit: 500 }))
    ]).then(function (res) {
      var models = res[0], memory = res[1], detections = res[2];
      clear(host);

      host.appendChild(el("h3", { text: "Models" }));
      var deskModels = models.desktop_models || {};
      // The language model has its own section below, so it is left out of this
      // list to avoid showing it in two places.
      var mkeys = Object.keys(deskModels).filter(function (k) { return k !== "llm"; });
      var mlist = el("div", { class: "kv-list" });
      if (mkeys.length) {
        mkeys.forEach(function (k) {
          mlist.appendChild(el("div", { class: "kv-row" }, [
            el("span", { class: "kv-key", text: k }),
            el("span", { class: "kv-val", text: describeModel(deskModels[k]) })
          ]));
        });
      } else {
        mlist.appendChild(el("p", { class: "card-note", text: "No desktop models are declared in the configuration." }));
      }
      host.appendChild(mlist);

      host.appendChild(el("h3", { text: "Language model" }));
      var llmHost = el("div", { class: "llm-manager" });
      host.appendChild(llmHost);
      renderLlmManager(llmHost);

      host.appendChild(el("h3", { text: "Site memory" }));
      var baselines = memory.site_baselines || [];
      host.appendChild(el("p", { class: "card-note", text: fmtNum(memory.baseline_count || baselines.length) + " baseline cells. " + (memory.note || "") }));
      host.appendChild(deferredNote("Editing or annotating memory writes desktop-owned records, which arrives with the memory write path."));

      host.appendChild(el("h3", { text: "Species profiles" }));
      host.appendChild(speciesProfiles(detections));
    }).catch(function (e) { setState(host, "empty-state", "Could not load models and memory: " + e.message); });
  }

  function describeModel(m) {
    if (m === null || m === undefined) { return "-"; }
    if (typeof m !== "object") { return String(m); }
    var bits = [];
    if (m.version) { bits.push("version " + m.version); }
    if (m.path) { bits.push(m.path); }
    if (m.citation) { bits.push(m.citation); }
    return bits.join(" · ") || JSON.stringify(m);
  }

  // The desktop language model that powers the dream pass and interpretation:
  // what is installed, which one is active, and controls to select a model or
  // learn how to drop a new one in. A model change applies on the next start.
  function renderLlmManager(host) {
    clear(host);
    host.appendChild(el("p", { class: "card-note", text: "Loading language model." }));
    apiGet("/brain/llm").then(function (info) {
      clear(host);
      host.appendChild(el("p", { class: "card-note", text:
        (info.runtime_available
          ? "The model runtime is installed. "
          : "The model runtime (llama-cpp-python) is not installed yet, so the model cannot run until it is. ") +
        (info.note || "") }));

      var available = info.available || [];
      if (!available.length) {
        host.appendChild(el("p", { class: "card-note", text:
          "No GGUF model is installed. Drop a .gguf file into " + info.directory + " and reload." }));
        return;
      }

      var list = el("div", { class: "kv-list" });
      available.forEach(function (m) {
        var isActive = m.name === info.active;
        list.appendChild(el("div", { class: "kv-row" }, [
          el("span", { class: "kv-key" }, [
            el("span", { text: m.name }),
            isActive ? badge("active", "source") : null
          ]),
          el("span", { class: "kv-val" }, [
            el("span", { class: "card-meta", text: fmtBytes(m.size_bytes) }),
            isActive ? null : el("button", {
              type: "button", class: "btn btn-small", text: "Use this model",
              onclick: function () { selectLlm(host, m.name); }
            })
          ])
        ]));
      });
      host.appendChild(list);
      host.appendChild(el("p", { class: "form-hint", text:
        "To add a model, drop a .gguf file into " + info.directory +
        " and reload. Changing the model applies the next time the station starts." }));
    }).catch(function (e) { setState(host, "card-note", "Could not load the language model: " + e.message); });
  }

  function selectLlm(host, name) {
    apiSend("/brain/llm/select", "POST", { name: name }).then(function () {
      renderLlmManager(host);
    }).catch(function (e) { window.alert("Could not select the model: " + e.message); });
  }

  function fmtBytes(n) {
    if (n === null || n === undefined) { return "size unknown"; }
    var units = ["B", "KB", "MB", "GB"];
    var v = Number(n), i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
    return v.toFixed(i ? 1 : 0) + " " + units[i];
  }

  // Build one profile card per taxon from the detection record: how many
  // events, the mean and best confidence, and when it was last seen.
  function speciesProfiles(detections) {
    var byTaxon = {};
    detections.forEach(function (obs) {
      (obs.vision_detections || []).forEach(function (det) {
        var key = taxonName(det);
        if (!byTaxon[key]) { byTaxon[key] = { name: key, scientific: det.scientific_name || "", count: 0, sum: 0, best: 0, last: null }; }
        var t = byTaxon[key];
        t.count += 1;
        if (det.confidence !== null && det.confidence !== undefined) {
          t.sum += Number(det.confidence);
          t.best = Math.max(t.best, Number(det.confidence));
        }
        if (!t.last || (obs.first_seen && obs.first_seen > t.last)) { t.last = obs.first_seen; }
      });
    });
    var taxa = Object.keys(byTaxon).map(function (k) { return byTaxon[k]; })
      .sort(function (a, b) { return b.count - a.count; });
    if (!taxa.length) { return el("p", { class: "card-note", text: "No resolved taxa in the record yet." }); }
    var grid = el("div", { class: "card-grid" });
    taxa.forEach(function (t) {
      grid.appendChild(el("article", { class: "card" }, [
        el("div", { class: "card-title", text: t.name }),
        t.scientific ? el("div", { class: "card-meta", text: t.scientific }) : null,
        el("div", { class: "card-stats", text:
          "detections: " + fmtNum(t.count) +
          "  mean confidence: " + fmtConfidence(t.count ? t.sum / t.count : null) +
          "  best: " + fmtConfidence(t.best) }),
        el("div", { class: "card-meta", text: "last seen: " + fmtTime(t.last) })
      ]));
    });
    return grid;
  }

  // Brain, Learning: the live longitudinal status with pause and resume, the
  // candidate patterns it has produced, and an audit summary.
  function loadBrainLearning() {
    var host = region("brain-learning");
    clear(host);
    host.appendChild(el("p", { class: "empty-state", text: "Loading learning and audit history." }));
    Promise.all([
      apiGet("/dream/status"),
      apiGet("/brain/learning"),
      apiGet("/analytics" + query({ station_id: state.stationId }))
    ]).then(function (res) {
      var status = res[0], learning = res[1], audit = res[2];
      clear(host);

      host.appendChild(el("h3", { text: "Longitudinal pass" }));
      host.appendChild(dreamStatusView(status));

      host.appendChild(el("h3", { text: "Audit" }));
      host.appendChild(metricRow([
        metricCard("Events", fmtNum(audit.total_events)),
        metricCard("Verified share", fmtPct(audit.verified_fraction)),
        metricCard("Species", fmtNum(audit.species_richness))
      ]));
      host.appendChild(deferredNote("Detection retraining, acoustic training, and fine-tuning are guided export workflows that arrive with the learning write path."));

      host.appendChild(el("h3", { text: "Candidate patterns" }));
      var patterns = learning.patterns || [];
      if (!patterns.length) { host.appendChild(el("p", { class: "card-note", text: "No candidate patterns yet." })); }
      else {
        var grid = el("div", { class: "card-grid" });
        patterns.forEach(function (p) {
          grid.appendChild(el("article", { class: "card" }, [
            el("div", { class: "badge-row" }, [badge("candidate hypothesis", "muted"), badge(p.dream_phase || "", "source"), badge(p.status || "candidate", "status")]),
            el("div", { class: "card-title", text: p.description || p.pattern_type || "pattern" }),
            el("div", { class: "card-stats", text:
              "effect: " + fmtNum(p.effect_size, 2) + " " + (p.effect_size_type || "") +
              "  n: " + fmtNum(p.n) +
              "  span: " + fmtTime(p.data_span_start) + " to " + fmtTime(p.data_span_end) }),
            (p.supporting_observation_ids && p.supporting_observation_ids.length)
              ? el("div", { class: "card-meta", text: fmtNum(p.supporting_observation_ids.length) + " supporting observations" })
              : null
          ]));
        });
        host.appendChild(grid);
      }
    }).catch(function (e) { setState(host, "empty-state", "Could not load learning: " + e.message); });
  }

  function dreamStatusView(status) {
    var active = status.active;
    var wrap = el("div", { class: "dream-status" });
    if (!active) {
      var passes = status.passes || [];
      wrap.appendChild(el("p", { class: "card-note", text: passes.length ? "No pass is running. Most recent phase reached: " + (passes[0].phase_reached || "unknown") + "." : "No longitudinal pass has run yet." }));
      return wrap;
    }
    wrap.appendChild(el("div", { class: "card-stats", text:
      "phase: " + (active.phase_reached || "unknown") +
      "  cycle: " + fmtNum(active.cycles_completed) +
      "  status: " + (active.status || "unknown") }));
    var controls = el("div", { class: "control-row" });
    var pause = el("button", { type: "button", class: "btn", text: "Pause after this cycle" });
    pause.addEventListener("click", function () {
      pause.disabled = true;
      apiSend("/dream/" + encodeURIComponent(active.id) + "/pause", "POST").then(loaders.brain).catch(function (e) {
        pause.disabled = false; window.alert("Could not pause: " + e.message);
      });
    });
    var resume = el("button", { type: "button", class: "btn", text: "Resume" });
    resume.addEventListener("click", function () {
      resume.disabled = true;
      apiSend("/dream/" + encodeURIComponent(active.id) + "/resume", "POST").then(loaders.brain).catch(function (e) {
        resume.disabled = false; window.alert("Could not resume: " + e.message);
      });
    });
    if (active.status === "paused") { controls.appendChild(resume); }
    else { controls.appendChild(pause); }
    wrap.appendChild(controls);
    return wrap;
  }

  // Brain, Skills: user-authored rules, filterable by tier, with create and edit.
  // Skills are always written by a person here; nothing on this panel generates
  // one. Each skill carries a type that decides which tier evaluates it, which is
  // what keeps a measured flag and an interpretive note on their proper sides.
  //
  // The Brain panel refreshes on a timer for its live parts. This flag tells that
  // refresh to leave an open create or edit form alone, so typing is never wiped
  // out by a background reload.
  var skillFormOpen = false;

  function loadBrainSkills() {
    var host = region("brain-skills");
    var filters = region("brain-skills-filters");
    if (filters && !filters.hasChildNodes()) {
      var sel = el("select", { class: "filter-station", "aria-label": "Skill tier" });
      [["", "All tiers"], ["deterministic_flag", "Field, deterministic flag"], ["interpretive", "Desktop, interpretive"]].forEach(function (o) {
        sel.appendChild(el("option", { value: o[0], text: o[1] }));
      });
      sel.addEventListener("change", function () { renderSkills(host, sel.value); });
      filters.appendChild(el("label", { class: "filter-field" }, [el("span", { text: "Tier" }), sel]));
      filters.appendChild(el("button", {
        type: "button", class: "btn btn-primary", text: "New skill",
        onclick: function () { showSkillForm(host, sel.value, null); }
      }));
    }
    // A background refresh must not close a form the person is filling in.
    if (!skillFormOpen) { renderSkills(host, ""); }
  }

  function renderSkills(host, tier) {
    // Showing the list means no form is open, so the background refresh may
    // repaint the panel again.
    skillFormOpen = false;
    clear(host);
    host.appendChild(el("p", { class: "empty-state", text: "Loading skills." }));
    apiGet("/brain/skills" + query({ tier: tier })).then(function (skills) {
      clear(host);
      if (!skills.length) { setState(host, "empty-state", "No skills defined yet. Use New skill to author one."); return; }
      var grid = el("div", { class: "card-grid" });
      skills.forEach(function (s) {
        grid.appendChild(el("article", { class: "card" }, [
          el("div", { class: "badge-row" }, [badge(s.tier === "interpretive" ? "desktop" : "field", "source"), badge(s.tier, "status")]),
          el("div", { class: "card-title", text: s.title }),
          el("div", { class: "card-stats", text: "When: " + s.trigger_condition }),
          el("div", { class: "card-stats", text: "Do: " + s.instruction }),
          el("div", { class: "card-actions" }, [
            el("button", { type: "button", class: "btn btn-small", text: "Edit", onclick: function () { showSkillForm(host, tier, s); } }),
            el("button", { type: "button", class: "btn btn-small", text: "Delete", onclick: function () { deleteSkill(host, tier, s); } })
          ])
        ]));
      });
      host.appendChild(grid);
    }).catch(function (e) { setState(host, "empty-state", "Could not load skills: " + e.message); });
  }

  // The create and edit form. With no existing skill it authors a new one;
  // given one it edits it in place, keeping the skill's identity.
  function showSkillForm(host, tier, existing) {
    // Mark a form as open so the background refresh does not repaint over it.
    skillFormOpen = true;
    clear(host);
    var isEdit = !!existing;

    function field(labelText, control) {
      return el("label", { class: "form-field" }, [el("span", { class: "form-label", text: labelText }), control]);
    }

    var titleInput = el("input", { type: "text", class: "form-input", maxLength: 200, value: existing ? existing.title : "" });
    var triggerInput = el("textarea", { class: "form-input", rows: 2 });
    triggerInput.value = existing ? existing.trigger_condition : "";
    var instructionInput = el("textarea", { class: "form-input", rows: 3 });
    instructionInput.value = existing ? existing.instruction : "";

    var tierSelect = el("select", { class: "form-input", "aria-label": "Skill type" });
    [["deterministic_flag", "Field, deterministic flag (runs on the station)"],
     ["interpretive", "Desktop, interpretive (runs on the desktop)"]].forEach(function (o) {
      var opt = el("option", { value: o[0], text: o[1] });
      if (existing && existing.tier === o[0]) { opt.selected = true; }
      tierSelect.appendChild(opt);
    });

    var message = el("p", { class: "form-message" });
    var save = el("button", { type: "button", class: "btn btn-primary", text: isEdit ? "Save changes" : "Create skill" });
    save.addEventListener("click", function () {
      var body = {
        title: titleInput.value.trim(),
        trigger_condition: triggerInput.value.trim(),
        instruction: instructionInput.value.trim(),
        tier: tierSelect.value
      };
      if (!body.title || !body.trigger_condition || !body.instruction) {
        message.textContent = "Title, trigger, and instruction are all required.";
        return;
      }
      save.disabled = true;
      message.textContent = "Saving.";
      var req = isEdit
        ? apiSend("/brain/skills/" + encodeURIComponent(existing.id), "PUT", body)
        : apiSend("/brain/skills", "POST", body);
      req.then(function () { renderSkills(host, tier); })
         .catch(function (e) { save.disabled = false; message.textContent = "Could not save: " + e.message; });
    });
    var cancel = el("button", { type: "button", class: "btn", text: "Cancel", onclick: function () { renderSkills(host, tier); } });

    host.appendChild(el("div", { class: "skill-form card" }, [
      el("div", { class: "card-title", text: isEdit ? "Edit skill" : "New skill" }),
      field("Title", titleInput),
      field("When to use (trigger)", triggerInput),
      field("How to apply (instruction)", instructionInput),
      field("Type", tierSelect),
      el("p", { class: "form-hint", text: "A deterministic flag is a pure function of a measured value and runs on the station. An interpretive skill reasons on the desktop, and its output is recorded as labeled inference." }),
      message,
      el("div", { class: "form-actions" }, [save, cancel])
    ]));
  }

  function deleteSkill(host, tier, s) {
    if (!window.confirm('Delete the skill "' + s.title + '"?')) { return; }
    apiSend("/brain/skills/" + encodeURIComponent(s.id), "DELETE").then(function () {
      renderSkills(host, tier);
    }).catch(function (e) { setState(host, "empty-state", "Could not delete: " + e.message); });
  }

  // Reports: existing bundles, and a form that asks the backend to produce a
  // new one in the background.
  loaders.reports = function () {
    var actions = region("reports-actions");
    if (actions && !actions.hasChildNodes()) { actions.appendChild(reportForm()); }
    var host = region("reports-list");
    clear(host);
    host.appendChild(el("p", { class: "empty-state", text: "Loading reports." }));
    apiGet("/reports").then(function (r) {
      clear(host);
      var bundles = r.bundles || [];
      if (!bundles.length) { setState(host, "empty-state", "No reports generated yet."); return; }
      bundles.forEach(function (b) {
        var files = el("div", { class: "file-row" });
        (b.files || []).forEach(function (f) {
          files.appendChild(el("a", { class: "file-link", href: API + "/reports/file" + query({ path: b.name + "/" + f }), text: f, target: "_blank", rel: "noopener" }));
        });
        host.appendChild(el("article", { class: "card" }, [
          el("div", { class: "card-title", text: b.name }),
          el("div", { class: "card-meta", text: "generated: " + fmtTime(b.modified_utc) }),
          files
        ]));
      });
    }).catch(function (e) { setState(host, "empty-state", "Could not load reports: " + e.message); });
  };

  function reportForm() {
    var form = el("form", { class: "report-form" });
    var start = el("input", { type: "date", "aria-label": "Start date" });
    var end = el("input", { type: "date", "aria-label": "End date" });
    var submit = el("button", { type: "submit", class: "btn btn-primary", text: "Generate report" });
    var note = el("span", { class: "form-note" });
    form.appendChild(el("label", { class: "filter-field" }, [el("span", { text: "From" }), start]));
    form.appendChild(el("label", { class: "filter-field" }, [el("span", { text: "To" }), end]));
    form.appendChild(submit);
    form.appendChild(note);
    form.appendChild(deferredNote("Targeting a report at one species or an environmental focus arrives with the report write path."));
    form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      submit.disabled = true;
      note.textContent = "Requesting generation.";
      apiSend("/reports", "POST", {
        station_id: state.stationId || null,
        start: start.value || null,
        end: end.value || null
      }).then(function (r) {
        note.textContent = r.note || "Generation started.";
        submit.disabled = false;
        window.setTimeout(loaders.reports, 1500);
      }).catch(function (e) {
        note.textContent = "Could not start generation: " + e.message;
        submit.disabled = false;
      });
    });
    return form;
  }

  // A plain-language description shown at the top of each settings group, so a
  // reader who has never seen the configuration knows what the group is for.
  var GROUP_DESC = {
    "settings-stations": "The field stations this hub manages, with their environment and sensors.",
    "settings-sensors": "Which inputs each station records.",
    "settings-capture": "How images and audio are captured and stored.",
    "settings-models": "The models this hub uses and where they are stored.",
    "settings-schedules": "When reports and the longitudinal pass run.",
    "settings-credentials": "Whether species-data credentials are in place.",
    "settings-analysis": "Where per-observation analysis runs.",
    "settings-storage": "Buffer limits and how the field buffer is kept."
  };

  // Settings subjects shown one at a time through a tab bar, so the page stays
  // short: the person picks a subject and sees only that subject. The tabs are
  // built from the subject sections already in the page, so adding or removing a
  // section needs no change here.
  var settingsTabs = [];

  function buildSettingsSubnav() {
    var nav = region("settings-subnav");
    if (!nav) { return; }
    clear(nav);
    settingsTabs = [];
    var groupsContainer = $(".settings-groups");
    if (groupsContainer) { groupsContainer.classList.add("tabbed"); }

    $all(".settings-group").forEach(function (group) {
      var heading = $("h3", group);
      var label = heading ? heading.textContent : "Settings";
      var key = group.getAttribute("aria-labelledby") || label;
      var btn = el("button", { type: "button", class: "subnav-item", text: label });
      btn.addEventListener("click", function () { showSettingsGroup(key); });
      nav.appendChild(btn);
      settingsTabs.push({ key: key, group: group, btn: btn });
    });

    var saved = store(STORE.settingsGroup);
    var hasSaved = settingsTabs.some(function (t) { return t.key === saved; });
    showSettingsGroup(hasSaved ? saved : (settingsTabs[0] && settingsTabs[0].key));
  }

  function showSettingsGroup(key) {
    settingsTabs.forEach(function (t) {
      var active = t.key === key;
      t.group.hidden = !active;
      if (active) { t.btn.setAttribute("aria-current", "page"); }
      else { t.btn.removeAttribute("aria-current"); }
    });
    if (key) { store(STORE.settingsGroup, key); }
  }

  // Settings: an appearance chooser and a display timezone that persist in the
  // browser, plus a friendly, labeled view of the current configuration.
  loaders.settings = function () {
    buildSettingsSubnav();
    renderAppearance();
    renderTimezone();
    apiGet("/settings").then(function (s) {
      var cfg = s.config || {};
      var stations = cfg.stations || [];
      var sensorsByStation = {};
      stations.forEach(function (st) { sensorsByStation[st.station_name || st.station_id] = st.sensors; });

      renderGroup("settings-stations", stations, true);
      renderGroup("settings-sensors", stations.length ? sensorsByStation : null, true);
      renderGroup("settings-capture", cfg.media || cfg.capture, true);
      renderGroup("settings-models", cfg.desktop_models, true);
      renderGroup("settings-schedules", cfg.schedules, true);
      renderGroup("settings-credentials", { credentials_configured: !!s.secrets_configured }, true);
      renderGroup("settings-analysis", cfg.analysis, true);
      renderGroup("settings-storage", cfg.buffer || cfg.storage, true);

      var guide = region("settings-guide");
      if (guide) {
        clear(guide);
        guide.appendChild(el("p", { class: "card-note", text: "Connect a station, confirm its sensors are on, then open Detections to watch events arrive as they happen." }));
      }
    }).catch(function () {
      ["settings-stations", "settings-sensors", "settings-capture", "settings-models", "settings-schedules", "settings-credentials", "settings-analysis", "settings-storage"].forEach(function (r) {
        var h = region(r); if (h) { setState(h, "card-note", "Configuration is unavailable."); }
      });
    });
  };

  // Render one settings group: a description, then the values as labeled rows
  // with switches for on and off fields, and a note that this view is read-only
  // for now.
  function renderGroup(regionName, value, readonly) {
    var host = region(regionName);
    if (!host) { return; }
    clear(host);
    if (GROUP_DESC[regionName]) { host.appendChild(el("p", { class: "settings-desc", text: GROUP_DESC[regionName] })); }
    var empty = value === undefined || value === null ||
      (Array.isArray(value) && !value.length) ||
      (typeof value === "object" && !Array.isArray(value) && !Object.keys(value).length);
    if (empty) { host.appendChild(el("p", { class: "card-note", text: "Not configured." })); return; }
    appendFields(host, value);
    if (readonly) { host.appendChild(el("p", { class: "card-note", text: "Shows your saved configuration. Changing it here arrives with the settings update step." })); }
  }

  function humanize(key) {
    return String(key).replace(/_/g, " ").replace(/^\w/, function (c) { return c.toUpperCase(); });
  }

  function toggleView(on) {
    return el("span", { class: "toggle" + (on ? " on" : ""), "aria-hidden": "true" }, [el("span", { class: "toggle-knob" })]);
  }

  function fieldRow(label, value) {
    var valNode;
    if (typeof value === "boolean") { valNode = toggleView(value); }
    else if (value === null || value === "") { valNode = el("span", { class: "field-value muted", text: "not set" }); }
    else { valNode = el("span", { class: "field-value", text: String(value) }); }
    return el("div", { class: "field-row" }, [el("span", { class: "field-label", text: label }), valNode]);
  }

  // Walk a configuration value and render it as labeled rows. Nested objects
  // become titled sub-groups; booleans become switches; empty values read
  // plainly, so nothing appears as raw code.
  function appendFields(container, value) {
    if (value === null || value === undefined) { container.appendChild(el("p", { class: "card-note", text: "Not set." })); return; }
    if (Array.isArray(value)) {
      if (!value.length) { container.appendChild(el("p", { class: "card-note", text: "None." })); return; }
      // Each entry (a station, a channel) becomes a uniform, rounded card that is
      // collapsed by default, and the cards tile several across, so a long list
      // of stations or channels reads as a short, scannable row rather than a
      // wall of nested fields. Native details elements handle open and close, so
      // nothing here can leave a card half-open.
      var grid = el("div", { class: "config-grid" });
      value.forEach(function (item, i) {
        var title = (item && (item.station_name || item.name || item.id)) || ("Item " + (i + 1));
        var card = el("details", { class: "config-card" }, [el("summary", { text: title })]);
        if (item !== null && typeof item === "object") { appendFields(card, item); }
        else { card.appendChild(fieldRow("Value", item)); }
        grid.appendChild(card);
      });
      container.appendChild(grid);
      return;
    }
    if (typeof value === "object") {
      Object.keys(value).forEach(function (k) {
        var v = value[k];
        if (v !== null && typeof v === "object") {
          var sub = el("div", { class: "subgroup" }, [el("h4", { text: humanize(k) })]);
          appendFields(sub, v);
          container.appendChild(sub);
        } else {
          container.appendChild(fieldRow(humanize(k), v));
        }
      });
      return;
    }
    container.appendChild(fieldRow("Value", value));
  }

  function renderAppearance() {
    var host = region("settings-appearance");
    if (!host) { return; }
    clear(host);
    var sel = el("select", { id: "appearance-theme", "aria-label": "Theme" });
    Object.keys(THEMES).forEach(function (name) {
      var o = el("option", { value: name, text: THEMES[name].label + " (" + THEMES[name].mode + ")" });
      if (name === currentTheme()) { o.selected = true; }
      sel.appendChild(o);
    });
    sel.addEventListener("change", function () { applyTheme(sel.value); });
    host.appendChild(el("label", { class: "filter-field" }, [el("span", { text: "Theme" }), sel]));
    host.appendChild(el("p", { class: "card-note", text: "Your theme is remembered on this computer." }));
  }

  function renderTimezone() {
    var host = region("settings-time");
    if (!host) { return; }
    clear(host);
    host.appendChild(el("p", { class: "settings-desc", text: "Times are stored in UTC and shown in the zone you choose here." }));
    var input = el("input", { type: "text", id: "appearance-tz", value: displayTimezone(), "aria-label": "Display timezone" });
    var save = el("button", { type: "button", class: "btn", text: "Use timezone" });
    save.addEventListener("click", function () {
      store(STORE.timezone, input.value.trim());
      if (state.activePanel) { loaders[state.activePanel](); }
    });
    var row = el("div", { class: "tz-row" }, [
      el("label", { class: "filter-field" }, [el("span", { text: "Display timezone" }), input]),
      save
    ]);
    host.appendChild(row);
    host.appendChild(el("p", { class: "card-note", text: "Display only, and remembered on this computer." }));
  }

  // ----------------------------------------------------------------------
  // 9. Startup
  // ----------------------------------------------------------------------

  function start() {
    initTheme();
    initRouter();

    apiGet("/stations").then(function (list) {
      state.stations = list || [];
    }).catch(function () {
      state.stations = [];
    }).then(function () {
      refreshHealth();
      window.setInterval(refreshHealth, HEALTH_INTERVAL_MS);
      var sub = store(STORE.subpanel);
      if (sub) { activateSubpanel(sub); }
      activatePanel(store(STORE.panel) || "detections");
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
