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

  // The "All" page size: large enough to return every match at Audtheia's scale
  // while staying within the endpoint's accepted limit.
  var PAGE_ALL = 100000;

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
    "cloud-dancer": { label: "Cloud Dancer", mode: "light" },
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
  // Surface the backend's own message on failure (FastAPI returns it under
  // "detail"), so a person sees why a request was refused, not just its status.
  function apiError(r) {
    return r.json().then(
      function (data) { throw new Error(data && data.detail ? data.detail : "request failed (" + r.status + ")"); },
      function () { throw new Error("request failed (" + r.status + ")"); }
    );
  }

  function apiGet(path) {
    return fetch(API + path, { headers: { "Accept": "application/json" } }).then(function (r) {
      if (!r.ok) { return apiError(r); }
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
      if (!r.ok) { return apiError(r); }
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

    // An optional species dropdown, sitting beside the station one. Its options
    // are filled in by the detections loader from the species actually present,
    // so the same control adapts to whatever a deployment has recorded.
    if (opts.species) {
      var sp = el("select", { class: "filter-species", "aria-label": "Species" });
      sp.appendChild(el("option", { value: "", text: "All species" }));
      sp.addEventListener("change", function () {
        state.speciesFilter = sp.value;
        onChange();
      });
      bar.appendChild(el("label", { class: "filter-field" }, [el("span", { text: "Species" }), sp]));
    }

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

  // The add-a-source control for the Detections and Audio panels. A detection or
  // audio source on the desktop is a station's desktop capture source (a webcam,
  // stream, URL, or file), which is why this sets capture.source rather than any
  // physical sensor. Physical environmental sensors and GPS are Pi-side and are
  // configured per station under Sensors, so they are deliberately not here.
  function captureSourceControl(kind, reload) {
    var wrap = el("div", { class: "source-control" });
    var btn = el("button", { type: "button", class: "btn", text: kind === "audio" ? "Set audio source" : "Set capture source" });
    btn.addEventListener("click", function () { toggleSourceForm(wrap, kind, reload); });
    wrap.appendChild(btn);
    return wrap;
  }

  function toggleSourceForm(wrap, kind, reload) {
    var open = wrap.querySelector(".source-form");
    if (open) { wrap.removeChild(open); return; }
    var form = el("div", { class: "source-form card" });
    var msg = el("p", { class: "form-message" });
    var stSel = el("select", { class: "form-input" });
    stSel.appendChild(el("option", { value: "", text: "Loading stations." }));
    apiGet("/settings").then(function (s) {
      clear(stSel);
      var stations = (s.config && s.config.stations) || [];
      if (!stations.length) { stSel.appendChild(el("option", { value: "", text: "No stations yet. Add one under Settings, Stations." })); return; }
      stations.forEach(function (st) {
        var o = el("option", { value: st.station_id, text: st.station_name || st.station_id });
        if (st.station_id === state.stationId) { o.selected = true; }
        stSel.appendChild(o);
      });
    }).catch(function () { clear(stSel); stSel.appendChild(el("option", { value: "", text: "Could not load stations." })); });
    var srcIn = el("input", { type: "text", class: "form-input" });
    srcIn.placeholder = kind === "audio" ? "leave blank for none" : "webcam:0, a YouTube or stream link, or file:C:/clip.mp4";
    var field = kind === "audio" ? "capture_source_audio" : "capture_source_video";
    var save = el("button", { type: "button", class: "btn btn-primary", text: "Save" });
    save.addEventListener("click", function () {
      var sid = stSel.value;
      if (!sid) { msg.textContent = "Choose a station."; return; }
      save.disabled = true; msg.textContent = "Saving.";
      apiSend("/settings/update", "POST", { changes: [{ scope: "station", station_id: sid, field: field, value: srcIn.value.trim() || null }] })
        .then(function () { if (wrap.contains(form)) { wrap.removeChild(form); } reload(); })
        .catch(function (e) { save.disabled = false; msg.textContent = "Could not save: " + e.message; });
    });
    var cancel = el("button", { type: "button", class: "btn", text: "Cancel", onclick: function () { wrap.removeChild(form); } });
    form.appendChild(el("div", { class: "card-title", text: kind === "audio" ? "Desktop audio source" : "Desktop capture source" }));
    form.appendChild(el("label", { class: "form-field" }, [el("span", { class: "form-label", text: "Station" }), stSel]));
    form.appendChild(el("label", { class: "form-field" }, [el("span", { class: "form-label", text: kind === "audio" ? "Audio source" : "Video source" }), srcIn]));
    form.appendChild(el("p", { class: "form-hint", text: kind === "audio"
      ? "This is the desktop hardware-free audio source. Physical microphones and hydrophones are wired to a Pi and set per station under Sensors."
      : "This runs desktop capture without a Pi. Live detection also needs a desktop model; set its path under Settings, Model paths, and place the file under models/." }));
    form.appendChild(msg);
    form.appendChild(el("div", { class: "form-actions" }, [save, cancel]));
    wrap.appendChild(form);
  }

  // Start and stop desktop capture from the interface, per station, so detection
  // runs without a terminal. Only stations with a capture source are listed.
  function captureRunControl(reload) {
    var wrap = el("div", { class: "capture-control" });
    var btn = el("button", { type: "button", class: "btn", text: "Capture" });
    btn.addEventListener("click", function () { toggleCapturePanel(wrap, reload); });
    wrap.appendChild(btn);
    return wrap;
  }

  function toggleCapturePanel(wrap, reload) {
    var open = wrap.querySelector(".capture-panel");
    if (open) { wrap.removeChild(open); return; }
    openCapturePanel(wrap, reload);
  }

  function openCapturePanel(wrap, reload) {
    var existing = wrap.querySelector(".capture-panel");
    if (existing) { wrap.removeChild(existing); }
    var panel = el("div", { class: "capture-panel card" });
    panel.appendChild(el("div", { class: "card-title", text: "Desktop capture" }));
    var body = el("div");
    panel.appendChild(body);
    body.appendChild(el("p", { class: "card-note", text: "Loading." }));
    wrap.appendChild(panel);
    Promise.all([apiGet("/settings"), apiGet("/capture/status")]).then(function (res) {
      var cfg = res[0].config || {};
      var running = res[1].running || [];
      clear(body);
      var stations = (cfg.stations || []).filter(function (st) { return ((st.capture && st.capture.source) || {}).video; });
      if (!stations.length) {
        body.appendChild(el("p", { class: "card-note", text: "No station has a capture source yet. Use Set capture source first." }));
        return;
      }
      stations.forEach(function (st) {
        var isRun = running.indexOf(st.station_id) !== -1;
        var toggle = el("button", { type: "button", class: "btn btn-small" + (isRun ? "" : " btn-primary"), text: isRun ? "Stop" : "Start" });
        toggle.addEventListener("click", function () {
          toggle.disabled = true;
          var url = "/capture/" + encodeURIComponent(st.station_id) + (isRun ? "/stop" : "/start");
          apiSend(url, "POST").then(function (r) {
            if (r && r.warning) { window.alert(r.warning); }
            openCapturePanel(wrap, reload);
            reload();
          }).catch(function (e) { toggle.disabled = false; window.alert("Could not " + (isRun ? "stop" : "start") + " capture: " + e.message); });
        });
        body.appendChild(el("div", { class: "capture-row" }, [
          el("span", { class: "capture-row-name", text: (st.station_name || st.station_id) + " . " + (((st.capture.source) || {}).video || "") }),
          isRun ? badge("capturing", "source") : null,
          toggle
        ]));
      });
      body.appendChild(el("p", { class: "form-hint", text: "Start opens the source and runs detection; detections appear below as they are found. A desktop model must be set for anything to be detected." }));
    }).catch(function (e) { clear(body); body.appendChild(el("p", { class: "card-note", text: "Could not load capture: " + e.message })); });
  }

  // Refill the species dropdown from the names present in the current load,
  // keeping the active choice selected even when nothing matches it this time,
  // so a chosen species does not silently reset on refresh.
  function populateSpecies(names) {
    var select = document.querySelector(".filter-species");
    if (!select) { return; }
    var current = state.speciesFilter || "";
    clear(select);
    select.appendChild(el("option", { value: "", text: "All species" }));
    var found = false;
    names.forEach(function (n) {
      var o = el("option", { value: n, text: n });
      if (n === current) { o.selected = true; found = true; }
      select.appendChild(o);
    });
    if (current && !found) {
      var o = el("option", { value: current, text: current });
      o.selected = true;
      select.appendChild(o);
    }
  }

  // The Delete control: a button that turns the detections grid into a selection
  // mode, and a panel (opened like the capture panel) with Select all, a live
  // count, and Delete selected. Nothing is removed until Delete selected is
  // confirmed, so entering the mode is always safe.
  function deleteControl(reload) {
    var wrap = el("div", { class: "delete-control" });
    var btn = el("button", { type: "button", class: "btn", text: "Delete" });
    btn.addEventListener("click", function () { toggleDeleteMode(wrap, reload); });
    wrap.appendChild(btn);
    return wrap;
  }

  function toggleDeleteMode(wrap, reload) {
    state.selecting = !state.selecting;
    if (!state.selecting) { state.selection = {}; }
    var btn = wrap.querySelector(".btn");
    if (btn) {
      btn.textContent = state.selecting ? "Cancel" : "Delete";
      btn.classList.toggle("btn-danger", state.selecting);
    }
    var panel = wrap.querySelector(".delete-panel");
    if (state.selecting && !panel) {
      wrap.appendChild(buildDeletePanel(wrap, reload));
    } else if (!state.selecting && panel) {
      wrap.removeChild(panel);
    }
    reload();
  }

  function buildDeletePanel(wrap, reload) {
    var panel = el("div", { class: "delete-panel card" });
    panel.appendChild(el("div", { class: "card-title", text: "Delete detections" }));

    var selectAll = el("button", { type: "button", class: "btn btn-small select-all", text: "Select all" });
    selectAll.addEventListener("click", function () {
      var ids = state.detectionIds || [];
      var allSel = ids.length && ids.every(function (id) { return state.selection[id]; });
      state.selection = {};
      if (!allSel) { ids.forEach(function (id) { state.selection[id] = true; }); }
      reload();
    });

    var count = el("span", { class: "delete-count", text: "0 selected" });

    var del = el("button", { type: "button", class: "btn btn-small btn-danger delete-selected", text: "Delete selected" });
    del.disabled = true;
    del.addEventListener("click", function () {
      var ids = Object.keys(state.selection).filter(function (id) { return state.selection[id]; });
      if (!ids.length) { return; }
      if (!window.confirm("Delete " + ids.length + " detection" + (ids.length > 1 ? "s" : "") +
        " and their stored frames? This cannot be undone.")) { return; }
      del.disabled = true;
      apiSend("/detections/delete", "POST", { ids: ids }).then(function () {
        state.selection = {};
        state.selecting = false;
        var b = wrap.querySelector(".btn");
        if (b) { b.textContent = "Delete"; b.classList.remove("btn-danger"); }
        var p = wrap.querySelector(".delete-panel");
        if (p) { wrap.removeChild(p); }
        reload();
      }).catch(function (e) { del.disabled = false; window.alert("Could not delete: " + e.message); });
    });

    panel.appendChild(el("div", { class: "delete-panel-row" }, [selectAll, count, del]));
    panel.appendChild(el("p", { class: "form-hint", text: "Tick the detections to remove, then Delete selected." }));
    return panel;
  }

  // Keep the delete panel's count, the Delete-selected enabled state, and the
  // Select-all label in step with the current selection after every render.
  function updateDeleteUI() {
    var selected = Object.keys(state.selection).filter(function (id) { return state.selection[id]; });
    var count = document.querySelector(".delete-count");
    if (count) { count.textContent = selected.length + " selected"; }
    var del = document.querySelector(".delete-selected");
    if (del) { del.disabled = selected.length === 0; }
    var selectAll = document.querySelector(".select-all");
    if (selectAll) {
      var ids = state.detectionIds || [];
      var allSel = ids.length && ids.every(function (id) { return state.selection[id]; });
      selectAll.textContent = allSel ? "Unselect all" : "Select all";
    }
  }

  // A stable colour per species, derived from its name — no stored state, so the
  // same species is always the same colour on every machine and session, and
  // there is nothing to persist, migrate, or lose. The name is hashed (FNV-1a) to
  // a hue; saturation and lightness are nudged by the hash for extra separation
  // but kept in a readable band, and the label text colour is chosen for contrast.
  function _hslToRgb(h, s, l) {
    var c = (1 - Math.abs(2 * l - 1)) * s;
    var hp = h / 60;
    var x = c * (1 - Math.abs((hp % 2) - 1));
    var r = 0, g = 0, b = 0;
    if (hp < 1) { r = c; g = x; }
    else if (hp < 2) { r = x; g = c; }
    else if (hp < 3) { g = c; b = x; }
    else if (hp < 4) { g = x; b = c; }
    else if (hp < 5) { r = x; b = c; }
    else { r = c; b = x; }
    var m = l - c / 2;
    return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
  }

  var _speciesColorCache = {};
  function speciesColor(name) {
    name = String(name == null ? "unknown" : name);
    if (_speciesColorCache[name]) { return _speciesColorCache[name]; }
    var h = 2166136261;
    for (var i = 0; i < name.length; i++) {
      h ^= name.charCodeAt(i);
      h = (h * 16777619) >>> 0;
    }
    var hue = h % 360;
    var sat = 0.60 + ((h >>> 9) % 1000) / 1000 * 0.18;   // 0.60–0.78
    var light = 0.50 + ((h >>> 17) % 1000) / 1000 * 0.08; // 0.50–0.58
    var rgb = _hslToRgb(hue, sat, light);
    var lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    var pair = { bg: "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")", fg: lum > 150 ? "#0b1f18" : "#ffffff" };
    _speciesColorCache[name] = pair;
    return pair;
  }

  // Draw each stored detection box over its frame. Boxes are recorded in the
  // frame's own pixels and the saved frame keeps its native size, so once the
  // image has loaded its naturalWidth/Height give the exact scale, and each box
  // is placed as a percentage so it tracks the image at any displayed size. The
  // saved JPEG is never altered; the box is an overlay the interface draws.
  function drawBoxes(container, img, boxes) {
    var old = container.querySelectorAll(".detection-box");
    Array.prototype.forEach.call(old, function (n) { if (n.parentNode) { n.parentNode.removeChild(n); } });
    var nw = img.naturalWidth, nh = img.naturalHeight;
    if (!nw || !nh || !boxes) { return; }
    var wrapW = container.clientWidth || 0;
    boxes.forEach(function (b) {
      if (b.bbox_x == null || b.bbox_y == null || b.bbox_w == null || b.bbox_h == null) { return; }
      var box = el("div", { class: "detection-box" });
      var leftPct = b.bbox_x / nw * 100;
      box.style.cssText = "left:" + leftPct + "%;top:" + (b.bbox_y / nh * 100) +
        "%;width:" + (b.bbox_w / nw * 100) + "%;height:" + (b.bbox_h / nh * 100) + "%;";
      var conf = (b.confidence == null) ? "" : "  " + Math.round(Number(b.confidence) * 100) + "%";
      var label = el("span", { class: "detection-box-label", text: taxonName(b) + conf });
      // Colour the box and its label by species, so each taxon reads distinctly.
      var col = speciesColor(taxonName(b));
      box.style.borderColor = col.bg;
      label.style.backgroundColor = col.bg;
      label.style.color = col.fg;
      box.appendChild(label);
      container.appendChild(box);
      // Keep a long label from spilling past the frame. The label is capped to
      // the frame width, and when its box sits near the right edge the label is
      // shifted left so its right end lands on the image edge instead of running
      // outside the card. The box itself is untouched; only its label moves.
      if (wrapW) {
        label.style.maxWidth = wrapW + "px";
        var boxLeftPx = leftPct / 100 * wrapW;
        var labelW = label.offsetWidth;
        var shift = Math.max(-boxLeftPx, Math.min(0, wrapW - labelW - boxLeftPx));
        if (shift < 0) { label.style.left = shift + "px"; }
      }
    });
  }

  // Attach box drawing to an image, both for the case where it is already
  // decoded and for the normal asynchronous load.
  function withBoxes(container, img, boxes) {
    if (!boxes || !boxes.length) { return; }
    var draw = function () { drawBoxes(container, img, boxes); };
    img.addEventListener("load", draw);
    if (img.complete && img.naturalWidth) { draw(); }
  }

  function detectionFrame(path, caption, boxes, onClick) {
    var src = API + "/media" + query({ path: path });
    var wrap = el("div", { class: "detection-frame-wrap" });
    var img = el("img", { class: "detection-frame", src: src, alt: caption || "detection frame", loading: "lazy" });
    img.addEventListener("click", onClick || function () { openLightbox(src, caption, boxes); });
    wrap.appendChild(img);
    withBoxes(wrap, img, boxes);
    return wrap;
  }

  function openLightbox(src, caption, boxes) {
    var overlay = el("div", { class: "lightbox" });
    overlay.addEventListener("click", function () { if (overlay.parentNode) { overlay.parentNode.removeChild(overlay); } });
    var frame = el("div", { class: "lightbox-frame" });
    var img = el("img", { class: "lightbox-img", src: src, alt: caption || "" });
    frame.appendChild(img);
    withBoxes(frame, img, boxes);
    overlay.appendChild(frame);
    if (caption) { overlay.appendChild(el("div", { class: "lightbox-caption", text: caption })); }
    document.body.appendChild(overlay);
  }

  // The frame-audit modal: a zoom-style view opened from a detection card. It
  // shows every stored frame of the event with its own box and confidence, and a
  // panel deriving the card's numbers (frame count, duration, confidence,
  // salience) from those frames, so a scientist can verify the stats rather than
  // trust them.
  function openFrameAudit(obs) {
    var overlay = el("div", { class: "audit-modal" });
    function close() {
      if (overlay.parentNode) { overlay.parentNode.removeChild(overlay); }
      document.removeEventListener("keydown", onKey);
    }
    function onKey(e) { if (e.key === "Escape") { close(); } }
    overlay.addEventListener("click", function (e) { if (e.target === overlay) { close(); } });
    document.addEventListener("keydown", onKey);

    var panel = el("div", { class: "audit-panel" });
    var closeBtn = el("button", { type: "button", class: "audit-close", "aria-label": "Close", text: "×" });
    closeBtn.addEventListener("click", close);
    panel.appendChild(closeBtn);

    var species = (obs.vision_detections || []).map(taxonName).join(", ") || "no resolved taxon";
    panel.appendChild(el("div", { class: "audit-title", text: species }));
    panel.appendChild(el("div", { class: "card-meta", text: obs.event_name || obs.id }));

    var body = el("div", { class: "audit-body" });
    body.appendChild(el("p", { class: "card-note", text: "Loading frames." }));
    panel.appendChild(body);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    apiGet("/detections/" + encodeURIComponent(obs.id) + "/frames").then(function (data) {
      clear(body);
      var frames = data.frames || [];
      body.appendChild(auditDerivation(obs, frames));
      if (!frames.length) {
        body.appendChild(el("p", { class: "empty-state", text: "No stored frames were found for this event on disk." }));
      } else {
        body.appendChild(auditFrameStrip(frames, species));
      }
    }).catch(function (e) {
      clear(body);
      body.appendChild(el("p", { class: "empty-state", text: "Could not load frames: " + e.message }));
    });
  }

  function auditDerivation(obs, frames) {
    var wrap = el("div", { class: "audit-derivation" });
    var n = frames.length;
    var confs = frames.map(function (f) { return Number(f.confidence) || 0; });
    var maxConf = confs.length ? Math.max.apply(null, confs) : null;
    var times = frames.map(function (f) { return f.captured_at; }).filter(Boolean);
    function row(label, value, ok) {
      var v = el("span", { class: "audit-v", text: value });
      if (ok === true) { v.appendChild(el("span", { class: "audit-ok", text: "  ✓ matches" })); }
      return el("div", { class: "audit-row" }, [el("span", { class: "audit-k", text: label }), v]);
    }
    wrap.appendChild(el("div", { class: "card-title", text: "How these numbers were derived" }));
    wrap.appendChild(row("Tracked across", n + " frames  (= " + n + " saved frames below)",
      obs.frame_count != null && Number(obs.frame_count) === n));
    if (times.length >= 2) {
      var t0 = new Date(times[0]).getTime(), t1 = new Date(times[times.length - 1]).getTime();
      var secs = (t1 - t0) / 1000;
      wrap.appendChild(row("Duration", secs.toFixed(1) + " s  (last frame " + fmtTime(times[times.length - 1]) +
        " − first frame " + fmtTime(times[0]) + ")"));
    }
    if (maxConf != null) {
      wrap.appendChild(row("Confidence", Math.round(maxConf * 100) + "%  (highest of the " + n + " per-frame values)"));
    }
    wrap.appendChild(row("Salience", fmtNum(obs.salience_provisional, 2) +
      "  = D · (0.5·N + 0.5·R), Shannon-surprisal novelty & rarity (docs/salience.md)"));
    wrap.appendChild(el("p", { class: "form-hint", text:
      "Each frame below is a saved detection with its own confidence and box, so the frame count and the true duration are directly verifiable. Confidence is the peak across frames; salience is computed from the whole record at capture." }));
    return wrap;
  }

  function auditFrameStrip(frames, caption) {
    var wrap = el("div", { class: "audit-strip-wrap" });
    wrap.appendChild(el("div", { class: "card-note", text: frames.length + " frames · scroll to review · click any frame to enlarge" }));
    var strip = el("div", { class: "audit-strip" });
    frames.forEach(function (f) {
      var src = API + "/media" + query({ path: f.path });
      var boxes = (f.bbox_w != null) ? [{
        bbox_x: f.bbox_x, bbox_y: f.bbox_y, bbox_w: f.bbox_w, bbox_h: f.bbox_h,
        confidence: f.confidence, common_name: f.class_name
      }] : [];
      var cap = caption + " · frame " + f.index;
      var cell = el("div", { class: "audit-cell" });
      var fw = el("div", { class: "detection-frame-wrap audit-thumb-wrap" });
      var img = el("img", { class: "detection-frame", src: src, alt: cap, loading: "lazy" });
      img.addEventListener("click", function () { openLightbox(src, cap, boxes); });
      fw.appendChild(img);
      withBoxes(fw, img, boxes);
      cell.appendChild(fw);
      cell.appendChild(el("div", { class: "audit-cell-meta", text: "#" + f.index +
        (f.confidence != null ? "  ·  " + Math.round(Number(f.confidence) * 100) + "%" : "") }));
      strip.appendChild(cell);
    });
    wrap.appendChild(strip);
    return wrap;
  }

  // ----------------------------------------------------------------------
  // 4. Application state
  // ----------------------------------------------------------------------

  var state = {
    stations: [],
    stationId: store(STORE.station) || "",
    serverTimezone: null,
    activePanel: null,
    pollTimer: null,
    // Detections view: the chosen species filter, the ids currently shown (so
    // Select all knows its scope), whether the delete selection mode is on, and
    // the set of ids ticked for deletion.
    speciesFilter: "",
    detectionIds: [],
    selecting: false,
    selection: {},
    // Detections paging: page size (PAGE_ALL = every match), current offset, and
    // a signature so changing the filters resets to page one while a plain page
    // turn does not.
    pageSize: 100,
    pageOffset: 0,
    _detSig: null
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
  var LIVE_PANELS = { detections: true, audio: true, brain: true, analytics: true, sensors: true };

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
    if (filters && !filters.hasChildNodes()) {
      filters.appendChild(filterBar(loaders.detections, { species: true }));
      filters.appendChild(captureSourceControl("video", loaders.detections));
      filters.appendChild(captureRunControl(loaders.detections));
      filters.appendChild(deleteControl(loaders.detections));
    }
    // Reset to the first page whenever the station, species, or page size
    // changes; a plain page turn (prev/next) keeps the same signature.
    var sig = state.stationId + "|" + state.speciesFilter + "|" + state.pageSize;
    if (sig !== state._detSig) { state.pageOffset = 0; state._detSig = sig; }

    // Fill the species dropdown from the full server-side list (every recorded
    // species), not just the current page.
    apiGet("/detections/species" + query({ station_id: state.stationId }))
      .then(function (list) { populateSpecies(list || []); })
      .catch(function () {});

    apiGet("/detections" + query({
      station_id: state.stationId,
      species: state.speciesFilter,
      limit: state.pageSize,
      offset: state.pageOffset
    })).then(function (resp) {
      clear(host);
      var rows = (resp && resp.items) || [];
      var total = (resp && resp.total) || 0;
      state.detectionIds = rows.map(function (obs) { return obs.id; });

      if (!rows.length) {
        setState(host, "empty-state", (state.speciesFilter || state.stationId)
          ? "No detections match these filters."
          : "No detections yet. Use Set capture source to run desktop detection, or connect a Pi under Settings, Stations.");
        updateDeleteUI();
        return;
      }

      var grid = el("div", { class: "card-grid" + (state.selecting ? " selecting" : "") });
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
        var card = el("article", { class: "card" });

        // In selection mode each card carries a checkbox in its top-right corner;
        // ticking it adds the observation to the delete set. The checkbox stops
        // the click from reaching the frame so selecting never opens the lightbox.
        if (state.selecting) {
          card.classList.add("selectable");
          var checked = !!state.selection[obs.id];
          if (checked) { card.classList.add("selected"); }
          (function (id, cardEl) {
            var cb = el("button", {
              type: "button",
              class: "card-select" + (checked ? " checked" : ""),
              "aria-label": "Select this detection",
              "aria-pressed": checked ? "true" : "false"
            });
            cb.addEventListener("click", function (ev) {
              ev.stopPropagation();
              if (state.selection[id]) {
                delete state.selection[id];
                cardEl.classList.remove("selected");
                cb.classList.remove("checked");
                cb.setAttribute("aria-pressed", "false");
              } else {
                state.selection[id] = true;
                cardEl.classList.add("selected");
                cb.classList.add("checked");
                cb.setAttribute("aria-pressed", "true");
              }
              updateDeleteUI();
            });
            cardEl.appendChild(cb);
          })(obs.id, card);
        }

        if (obs.representative_frame) {
          card.appendChild(detectionFrame(obs.representative_frame, species, obs.vision_detections,
            (function (o) { return function () { openFrameAudit(o); }; })(obs)));
          card.appendChild(el("div", { class: "frame-note", text: "highest-confidence frame of this event" }));
        }
        card.appendChild(el("div", { class: "card-meta", text: meta }));
        card.appendChild(el("div", { class: "card-title", text: species }));
        card.appendChild(el("div", { class: "card-stats", text:
          "tracked across " + fmtNum(obs.frame_count) + " frames" +
          "  ·  " + fmtNum(obs.duration, 1) + "s" +
          "  ·  confidence " + fmtConfidence(obs.screening_confidence) +
          "  ·  salience " + fmtNum(obs.salience_provisional, 2) }));
        card.appendChild(el("div", { class: "badge-row" }, badges));
        grid.appendChild(card);
      });
      host.appendChild(grid);
      host.appendChild(detectionsPager(total));
      updateDeleteUI();
    }).catch(function (e) { setState(host, "empty-state", "Could not load detections: " + e.message); });
  };

  // The bottom-of-list pager: a page-size selector (20/40/80/100/200/All), a
  // "start–end of total" readout, and Prev/Next. Changing the size or turning a
  // page re-runs the loader, which fetches just that page from the server.
  function detectionsPager(total) {
    var wrap = el("div", { class: "pager" });
    var isAll = state.pageSize >= PAGE_ALL;
    var size = state.pageSize;
    var start = total === 0 ? 0 : state.pageOffset + 1;
    var end = isAll ? total : Math.min(state.pageOffset + size, total);

    var sizeSel = el("select", { class: "pager-size", "aria-label": "Detections per page" });
    [20, 40, 80, 100, 200].forEach(function (n) {
      var o = el("option", { value: String(n), text: String(n) });
      if (!isAll && n === size) { o.selected = true; }
      sizeSel.appendChild(o);
    });
    var allOpt = el("option", { value: "all", text: "All" });
    if (isAll) { allOpt.selected = true; }
    sizeSel.appendChild(allOpt);
    sizeSel.addEventListener("change", function () {
      state.pageSize = sizeSel.value === "all" ? PAGE_ALL : parseInt(sizeSel.value, 10);
      loaders.detections();
    });
    wrap.appendChild(el("label", { class: "filter-field" }, [el("span", { text: "Per page" }), sizeSel]));

    wrap.appendChild(el("span", { class: "pager-info", text: total ? (start + "–" + end + " of " + total) : "0 of 0" }));

    var prev = el("button", { type: "button", class: "btn btn-small", text: "Prev" });
    prev.disabled = isAll || state.pageOffset <= 0;
    prev.addEventListener("click", function () {
      state.pageOffset = Math.max(0, state.pageOffset - size);
      loaders.detections();
    });
    var next = el("button", { type: "button", class: "btn btn-small", text: "Next" });
    next.disabled = isAll || end >= total;
    next.addEventListener("click", function () {
      state.pageOffset = state.pageOffset + size;
      loaders.detections();
    });
    wrap.appendChild(prev);
    wrap.appendChild(next);
    return wrap;
  }

  // Audio: acoustic detections tied to events, with the true clip duration.
  loaders.audio = function () {
    var host = region("audio-list");
    var filters = region("audio-filters");
    if (filters && !filters.hasChildNodes()) {
      filters.appendChild(filterBar(loaders.audio));
      filters.appendChild(captureSourceControl("audio", loaders.audio));
    }
    apiGet("/audio" + query({ station_id: state.stationId, limit: 100 })).then(function (rows) {
      clear(host);
      if (!rows.length) { setState(host, "empty-state", "No acoustic detections yet. Set an audio source, or connect a Pi with a microphone or hydrophone."); return; }
      var grid = el("div", { class: "card-grid" });
      rows.forEach(function (a) {
        var species = (a.audio_detections || []).map(taxonName).join(", ") || "unclassified sound";
        var acard = el("article", { class: "card" });
        acard.appendChild(el("div", { class: "card-meta", text: (a.event_name || a.observation_id) + " · " + fmtTime(a.first_seen) }));
        acard.appendChild(el("div", { class: "card-title", text: species }));
        acard.appendChild(el("div", { class: "card-stats", text:
          "true duration: " + fmtNum(a.audio_true_duration_seconds, 1) + "s" +
          (a.audio_capped ? " (stored clip capped)" : "") +
          "  model: " + (a.acoustic_model_version || "unstated") }));
        if (a.audio_clip_path) {
          acard.appendChild(el("audio", { class: "audio-clip", controls: true, preload: "none", src: API + "/media" + query({ path: a.audio_clip_path }) }));
        } else {
          acard.appendChild(el("div", { class: "card-note", text: "No stored clip for this event." }));
        }
        grid.appendChild(acard);
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

  // ----------------------------------------------------------------------
  // Sensors: a live per-station overview, and the environmental-channel manager
  // in Settings. Environmental sensors are the station channels; they are
  // physical hardware on the Pi's I2C bus, so they are configured by driver and
  // address rather than a URL, and their readings are captured at detection
  // events and shown here once a station is capturing.
  // ----------------------------------------------------------------------

  // The reference environmental sensors from the hardware guide, each with its
  // interface, address, unit, and typical quality-control ranges, so adding a
  // standard channel is a single choice. Custom leaves every field open.
  var CHANNEL_PRESETS = [
    { group: "Marine", label: "Water temperature", id: "water_temp_c", unit: "degC", marine: true, driver: { interface: "i2c", address: "0x66", type: "atlas_ezo_rtd" }, gross: [-2, 40], sensor: [-5, 50] },
    { group: "Marine", label: "pH", id: "ph", unit: "pH", marine: true, driver: { interface: "i2c", address: "0x63", type: "atlas_ezo_ph" }, gross: [6, 9], sensor: [0, 14] },
    { group: "Marine", label: "Dissolved oxygen", id: "dissolved_oxygen_mg_l", unit: "mg/L", marine: true, driver: { interface: "i2c", address: "0x61", type: "atlas_ezo_do" }, gross: [0, 20], sensor: [0, 50] },
    { group: "Marine", label: "Salinity", id: "salinity_psu", unit: "PSU", marine: true, driver: { interface: "i2c", address: "0x64", type: "atlas_ezo_ec" }, gross: [0, 42], sensor: [0, 80] },
    { group: "Terrestrial", label: "Air temperature", id: "air_temp_c", unit: "degC", marine: false, driver: { interface: "i2c", address: "0x44", type: "sht4x" }, gross: [-20, 55], sensor: [-40, 85] },
    { group: "Terrestrial", label: "Relative humidity", id: "relative_humidity_pct", unit: "%", marine: false, driver: { interface: "i2c", address: "0x44", type: "sht4x" }, gross: [0, 100], sensor: [0, 100] },
    { group: "Terrestrial", label: "Soil moisture", id: "soil_moisture_pct", unit: "%", marine: false, driver: { interface: "i2c", address: "0x36", type: "capacitive_soil" }, gross: [0, 100], sensor: [0, 100] },
    { group: "Terrestrial", label: "Illuminance", id: "illuminance_lux", unit: "lux", marine: false, driver: { interface: "i2c", address: "0x23", type: "bh1750" }, gross: [0, 100000], sensor: [0, 120000] },
    { group: "Other", label: "Custom channel", id: "", unit: "", marine: false, driver: { interface: "i2c", address: "", type: "" }, gross: [null, null], sensor: [null, null] }
  ];

  loaders.sensors = function () {
    var host = region("sensors-overview");
    var filters = region("sensors-filters");
    if (filters && !filters.hasChildNodes()) { filters.appendChild(filterBar(loaders.sensors)); }
    apiGet("/sensors" + query({ station_id: state.stationId })).then(function (data) {
      clear(host);
      var stations = data.stations || [];
      if (!stations.length) { setState(host, "empty-state", "No stations are configured."); return; }
      stations.forEach(function (st) { host.appendChild(sensorOverviewCard(st)); });
      host.appendChild(el("p", { class: "card-note", text: data.note || "" }));
    }).catch(function (e) { setState(host, "empty-state", "Could not load sensors: " + e.message); });
  };

  function sensorOverviewCard(st) {
    var card = el("section", { class: "sensor-station" });
    card.appendChild(el("div", { class: "sensor-station-head" }, [
      el("span", { class: "sensor-station-name", text: st.station_name || st.station_id }),
      badge(st.environment_type || "", "source")
    ]));
    var dev = st.sensors || {};
    card.appendChild(el("div", { class: "sensor-devices" }, [
      sensorChip("Camera", !!(dev.camera && dev.camera.enabled)),
      sensorChip("Audio", !!(dev.audio && dev.audio.enabled)),
      sensorChip("GPS", !!(dev.gps && dev.gps.enabled))
    ]));
    var channels = st.channels || [];
    if (!channels.length) {
      card.appendChild(el("p", { class: "card-note", text: "No environmental channels yet. Add them in Settings, Sensors." }));
      return card;
    }
    var list = el("div", { class: "sensor-channels" });
    channels.forEach(function (ch) { list.appendChild(sensorReadingRow(ch)); });
    card.appendChild(list);
    return card;
  }

  function sensorChip(label, on) {
    return el("span", { class: "sensor-chip" }, [
      el("span", { class: "sensor-chip-label", text: label }),
      statusPill(on, "Enabled", "Disabled")
    ]);
  }

  function sensorReadingRow(ch) {
    var r = ch.latest_reading;
    var hasValue = r && r.value !== null && r.value !== undefined;
    var valueText = hasValue ? (fmtNum(r.value, 2) + " " + (r.unit || ch.unit || "")) : "no readings yet";
    return el("div", { class: "sensor-channel" + (ch.enabled ? "" : " is-off") }, [
      el("div", { class: "sensor-channel-main" }, [
        el("span", { class: "sensor-channel-id", text: ch.id }),
        ch.unit ? el("span", { class: "sensor-channel-unit", text: ch.unit }) : null
      ]),
      el("div", { class: "sensor-channel-reading" }, [
        el("span", { class: "sensor-value" + (hasValue ? "" : " muted"), text: valueText }),
        r ? qcBadge(r) : null,
        (r && r.created_at) ? el("span", { class: "card-meta", text: fmtTime(r.created_at) }) : null
      ]),
      statusPill(!!ch.enabled, "Enabled", "Disabled")
    ]);
  }

  function qcBadge(r) {
    if (r.qartod_flag !== null && r.qartod_flag !== undefined) {
      var map = { 1: ["QARTOD good", "ok"], 2: ["QARTOD not evaluated", "muted"], 3: ["QARTOD suspect", "status"], 4: ["QARTOD fail", "muted"], 9: ["QARTOD missing", "muted"] };
      var m = map[r.qartod_flag] || ["QARTOD " + r.qartod_flag, "muted"];
      return badge(m[0], m[1]);
    }
    if (r.status) { return badge(String(r.status).replace(/_/g, " "), r.status === "measured" ? "ok" : "muted"); }
    return null;
  }

  // Settings, Sensors: manage the device inputs and the environmental channels
  // per station. Device toggles and channel enable write immediately; adding,
  // editing, or removing a channel goes through the guarded channel endpoints.
  function renderSensorsSettings(stations) {
    var host = region("settings-sensors");
    if (!host) { return; }
    clear(host);
    host.appendChild(el("p", { class: "settings-desc", text: "Camera, audio, and GPS inputs, and the environmental channels (pH, temperature, salinity, and so on) each station records. Environmental sensors are wired to the Pi over I2C, so they are set by driver and address, not a URL." }));
    if (!stations.length) { host.appendChild(el("p", { class: "card-note", text: "No stations are configured." })); return; }
    var list = el("div", { class: "sensor-config-list" });
    stations.forEach(function (st) { list.appendChild(sensorConfigCard(st)); });
    host.appendChild(list);
    if (!state.settingsCanEdit) { host.appendChild(el("p", { class: "card-note", text: "Editing sensors is available on the desktop hub." })); }
  }

  function sensorConfigCard(st) {
    var channels = st.channels || [];
    var card = el("details", { class: "config-card sensor-config-card" }, [
      el("summary", { text: (st.station_name || st.station_id) + " (" + channels.length + " channel" + (channels.length === 1 ? "" : "s") + ")" })
    ]);
    var body = el("div", { class: "config-card-body" });
    card.appendChild(body);
    renderSensorConfig(body, st);
    return card;
  }

  function renderSensorConfig(body, st) {
    clear(body);
    var sid = st.station_id;
    var canEdit = state.settingsCanEdit;

    body.appendChild(el("h4", { text: "Device inputs" }));
    var sensors = st.sensors || {};
    [["camera", "sensor_camera_enabled", "Camera"], ["audio", "sensor_audio_enabled", "Audio"], ["gps", "sensor_gps_enabled", "GPS"]].forEach(function (t) {
      var on = !!(sensors[t[0]] && sensors[t[0]].enabled);
      var control = canEdit
        ? liveSwitch(on, function (next, done) {
            saveSettings([{ scope: "station", station_id: sid, field: t[1], value: next }])
              .then(function () { done(true); })
              .catch(function (e) { window.alert("Could not update: " + e.message); done(false); });
          })
        : statusPill(on, "Enabled", "Disabled");
      body.appendChild(el("div", { class: "switch-field" }, [el("span", { class: "form-label", text: t[2] }), control]));
    });

    body.appendChild(el("h4", { text: "Environmental channels" }));
    var channels = st.channels || [];
    if (!channels.length) {
      body.appendChild(el("p", { class: "card-note", text: "No environmental channels yet." }));
    } else {
      channels.forEach(function (ch) {
        var on = !!ch.enabled;
        var control = canEdit
          ? liveSwitch(on, function (next, done) {
              saveSettings([{ scope: "channel", station_id: sid, channel_id: ch.id, field: "enabled", value: next }])
                .then(function () { done(true); })
                .catch(function (e) { window.alert("Could not update: " + e.message); done(false); });
            })
          : statusPill(on, "Enabled", "Disabled");
        var actions = canEdit ? el("div", { class: "channel-row-actions" }, [
          el("button", { type: "button", class: "btn btn-small", text: "Edit", onclick: function () { renderChannelForm(body, st, ch); } }),
          el("button", { type: "button", class: "btn btn-small", text: "Delete", onclick: function () { deleteChannel(sid, ch.id); } })
        ]) : null;
        body.appendChild(el("div", { class: "channel-row" }, [
          el("div", { class: "channel-row-main" }, [
            el("span", { class: "sensor-channel-id", text: ch.id }),
            el("span", { class: "sensor-channel-unit", text: (ch.unit || "no unit") + (ch.marine ? " . marine" : "") })
          ]),
          control,
          actions
        ]));
      });
    }

    if (canEdit) {
      var add = el("button", { type: "button", class: "btn btn-primary btn-small", text: "Add channel" });
      add.addEventListener("click", function () { renderChannelForm(body, st, null); });
      body.appendChild(el("div", { class: "form-actions" }, [add]));
    }
  }

  function deleteChannel(sid, cid) {
    if (!window.confirm('Remove the channel "' + cid + '"? Past readings stay in the record.')) { return; }
    apiSend("/settings/stations/" + encodeURIComponent(sid) + "/channels/" + encodeURIComponent(cid), "DELETE")
      .then(function () { loaders.settings(); })
      .catch(function (e) { window.alert("Could not remove the channel: " + e.message); });
  }

  function renderChannelForm(body, st, existing) {
    clear(body);
    var sid = st.station_id;
    var isEdit = !!existing;
    var msg = el("p", { class: "form-message" });

    body.appendChild(el("h4", { text: isEdit ? "Edit channel" : "Add channel" }));

    var presetSelect = null;
    if (!isEdit) {
      presetSelect = el("select", { class: "form-input" });
      CHANNEL_PRESETS.forEach(function (p, i) {
        presetSelect.appendChild(el("option", { value: String(i), text: p.group + ": " + p.label }));
      });
      body.appendChild(editField("Sensor preset", presetSelect, "Pick a reference sensor to fill the fields, then adjust as needed."));
    }

    var idIn = el("input", { type: "text", class: "form-input", value: existing ? existing.id : "" });
    body.appendChild(isEdit
      ? lockedRow("Channel id", existing.id)
      : editField("Channel id", idIn, "A short identifier, for example water_temp_c."));

    var unitIn = el("input", { type: "text", class: "form-input", value: existing ? (existing.unit || "") : "" });
    body.appendChild(editField("Unit", unitIn, "For example degC, pH, PSU, mg/L."));

    var marineSw = editSwitch(existing ? !!existing.marine : false);
    body.appendChild(el("div", { class: "switch-field" }, [el("span", { class: "form-label", text: "Marine channel (carries a QARTOD quality flag)" }), marineSw.node]));

    var driver = (existing && existing.driver) || {};
    var ifaceIn = el("input", { type: "text", class: "form-input", value: driver.interface || "i2c" });
    body.appendChild(editField("Driver interface", ifaceIn, "How it connects, for example i2c."));
    var addrIn = el("input", { type: "text", class: "form-input", value: driver.address || "" });
    body.appendChild(editField("Driver address", addrIn, "The bus address, for example 0x63. A physical sensor has no URL."));
    var typeIn = el("input", { type: "text", class: "form-input", value: driver.type || "" });
    body.appendChild(editField("Driver type", typeIn, "The sensor model, for example atlas_ezo_ph."));

    var qc = (existing && existing.qc) || {};
    var gr = qc.gross_range || {}, sr = qc.sensor_range || {};
    function num(val) { return (val === null || val === undefined) ? "" : String(val); }
    var grMin = el("input", { type: "number", step: "any", class: "form-input", value: num(gr.min) });
    var grMax = el("input", { type: "number", step: "any", class: "form-input", value: num(gr.max) });
    var srMin = el("input", { type: "number", step: "any", class: "form-input", value: num(sr.min) });
    var srMax = el("input", { type: "number", step: "any", class: "form-input", value: num(sr.max) });
    body.appendChild(editField("QC gross range min", grMin));
    body.appendChild(editField("QC gross range max", grMax));
    body.appendChild(editField("QC sensor range min", srMin));
    body.appendChild(editField("QC sensor range max", srMax));

    if (presetSelect) {
      presetSelect.addEventListener("change", function () {
        var p = CHANNEL_PRESETS[Number(presetSelect.value)];
        if (!p) { return; }
        idIn.value = p.id; unitIn.value = p.unit;
        marineSw.node.classList.toggle("on", !!p.marine);
        marineSw.node.setAttribute("aria-pressed", p.marine ? "true" : "false");
        ifaceIn.value = p.driver.interface; addrIn.value = p.driver.address; typeIn.value = p.driver.type;
        grMin.value = num(p.gross[0]); grMax.value = num(p.gross[1]);
        srMin.value = num(p.sensor[0]); srMax.value = num(p.sensor[1]);
      });
    }

    body.appendChild(msg);
    var save = el("button", { type: "button", class: "btn btn-primary", text: isEdit ? "Save channel" : "Add channel" });
    var cancel = el("button", { type: "button", class: "btn", text: "Cancel", onclick: function () { renderSensorConfig(body, st); } });

    save.addEventListener("click", function () {
      function optNum(input) { var s = input.value.trim(); return s === "" ? null : Number(s); }
      function badNum(input) { var s = input.value.trim(); return s !== "" && isNaN(Number(s)); }
      if (badNum(grMin) || badNum(grMax) || badNum(srMin) || badNum(srMax)) { msg.textContent = "A QC value is not a valid number."; return; }
      if (!unitIn.value.trim()) { msg.textContent = "A unit is required."; return; }

      if (isEdit) {
        var changes = [];
        function push(field, value) { changes.push({ scope: "channel", station_id: sid, channel_id: existing.id, field: field, value: value }); }
        if (unitIn.value.trim() !== (existing.unit || "")) { push("unit", unitIn.value.trim()); }
        if (marineSw.get() !== !!existing.marine) { push("marine", marineSw.get()); }
        if (ifaceIn.value.trim() && ifaceIn.value.trim() !== (driver.interface || "")) { push("driver_interface", ifaceIn.value.trim()); }
        if (addrIn.value.trim() !== (driver.address || "")) { push("driver_address", addrIn.value.trim() || null); }
        if (typeIn.value.trim() && typeIn.value.trim() !== (driver.type || "")) { push("driver_type", typeIn.value.trim()); }
        if (optNum(grMin) !== null && optNum(grMin) !== (gr.min === undefined ? null : gr.min)) { push("qc_gross_min", optNum(grMin)); }
        if (optNum(grMax) !== null && optNum(grMax) !== (gr.max === undefined ? null : gr.max)) { push("qc_gross_max", optNum(grMax)); }
        if (optNum(srMin) !== null && optNum(srMin) !== (sr.min === undefined ? null : sr.min)) { push("qc_sensor_min", optNum(srMin)); }
        if (optNum(srMax) !== null && optNum(srMax) !== (sr.max === undefined ? null : sr.max)) { push("qc_sensor_max", optNum(srMax)); }
        if (!changes.length) { renderSensorConfig(body, st); return; }
        save.disabled = true; msg.textContent = "Saving.";
        saveSettings(changes).then(function () { loaders.settings(); }).catch(function (e) { save.disabled = false; msg.textContent = "Could not save: " + e.message; });
      } else {
        if (!idIn.value.trim()) { msg.textContent = "A channel id is required."; return; }
        var qcOut = {};
        if (optNum(grMin) !== null || optNum(grMax) !== null) {
          qcOut.gross_range = {};
          if (optNum(grMin) !== null) { qcOut.gross_range.min = optNum(grMin); }
          if (optNum(grMax) !== null) { qcOut.gross_range.max = optNum(grMax); }
        }
        if (optNum(srMin) !== null || optNum(srMax) !== null) {
          qcOut.sensor_range = {};
          if (optNum(srMin) !== null) { qcOut.sensor_range.min = optNum(srMin); }
          if (optNum(srMax) !== null) { qcOut.sensor_range.max = optNum(srMax); }
        }
        var payload = {
          id: idIn.value.trim(), unit: unitIn.value.trim(), marine: marineSw.get(), enabled: true,
          driver: { interface: ifaceIn.value.trim(), address: addrIn.value.trim(), type: typeIn.value.trim() },
          qc: qcOut
        };
        save.disabled = true; msg.textContent = "Adding.";
        apiSend("/settings/stations/" + encodeURIComponent(sid) + "/channels", "POST", payload)
          .then(function () { loaders.settings(); })
          .catch(function (e) { save.disabled = false; msg.textContent = "Could not add the channel: " + e.message; });
      }
    });
    body.appendChild(el("div", { class: "form-actions" }, [save, cancel]));
  }

  // An on and off switch that writes its change immediately, reverting if the
  // write is refused, used where a single toggle is its own save.
  function liveSwitch(on, onToggle) {
    var node = el("button", { type: "button", class: "toggle" + (on ? " on" : ""), "aria-pressed": on ? "true" : "false", "aria-label": "Toggle" }, [el("span", { class: "toggle-knob" })]);
    node.addEventListener("click", function () {
      if (node.disabled) { return; }
      var next = !node.classList.contains("on");
      node.disabled = true;
      onToggle(next, function (ok) {
        node.disabled = false;
        if (ok) { node.classList.toggle("on", next); node.setAttribute("aria-pressed", next ? "true" : "false"); }
      });
    });
    return node;
  }

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
      var models = res[0], memory = res[1], detections = (res[2] && res[2].items) || [];
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
    apiGet("/settings").then(function (s) {
      var cfg = s.config || {};
      state.settingsCanEdit = s.node_role === "desktop";
      state.allowedHabitats = s.allowed_habitats || [];
      var stations = cfg.stations || [];

      renderStationsEditor(stations);
      renderSensorsSettings(stations);
      renderCaptureEditor(cfg.media);
      renderModelsEditor(cfg.desktop_models);
      renderSchedulesEditor(cfg.schedules);
      renderCredentialsEditor(s.secrets_status || {});
      renderAnalysisEditor(cfg.analysis);
      renderStorageEditor(cfg.buffer);
      renderTimezone(cfg);

      renderSetupGuide(region("settings-guide"));
    }).catch(function () {
      ["settings-stations", "settings-sensors", "settings-capture", "settings-models", "settings-schedules", "settings-credentials", "settings-analysis", "settings-storage", "settings-time"].forEach(function (r) {
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

  // A read-only status indicator for a boolean the person cannot change on this
  // view. It reads as a labeled state with a small dot, deliberately unlike a
  // switch, so a value that only reflects saved configuration never looks
  // operable. Where a boolean is user-owned and editable, the settings update
  // path renders a real switch instead.
  function statusPill(on, onLabel, offLabel) {
    var label = on ? (onLabel || "On") : (offLabel || "Off");
    return el("span", {
      class: "status-value " + (on ? "is-on" : "is-off"),
      role: "img", "aria-label": label
    }, [
      el("span", { class: "status-value-dot", "aria-hidden": "true" }),
      el("span", { text: label })
    ]);
  }

  function fieldRow(label, value) {
    var valNode;
    if (typeof value === "boolean") { valNode = statusPill(value); }
    else if (value === null || value === "") { valNode = el("span", { class: "field-value muted", text: "not set" }); }
    else { valNode = el("span", { class: "field-value", text: String(value) }); }
    return el("div", { class: "field-row" }, [el("span", { class: "field-label", text: label }), valNode]);
  }

  // A labeled boolean row that names its states, so a sensor reads "Enabled" or
  // "Disabled" rather than a bare on and off.
  function boolRow(label, on, onLabel, offLabel) {
    return el("div", { class: "field-row" }, [
      el("span", { class: "field-label", text: label }),
      statusPill(on, onLabel, offLabel)
    ]);
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

  function renderTimezone(cfg) {
    var host = region("settings-time");
    if (!host) { return; }
    clear(host);
    host.appendChild(el("p", { class: "settings-desc", text: "Times are stored in UTC and shown in the zone set here. Use \"auto\" to follow this computer's own zone." }));
    var saved = (cfg && cfg.localization && cfg.localization.local_timezone) || displayTimezone();
    var input = el("input", { type: "text", id: "appearance-tz", value: saved, "aria-label": "Display timezone" });
    var msg = el("p", { class: "form-message" });
    var save = el("button", { type: "button", class: "btn btn-primary", text: "Save timezone" });
    save.addEventListener("click", function () {
      var val = input.value.trim();
      if (!val) { msg.textContent = "Enter a zone name, or \"auto\"."; return; }
      if (!state.settingsCanEdit) {
        store(STORE.timezone, val === "auto" ? "" : val);
        if (state.activePanel) { loaders[state.activePanel](); }
        msg.textContent = "Applied on this computer.";
        return;
      }
      save.disabled = true; msg.textContent = "Saving.";
      saveSettings([{ scope: "global", field: "local_timezone", value: val }]).then(function () {
        store(STORE.timezone, val === "auto" ? "" : val);
        loaders.settings();
      }).catch(function (e) { save.disabled = false; msg.textContent = "Could not save: " + e.message; });
    });
    host.appendChild(el("div", { class: "tz-row" }, [
      el("label", { class: "filter-field" }, [el("span", { text: "Display timezone" }), input]),
      save
    ]));
    host.appendChild(msg);
    host.appendChild(el("p", { class: "card-note", text: "Stored data stays in UTC; this only changes how times are shown." }));
  }

  // ----------------------------------------------------------------------
  // Editable settings (guarded write path)
  //
  // Only fields the backend allowlist accepts are offered here; a system-owned
  // value such as a station's identifier is shown but locked. An edit gathers
  // only the fields a person actually changed into one batch, so a save is
  // small and a model path that is left alone never re-warns. The whole batch is
  // validated and written on the desktop, or refused with a clear reason.
  // ----------------------------------------------------------------------

  function saveSettings(changes) {
    return apiSend("/settings/update", "POST", { changes: changes });
  }

  // After a successful save, repaint the settings view from the stored state and
  // surface any non-fatal notes the backend returned (for example a model path
  // that names a file not present yet).
  function settingsSaved(res) {
    loaders.settings();
    if (res && res.warnings && res.warnings.length) {
      window.alert("Saved.\n\n" + res.warnings.join("\n"));
    }
  }

  // An operable on and off switch for a user-owned boolean inside an edit form.
  // It carries the same meaning as the read-only status pill but can be changed,
  // and reports its state through the returned getter.
  function editSwitch(on) {
    var node = el("button", {
      type: "button", class: "toggle" + (on ? " on" : ""),
      "aria-pressed": on ? "true" : "false", "aria-label": "Toggle"
    }, [el("span", { class: "toggle-knob" })]);
    node.addEventListener("click", function () {
      var next = node.classList.toggle("on");
      node.setAttribute("aria-pressed", next ? "true" : "false");
    });
    return { node: node, get: function () { return node.classList.contains("on"); } };
  }

  function editField(labelText, control, hint) {
    return el("label", { class: "form-field" }, [
      el("span", { class: "form-label", text: labelText }),
      control,
      hint ? el("span", { class: "form-hint", text: hint }) : null
    ]);
  }

  function switchField(labelText, on) {
    var sw = editSwitch(on);
    return {
      get: sw.get,
      row: el("div", { class: "switch-field" }, [el("span", { class: "form-label", text: labelText }), sw.node])
    };
  }

  // A value that is set by the system and cannot be changed from the interface,
  // shown plainly with a marker so a person is never left wondering why it will
  // not turn into an input.
  function lockedRow(label, value) {
    return el("div", { class: "field-row" }, [
      el("span", { class: "field-label", text: label }),
      el("span", { class: "field-value locked" }, [
        el("span", { text: (value === null || value === undefined || value === "") ? "not set" : String(value) }),
        el("span", { class: "lock-badge", text: "system" })
      ])
    ]);
  }

  // Turn an editor list into a change batch, coercing by kind and sending only
  // the fields whose value actually changed. Returns null after writing a
  // message when a required field is empty or a number is malformed.
  function collectChanges(editors, msg) {
    var changes = [];
    for (var i = 0; i < editors.length; i++) {
      var e = editors[i];
      var raw = e.get();
      var val;
      if (e.kind === "bool") {
        val = !!raw;
        if (val === !!e.original) { continue; }
      } else if (e.kind === "text") {
        val = String(raw).trim();
        if (!val) { msg.textContent = "A required field is empty."; return null; }
        if (val === (e.original == null ? "" : String(e.original))) { continue; }
      } else if (e.kind === "textOrNull") {
        val = String(raw).trim() || null;
        var orig = (e.original === "" || e.original == null) ? null : String(e.original);
        if (val === orig) { continue; }
      } else if (e.kind === "textSkipBlank") {
        val = String(raw).trim();
        if (!val) { continue; }
        if (val === (e.original == null ? "" : String(e.original))) { continue; }
      } else if (e.kind === "numberSkipBlank") {
        var s = String(raw).trim();
        if (s === "") { continue; }
        val = Number(s);
        if (isNaN(val)) { msg.textContent = "A number field has an invalid value."; return null; }
        if (e.original != null && Number(e.original) === val) { continue; }
      } else {
        continue;
      }
      var change = { scope: e.scope, field: e.field, value: val };
      if (e.station_id) { change.station_id = e.station_id; }
      if (e.channel_id) { change.channel_id = e.channel_id; }
      changes.push(change);
    }
    return changes;
  }

  function renderStationsEditor(stations) {
    var host = region("settings-stations");
    if (!host) { return; }
    clear(host);
    if (GROUP_DESC["settings-stations"]) { host.appendChild(el("p", { class: "settings-desc", text: GROUP_DESC["settings-stations"] })); }
    if (!stations.length) { host.appendChild(el("p", { class: "card-note", text: "No stations are configured yet." })); return; }
    var grid = el("div", { class: "config-grid" });
    stations.forEach(function (st) {
      var card = el("details", { class: "config-card" }, [el("summary", { text: st.station_name || st.station_id })]);
      var body = el("div", { class: "config-card-body" });
      card.appendChild(body);
      renderStationRead(body, st, card);
      grid.appendChild(card);
    });
    host.appendChild(grid);
    if (state.settingsCanEdit) {
      var addWrap = el("div", { class: "add-station" });
      var addBtn = el("button", { type: "button", class: "btn btn-primary", text: "Add station" });
      addBtn.addEventListener("click", function () { renderAddStationForm(addWrap); });
      addWrap.appendChild(addBtn);
      host.appendChild(addWrap);
    } else {
      host.appendChild(el("p", { class: "card-note", text: "Editing a station is available on the desktop hub." }));
    }
  }

  function renderAddStationForm(wrap) {
    clear(wrap);
    var msg = el("p", { class: "form-message" });
    var nameIn = el("input", { type: "text", class: "form-input" });
    var envSel = el("select", { class: "form-input" });
    ["marine", "terrestrial", "estuarine", "freshwater", "mixed"].forEach(function (v) { envSel.appendChild(el("option", { value: v, text: humanize(v) })); });
    var habSel = el("select", { class: "form-input" });
    habSel.appendChild(el("option", { value: "", text: "(not specified)" }));
    (state.allowedHabitats || []).forEach(function (h) { habSel.appendChild(el("option", { value: h, text: h })); });
    var create = el("button", { type: "button", class: "btn btn-primary", text: "Create station" });
    create.addEventListener("click", function () {
      var name = nameIn.value.trim();
      if (!name) { msg.textContent = "A station name is required."; return; }
      create.disabled = true; msg.textContent = "Creating.";
      apiSend("/settings/stations", "POST", { station_name: name, environment_type: envSel.value, habitat: habSel.value || null })
        .then(function () { loaders.settings(); })
        .catch(function (e) { create.disabled = false; msg.textContent = "Could not create the station: " + e.message; });
    });
    var cancel = el("button", { type: "button", class: "btn", text: "Cancel", onclick: function () { loaders.settings(); } });
    wrap.appendChild(el("div", { class: "card" }, [
      el("div", { class: "card-title", text: "New station" }),
      editField("Station name", nameIn),
      editField("Environment", envSel),
      editField("Habitat (optional)", habSel, "A controlled list, so records stay consistent. Choose one, or leave it not specified."),
      msg,
      el("div", { class: "form-actions" }, [create, cancel])
    ]));
  }

  function renderStationRead(body, st, card) {
    clear(body);
    body.appendChild(lockedRow("Station id", st.station_id));
    body.appendChild(fieldRow("Environment", st.environment_type));
    if (st.habitat) { body.appendChild(fieldRow("Habitat", st.habitat)); }
    body.appendChild(fieldRow("Station name", st.station_name));
    var src = (st.capture && st.capture.source) || {};
    body.appendChild(fieldRow("Capture source (video)", src.video || null));
    body.appendChild(fieldRow("Capture source (audio)", src.audio || null));
    var sensors = st.sensors || {};
    body.appendChild(boolRow("Camera", !!(sensors.camera && sensors.camera.enabled), "Enabled", "Disabled"));
    body.appendChild(boolRow("Audio", !!(sensors.audio && sensors.audio.enabled), "Enabled", "Disabled"));
    body.appendChild(boolRow("GPS", !!(sensors.gps && sensors.gps.enabled), "Enabled", "Disabled"));
    var channels = st.channels || [];
    body.appendChild(el("p", { class: "card-note", text: channels.length + " environmental channel" + (channels.length === 1 ? "" : "s") + " on this station." }));
    if (state.settingsCanEdit) {
      var edit = el("button", { type: "button", class: "btn btn-small", text: "Edit station" });
      edit.addEventListener("click", function () { renderStationEdit(body, st, card); });
      var connect = el("button", { type: "button", class: "btn btn-small", text: "Connect to Pi" });
      connect.addEventListener("click", function () { renderConnectPi(body, st, card); });
      var remove = el("button", { type: "button", class: "btn btn-small", text: "Remove" });
      remove.addEventListener("click", function () { removeStation(st); });
      body.appendChild(el("div", { class: "card-actions" }, [edit, connect, remove]));
    }
  }

  function removeStation(st) {
    if (!window.confirm('Remove the station "' + (st.station_name || st.station_id) + '"? Its configuration is deleted; any captured data stays in the record.')) { return; }
    apiSend("/settings/stations/" + encodeURIComponent(st.station_id), "DELETE")
      .then(function () { loaders.settings(); })
      .catch(function (e) { window.alert("Could not remove the station: " + e.message); });
  }

  // The guided, key-first flow to connect a station's Raspberry Pi: authorize the
  // desktop's key on the Pi (once, at flash time), then connect and send
  // everything with no password.
  function renderConnectPi(body, st, card) {
    clear(body);
    var sid = st.station_id;
    body.appendChild(el("h4", { text: "Connect this station's Raspberry Pi" }));
    body.appendChild(el("p", { class: "card-note", text: "Audtheia connects with a key, so no password is needed. Step 1 authorizes the desktop on the Pi; step 2 connects and sends the station's code, configuration, and models." }));

    var keyStep = el("div", { class: "connect-step" }, [el("div", { class: "connect-step-title", text: "1. Authorize the desktop key on your Pi" })]);
    var keyArea = el("textarea", { class: "form-input", rows: 3, readOnly: true });
    keyArea.value = "Loading key.";
    var copyBtn = el("button", { type: "button", class: "btn btn-small", text: "Copy key" });
    copyBtn.addEventListener("click", function () {
      keyArea.select();
      try { document.execCommand("copy"); copyBtn.textContent = "Copied"; window.setTimeout(function () { copyBtn.textContent = "Copy key"; }, 1500); } catch (e) { /* selection is left for a manual copy */ }
    });
    keyStep.appendChild(keyArea);
    keyStep.appendChild(el("p", { class: "form-hint", text: "When flashing the Pi with Raspberry Pi Imager, open the advanced options, enable SSH with public-key authentication, and paste this key. You can also append it to ~/.ssh/authorized_keys on a Pi that is already running." }));
    keyStep.appendChild(copyBtn);
    body.appendChild(keyStep);
    apiGet("/stations/" + encodeURIComponent(sid) + "/provision/key")
      .then(function (r) { keyArea.value = r.public_key || ""; })
      .catch(function (e) { keyArea.value = "Could not load the key: " + e.message; });

    var connStep = el("div", { class: "connect-step" }, [el("div", { class: "connect-step-title", text: "2. Connect to the Pi" })]);
    var prov = st.provisioning || {};
    var hostIn = el("input", { type: "text", class: "form-input", value: prov.host || ("audtheia-" + String(st.station_name || "").toLowerCase() + ".local") });
    var userIn = el("input", { type: "text", class: "form-input", value: prov.user || "pi" });
    var portIn = el("input", { type: "number", class: "form-input", value: String(prov.port || 22) });
    connStep.appendChild(editField("Pi address (an IP, or a name ending in .local)", hostIn));
    connStep.appendChild(editField("Login user", userIn));
    connStep.appendChild(editField("SSH port", portIn));
    var msg = el("p", { class: "form-message" });
    var logArea = el("pre", { class: "connect-log" });
    var connectBtn = el("button", { type: "button", class: "btn btn-primary", text: "Connect" });
    connectBtn.addEventListener("click", function () {
      startConnect(sid, hostIn.value.trim(), userIn.value.trim(), Number(portIn.value) || 22, connectBtn, msg, logArea);
    });
    connStep.appendChild(el("div", { class: "form-actions" }, [connectBtn]));
    connStep.appendChild(msg);
    connStep.appendChild(logArea);
    body.appendChild(connStep);

    body.appendChild(el("div", { class: "form-actions" }, [
      el("button", { type: "button", class: "btn", text: "Back", onclick: function () { renderStationRead(body, st, card); } })
    ]));
  }

  function startConnect(sid, host, user, port, btn, msg, logArea) {
    if (!host) { msg.textContent = "Enter the Pi's address."; return; }
    if (!user) { msg.textContent = "Enter the login user."; return; }
    btn.disabled = true; msg.textContent = "Connecting.";
    apiSend("/stations/" + encodeURIComponent(sid) + "/provision", "POST", { host: host, user: user, port: port })
      .then(function () { pollConnect(sid, btn, msg, logArea); })
      .catch(function (e) { btn.disabled = false; msg.textContent = "Could not start: " + e.message; });
  }

  function pollConnect(sid, btn, msg, logArea) {
    apiGet("/stations/" + encodeURIComponent(sid) + "/provision/status").then(function (s) {
      logArea.textContent = s.log || "";
      if (s.state === "running") {
        msg.textContent = "Connecting.";
        window.setTimeout(function () { pollConnect(sid, btn, msg, logArea); }, 1500);
      } else if (s.state === "succeeded") {
        msg.textContent = "Connected. The station is set up on the Pi.";
        btn.disabled = false;
      } else if (s.state === "failed") {
        msg.textContent = "Connection did not complete. See the log below.";
        btn.disabled = false;
      } else {
        msg.textContent = "";
        btn.disabled = false;
      }
    }).catch(function (e) { msg.textContent = "Could not read status: " + e.message; btn.disabled = false; });
  }

  function renderStationEdit(body, st, card) {
    clear(body);
    var sid = st.station_id;
    var editors = [];
    var msg = el("p", { class: "form-message" });

    body.appendChild(lockedRow("Station id", st.station_id));
    body.appendChild(lockedRow("Environment", st.environment_type));

    var nameIn = el("input", { type: "text", class: "form-input", value: st.station_name || "" });
    editors.push({ scope: "station", field: "station_name", station_id: sid, original: st.station_name || "", get: function () { return nameIn.value; }, kind: "text" });
    body.appendChild(editField("Station name", nameIn));

    var src = (st.capture && st.capture.source) || {};
    var vidIn = el("input", { type: "text", class: "form-input", value: src.video || "" });
    editors.push({ scope: "station", field: "capture_source_video", station_id: sid, original: src.video || "", get: function () { return vidIn.value; }, kind: "textOrNull" });
    body.appendChild(editField("Capture source (video)", vidIn, "webcam:0, url:rtsp://..., stream:<page url>, or file:C:/clip.mp4"));

    var audIn = el("input", { type: "text", class: "form-input", value: src.audio || "" });
    editors.push({ scope: "station", field: "capture_source_audio", station_id: sid, original: src.audio || "", get: function () { return audIn.value; }, kind: "textOrNull" });
    body.appendChild(editField("Capture source (audio)", audIn, "Optional. Leave blank for none."));

    var sensors = st.sensors || {};
    [["camera", "sensor_camera_enabled", "Camera"], ["audio", "sensor_audio_enabled", "Audio"], ["gps", "sensor_gps_enabled", "GPS"]].forEach(function (t) {
      var on = !!(sensors[t[0]] && sensors[t[0]].enabled);
      var f = switchField(t[2], on);
      editors.push({ scope: "station", field: t[1], station_id: sid, original: on, get: f.get, kind: "bool" });
      body.appendChild(f.row);
    });

    var models = st.models || {};
    var piPath = (models.visual_pi && models.visual_pi.path) || "";
    var piIn = el("input", { type: "text", class: "form-input", value: piPath });
    editors.push({ scope: "station", field: "visual_pi_path", station_id: sid, original: piPath, get: function () { return piIn.value; }, kind: "text" });
    body.appendChild(editField("Pi detection model (.hef)", piIn, "Path to this station's field detection model."));

    var dtPath = (models.visual_desktop && models.visual_desktop.path) || "";
    var dtIn = el("input", { type: "text", class: "form-input", value: dtPath });
    editors.push({ scope: "station", field: "visual_desktop_path", station_id: sid, original: dtPath, get: function () { return dtIn.value; }, kind: "textSkipBlank" });
    body.appendChild(editField("Desktop screening model (.onnx)", dtIn, "Optional. Used when running capture on the desktop without field hardware."));

    var cap = st.capture || {};
    var adv = el("details", { class: "subgroup edit-advanced" }, [el("summary", { text: "Advanced capture tuning" })]);
    function numEditor(parent, labelText, field, val, hint) {
      var input = el("input", { type: "number", class: "form-input", step: "any", value: (val === null || val === undefined) ? "" : String(val) });
      editors.push({ scope: "station", field: field, station_id: sid, original: val, get: function () { return input.value; }, kind: "numberSkipBlank" });
      parent.appendChild(editField(labelText, input, hint));
    }
    numEditor(adv, "Frames per second", "capture_fps", cap.fps);
    numEditor(adv, "Resolution width", "resolution_width", cap.resolution && cap.resolution.width);
    numEditor(adv, "Resolution height", "resolution_height", cap.resolution && cap.resolution.height);
    var bt = cap.bytetrack || {};
    numEditor(adv, "ByteTrack activation threshold", "bytetrack_track_activation_threshold", bt.track_activation_threshold);
    numEditor(adv, "ByteTrack matching threshold", "bytetrack_minimum_matching_threshold", bt.minimum_matching_threshold);
    numEditor(adv, "ByteTrack close frames", "bytetrack_track_close_frames", bt.track_close_frames);
    numEditor(adv, "ByteTrack frame rate", "bytetrack_frame_rate", bt.frame_rate);
    numEditor(adv, "Max event duration (seconds)", "max_event_duration_seconds", cap.max_event_duration_seconds);
    body.appendChild(adv);

    var channels = st.channels || [];
    if (channels.length) {
      var chWrap = el("details", { class: "subgroup edit-advanced" }, [el("summary", { text: "Environmental channels (" + channels.length + ")" })]);
      channels.forEach(function (ch) {
        var cCard = el("div", { class: "channel-edit" }, [el("div", { class: "channel-edit-title", text: ch.id + " (" + (ch.unit || "no unit") + ")" })]);
        var on = !!ch.enabled;
        var f = switchField("Enabled", on);
        editors.push({ scope: "channel", field: "enabled", station_id: sid, channel_id: ch.id, original: on, get: f.get, kind: "bool" });
        cCard.appendChild(f.row);
        var qc = ch.qc || {};
        var gr = qc.gross_range || {}, sr = qc.sensor_range || {};
        cChannelNum(editors, cCard, "QC gross min", "qc_gross_min", sid, ch.id, gr.min);
        cChannelNum(editors, cCard, "QC gross max", "qc_gross_max", sid, ch.id, gr.max);
        cChannelNum(editors, cCard, "QC sensor min", "qc_sensor_min", sid, ch.id, sr.min);
        cChannelNum(editors, cCard, "QC sensor max", "qc_sensor_max", sid, ch.id, sr.max);
        chWrap.appendChild(cCard);
      });
      body.appendChild(chWrap);
    }

    body.appendChild(msg);
    var save = el("button", { type: "button", class: "btn btn-primary", text: "Save changes" });
    var cancel = el("button", { type: "button", class: "btn", text: "Cancel", onclick: function () { renderStationRead(body, st, card); } });
    save.addEventListener("click", function () {
      var changes = collectChanges(editors, msg);
      if (changes === null) { return; }
      if (!changes.length) { renderStationRead(body, st, card); return; }
      save.disabled = true; msg.textContent = "Saving.";
      saveSettings(changes).then(settingsSaved).catch(function (e) { save.disabled = false; msg.textContent = "Could not save: " + e.message; });
    });
    body.appendChild(el("div", { class: "form-actions" }, [save, cancel]));
  }

  // A channel-scoped number editor, kept separate so the station and channel
  // scopes never get crossed in the change batch.
  function cChannelNum(editors, parent, labelText, field, sid, cid, val) {
    var input = el("input", { type: "number", class: "form-input", step: "any", value: (val === null || val === undefined) ? "" : String(val) });
    editors.push({ scope: "channel", field: field, station_id: sid, channel_id: cid, original: val, get: function () { return input.value; }, kind: "numberSkipBlank" });
    parent.appendChild(editField(labelText, input));
  }

  function renderSchedulesEditor(schedules) {
    schedules = schedules || {};
    var reports = schedules.reports || {}, dream = schedules.dream_pass || {};
    editableSection("settings-schedules",
      "When reports and the longitudinal pass run.",
      [
        { label: "Report schedule", field: "reports_schedule", value: reports.schedule, kind: "select", options: ["hourly", "daily", "weekly", "biweekly", "monthly"] },
        { label: "Report formats", field: "reports_formats", value: reports.formats, kind: "multiselect", options: ["pdf", "csv"], minOne: true },
        { label: "Dream pass schedule", field: "dream_schedule", value: dream.schedule, kind: "select", options: ["hourly", "daily", "weekly", "biweekly", "monthly"] }
      ],
      "Save schedules");
  }

  function renderModelsEditor(desktopModels) {
    desktopModels = desktopModels || {};
    var rf = desktopModels.visual_rfdetr || {};
    editableSection("settings-models",
      "The models this hub uses and where they are stored.",
      [
        { label: "Verification model (RF-DETR ONNX) path", field: "visual_rfdetr_path", value: rf.path, kind: "text", hint: "Must match the model file you placed under models/." },
        { label: "Version", field: "visual_rfdetr_version", value: rf.version, kind: "textOrNull", hint: "Optional. Set it to match your model." },
        { label: "Citation", field: "visual_rfdetr_citation", value: rf.citation, kind: "textOrNull", hint: "Optional. Credit the model's source." }
      ],
      "Save model paths",
      "The language model is chosen in Brain, under Models and Memory.");
  }

  // Shared builders for the global-scope editors below. Each records an editor in
  // the list so one Save writes the batch through the guarded settings path.
  function globalNumberField(editors, host, label, field, value, hint) {
    var input = el("input", { type: "number", step: "any", class: "form-input", value: (value === null || value === undefined) ? "" : String(value) });
    editors.push({ scope: "global", field: field, original: value, get: function () { return input.value; }, kind: "numberSkipBlank" });
    host.appendChild(editField(label, input, hint));
  }

  function globalTextField(editors, host, label, field, value, hint, kind) {
    var input = el("input", { type: "text", class: "form-input", value: (value === null || value === undefined) ? "" : String(value) });
    editors.push({ scope: "global", field: field, original: (value === null || value === undefined) ? "" : value, get: function () { return input.value; }, kind: kind || "text" });
    host.appendChild(editField(label, input, hint));
  }

  function globalSelectField(editors, host, label, field, value, options) {
    var sel = el("select", { class: "form-input" });
    options.forEach(function (o) { var opt = el("option", { value: o, text: humanize(o) }); if (o === value) { opt.selected = true; } sel.appendChild(opt); });
    editors.push({ scope: "global", field: field, original: value, get: function () { return sel.value; }, kind: "text" });
    host.appendChild(editField(label, sel));
  }

  function globalSwitchField(editors, host, label, field, on) {
    var sw = editSwitch(on);
    editors.push({ scope: "global", field: field, original: on, get: sw.get, kind: "bool" });
    host.appendChild(el("div", { class: "switch-field" }, [el("span", { class: "form-label", text: label }), sw.node]));
  }

  // A settings section that shows its values read-only with an Edit affordance,
  // and switches to a form with Save and Cancel, matching the station cards so
  // every configuration surface is edited the same way. Each spec is a field:
  //   { label, hint, field, value, kind, options, onLabel, offLabel, advanced }.
  function readRow(f) {
    if (f.kind === "switch") { return boolRow(f.label, !!f.value, f.onLabel, f.offLabel); }
    if (f.kind === "multiselect") { return fieldRow(f.label, (f.value && f.value.length) ? f.value.join(", ") : "none"); }
    return fieldRow(f.label, (f.value === undefined ? null : f.value));
  }

  function sectionAddControl(editors, multis, parent, f) {
    if (f.kind === "number") { globalNumberField(editors, parent, f.label, f.field, f.value, f.hint); }
    else if (f.kind === "select") { globalSelectField(editors, parent, f.label, f.field, f.value, f.options); }
    else if (f.kind === "switch") { globalSwitchField(editors, parent, f.label, f.field, !!f.value); }
    else if (f.kind === "textOrNull") { globalTextField(editors, parent, f.label, f.field, f.value, f.hint, "textOrNull"); }
    else if (f.kind === "multiselect") {
      var boxes = {};
      var wrap = el("div", { class: "format-choices" });
      f.options.forEach(function (o) {
        var cb = el("input", { type: "checkbox" }); cb.checked = (f.value || []).indexOf(o) !== -1;
        boxes[o] = cb;
        wrap.appendChild(el("label", { class: "format-choice" }, [cb, el("span", { text: String(o).toUpperCase() })]));
      });
      parent.appendChild(editField(f.label, wrap));
      multis.push({ field: f.field, boxes: boxes, options: f.options, original: (f.value || []).slice(), minOne: f.minOne });
    }
    else { globalTextField(editors, parent, f.label, f.field, f.value, f.hint, "text"); }
  }

  function sectionSplit(specs, container, render) {
    var advanced = [];
    specs.forEach(function (f) { if (f.advanced) { advanced.push(f); } else { render(container, f); } });
    if (advanced.length) {
      var adv = el("details", { class: "subgroup edit-advanced" }, [el("summary", { text: "Advanced" })]);
      advanced.forEach(function (f) { render(adv, f); });
      container.appendChild(adv);
    }
  }

  function editableSection(target, desc, specs, saveLabel, footer) {
    var host = (typeof target === "string") ? region(target) : target;
    if (!host) { return; }
    clear(host);
    host.appendChild(el("p", { class: "settings-desc", text: desc }));
    var body = el("div");
    host.appendChild(body);
    if (footer) { host.appendChild(el("p", { class: "card-note", text: footer })); }
    var canEdit = state.settingsCanEdit;

    function showRead() {
      clear(body);
      sectionSplit(specs, body, function (parent, f) { parent.appendChild(readRow(f)); });
      if (canEdit) {
        var edit = el("button", { type: "button", class: "btn btn-small", text: "Edit" });
        edit.addEventListener("click", showEdit);
        body.appendChild(el("div", { class: "card-actions" }, [edit]));
      } else {
        body.appendChild(el("p", { class: "card-note", text: "Editing is available on the desktop hub." }));
      }
    }

    function showEdit() {
      clear(body);
      var editors = [], multis = [];
      var msg = el("p", { class: "form-message" });
      sectionSplit(specs, body, function (parent, f) { sectionAddControl(editors, multis, parent, f); });
      body.appendChild(msg);
      var save = el("button", { type: "button", class: "btn btn-primary", text: saveLabel });
      save.addEventListener("click", function () {
        var changes = collectChanges(editors, msg);
        if (changes === null) { return; }
        for (var i = 0; i < multis.length; i++) {
          var m = multis[i];
          var chosen = m.options.filter(function (o) { return m.boxes[o].checked; });
          if (m.minOne && !chosen.length) { msg.textContent = "Choose at least one option."; return; }
          if (chosen.slice().sort().join(",") !== m.original.slice().sort().join(",")) {
            changes.push({ scope: "global", field: m.field, value: chosen });
          }
        }
        if (!changes.length) { msg.textContent = "No changes to save."; return; }
        save.disabled = true; msg.textContent = "Saving.";
        saveSettings(changes).then(settingsSaved).catch(function (e) { save.disabled = false; msg.textContent = "Could not save: " + e.message; });
      });
      var cancel = el("button", { type: "button", class: "btn", text: "Cancel", onclick: showRead });
      body.appendChild(el("div", { class: "form-actions" }, [save, cancel]));
    }

    showRead();
  }

  function renderCaptureEditor(media) {
    media = media || {};
    var img = media.image || {}, aud = media.audio || {};
    editableSection("settings-capture",
      "How captured frames and clips are stored. Per-station capture tuning (frame rate, resolution, ByteTrack) is set on each station under Stations.",
      [
        { label: "Image format", field: "media_image_format", value: img.format, kind: "text", hint: "For example jpg or png." },
        { label: "Image quality (1 to 100)", field: "media_image_quality", value: img.quality, kind: "number" },
        { label: "Audio format", field: "media_audio_format", value: aud.format, kind: "text", hint: "For example wav." },
        { label: "Audio sample width (bytes)", field: "media_audio_sample_width_bytes", value: aud.sample_width_bytes, kind: "number" }
      ],
      "Save capture settings");
  }

  function renderAnalysisEditor(analysis) {
    analysis = analysis || {};
    var baseline = analysis.baseline || {}, sal = analysis.salience || {}, w = sal.weights || {}, anom = sal.anomaly || {};
    var th = analysis.thresholds || {}, fq = th.field_qc || {}, ver = th.verification || {}, dr = th.dream || {};
    editableSection("settings-analysis",
      "Where per-observation analysis runs, and the scientific tuning behind salience and the dream pass.",
      [
        { label: "Per-observation analysis location", field: "analysis_location", value: analysis.per_observation_analysis_location, kind: "select", options: ["pi", "desktop"] },
        { label: "Baseline period granularity", field: "baseline_period_granularity", value: baseline.period_granularity, kind: "select", options: ["month", "iso_week", "doy"], advanced: true },
        { label: "Salience weight: confidence", field: "salience_weight_confidence", value: w.confidence, kind: "number", advanced: true },
        { label: "Salience weight: anomaly", field: "salience_weight_anomaly", value: w.anomaly, kind: "number", advanced: true },
        { label: "Salience weight: rarity", field: "salience_weight_rarity", value: w.rarity, kind: "number", advanced: true },
        { label: "Anomaly min effective n", field: "salience_min_effective_n", value: anom.min_effective_n, kind: "number", advanced: true },
        { label: "Field QC pass confidence (0 to 1)", field: "field_qc_pass_confidence", value: fq.pass_confidence, kind: "number", advanced: true },
        { label: "Verification clear confidence (0 to 1)", field: "verification_clear_confidence", value: ver.clear_confidence, kind: "number", advanced: true },
        { label: "Verification max frames scored", field: "verification_max_frames_scored", value: ver.max_frames_scored, kind: "number", advanced: true },
        { label: "Dream min periods for trend", field: "dream_min_periods_for_trend", value: dr.min_periods_for_trend, kind: "number", advanced: true },
        { label: "Dream min events for correlation", field: "dream_min_events_for_correlation", value: dr.min_events_for_correlation, kind: "number", advanced: true },
        { label: "Dream min events for co-occurrence", field: "dream_min_events_for_co_occurrence", value: dr.min_events_for_co_occurrence, kind: "number", advanced: true },
        { label: "Dream min absolute effect", field: "dream_min_abs_effect", value: dr.min_abs_effect, kind: "number", advanced: true },
        { label: "Dream max p-value (0 to 1)", field: "dream_max_p_value", value: dr.max_p_value, kind: "number", advanced: true }
      ],
      "Save analysis settings");
  }

  function renderStorageEditor(buffer) {
    var host = region("settings-storage");
    if (!host) { return; }
    clear(host);
    var statusHost = el("div", { class: "storage-panel" });
    host.appendChild(statusHost);
    statusHost.appendChild(el("p", { class: "empty-state", text: "Loading storage status." }));

    var bufferHost = el("div");
    host.appendChild(bufferHost);
    buffer = buffer || {};
    editableSection(bufferHost,
      "Buffer limits that govern how the field buffer is kept before a sync.",
      [
        { label: "High-water mark (%)", field: "buffer_high_water_pct", value: buffer.high_water_pct, kind: "number", hint: "Prompts a sync at this fill level." },
        { label: "Hard ceiling (%)", field: "buffer_hard_ceiling_pct", value: buffer.hard_ceiling_pct, kind: "number", hint: "Must be greater than the high-water mark." },
        { label: "Auto-sync when reachable", field: "buffer_auto_sync_when_reachable", value: !!buffer.auto_sync_when_reachable, kind: "switch" },
        { label: "Pause capture at ceiling", field: "buffer_pause_capture_at_ceiling", value: !!buffer.pause_capture_at_ceiling, kind: "switch" }
      ],
      "Save storage settings");

    apiGet("/storage").then(function (s) { renderStorageStatus(statusHost, s); })
      .catch(function (e) { setState(statusHost, "card-note", "Could not load storage status: " + e.message); });
  }

  function storageBar(used, total) {
    var pct = Math.max(0, Math.min(100, (Number(used) / Number(total)) * 100));
    var fill = el("div", { class: "storage-bar-fill" });
    fill.style.width = pct.toFixed(1) + "%";
    return el("div", { class: "storage-bar", "aria-hidden": "true" }, [fill]);
  }

  function renderStorageStatus(host, s) {
    clear(host);
    var disk = s.disk || {};
    if (disk.total) {
      host.appendChild(el("p", { class: "settings-desc", text: "Drive that holds your Audtheia data" }));
      host.appendChild(storageBar(disk.used, disk.total));
      host.appendChild(el("p", { class: "card-note", text: fmtBytes(disk.used) + " used of " + fmtBytes(disk.total) + " (" + fmtBytes(disk.free) + " free). This is the whole drive; Audtheia's own footprint is the Database and Captured data below, and stays small until detections are stored." }));
    } else {
      host.appendChild(el("p", { class: "card-note", text: "Disk capacity is unavailable on this system." }));
    }
    host.appendChild(metricRow([
      metricCard("Database", fmtBytes((s.database || {}).size)),
      metricCard("Captured data", fmtBytes((s.data || {}).size)),
      metricCard("Reports", fmtBytes((s.reports || {}).size)),
      metricCard("Awaiting sync", fmtNum(s.total_unsynced))
    ]));
    if (s.total_unsynced) {
      var rows = Object.keys(s.unsynced || {}).filter(function (k) { return s.unsynced[k]; }).map(function (k) {
        return el("div", { class: "field-row" }, [el("span", { class: "field-label", text: humanize(k) }), el("span", { class: "field-value", text: fmtNum(s.unsynced[k]) })]);
      });
      if (rows.length) { host.appendChild(el("div", { class: "subgroup" }, [el("h4", { text: "Records awaiting sync" })].concat(rows))); }
    }
    host.appendChild(el("p", { class: "card-note", text: s.note || "" }));
  }

  function credStatusRow(label, on) {
    return el("div", { class: "field-row" }, [el("span", { class: "field-label", text: label }), statusPill(!!on, "Configured", "Not set")]);
  }

  function renderCredentialsEditor(status) {
    var host = region("settings-credentials");
    if (!host) { return; }
    clear(host);
    host.appendChild(el("p", { class: "settings-desc", text: "Credentials for the one-time species-data fetch. GBIF taxonomy and occurrence are anonymous; only IUCN Red List status needs a free API token. They are kept in a local file that is never committed." }));
    status = status || {};
    var body = el("div");
    host.appendChild(body);
    credRead(body, status);
  }

  function credRead(body, status) {
    clear(body);
    body.appendChild(credStatusRow("IUCN API key", status.iucn_api_key));
    body.appendChild(credStatusRow("GBIF username", status.gbif_username));
    body.appendChild(credStatusRow("GBIF password", status.gbif_password));
    if (state.settingsCanEdit) {
      var edit = el("button", { type: "button", class: "btn btn-small", text: "Edit credentials" });
      edit.addEventListener("click", function () { credEdit(body, status); });
      body.appendChild(el("div", { class: "card-actions" }, [edit]));
    } else {
      body.appendChild(el("p", { class: "card-note", text: "Entering credentials is available on the desktop hub." }));
    }
  }

  function credEdit(body, status) {
    clear(body);
    var msg = el("p", { class: "form-message" });
    var fields = [
      ["iucn_api_key", "IUCN API key", "Get a free token at api.iucnredlist.org."],
      ["gbif_username", "GBIF username", "Optional. Only for the GBIF bulk download API, which Audtheia does not use."],
      ["gbif_password", "GBIF password", "Optional."]
    ];
    var inputs = {}, clears = {};
    fields.forEach(function (f) {
      var input = el("input", { type: "password", class: "form-input", value: "", autocomplete: "off" });
      input.placeholder = status[f[0]] ? "Configured. Type to replace." : "Not set.";
      inputs[f[0]] = input;
      var clearBox = el("input", { type: "checkbox" });
      clears[f[0]] = clearBox;
      body.appendChild(el("div", { class: "cred-field" }, [
        editField(f[1], input, f[2]),
        status[f[0]] ? el("label", { class: "cred-clear" }, [clearBox, el("span", { text: "Clear" })]) : null,
        el("span", { class: "cred-status" }, [statusPill(!!status[f[0]], "Configured", "Not set")])
      ]));
    });
    body.appendChild(msg);
    var save = el("button", { type: "button", class: "btn btn-primary", text: "Save credentials" });
    save.addEventListener("click", function () {
      var values = {};
      fields.forEach(function (f) {
        var k = f[0];
        if (clears[k] && clears[k].checked) { values[k] = ""; }
        else if (inputs[k].value) { values[k] = inputs[k].value; }
      });
      if (!Object.keys(values).length) { msg.textContent = "Enter a credential, or check Clear, to change something."; return; }
      save.disabled = true; msg.textContent = "Saving.";
      apiSend("/settings/secrets", "POST", { values: values }).then(function () { loaders.settings(); })
        .catch(function (e) { save.disabled = false; msg.textContent = "Could not save: " + e.message; });
    });
    var cancel = el("button", { type: "button", class: "btn", text: "Cancel", onclick: function () { credRead(body, status); } });
    body.appendChild(el("div", { class: "form-actions" }, [save, cancel]));
  }

  // A short, built-in walkthrough. It is plain guidance that points at the tabs
  // and panels where each step happens, so a first-time user has a path from a
  // fresh install to a running station without leaving the interface.
  function guideSection(host, title, steps) {
    host.appendChild(el("h4", { class: "guide-title", text: title }));
    var ol = el("ol", { class: "guide-steps" });
    steps.forEach(function (s) { ol.appendChild(el("li", { text: s })); });
    host.appendChild(ol);
  }

  function renderSetupGuide(host) {
    if (!host) { return; }
    clear(host);
    host.appendChild(el("p", { class: "settings-desc", text: "A short walkthrough to get Audtheia running, on the desktop alone or with a Raspberry Pi field station." }));

    guideSection(host, "Set up the visual detection model (desktop)", [
      "Get your trained detector's weights. From Roboflow, use Download Weights (not the dataset); this is the checkpoint file, for example weights.pt.",
      "Export it to ONNX. Install the exporter with: pip install \"rfdetr[onnxexport]\". Then export, for example: python -c \"from rfdetr import RFDETRMedium; RFDETRMedium(pretrain_weights=r'C:/path/weights.pt').export(output_dir=r'models/visual')\". This writes models/visual/inference_model.onnx; rename it if you like.",
      "Point the app at it in two places, because the desktop uses two models. Edit the station and set its Desktop screening model to that .onnx (the detector that runs during capture), and under Settings, Model paths set the Verification model to the same file (the re-score). One file serves both.",
      "The .onnx file must actually exist on disk. Setting a path or a citation does not create it; if a path points at no file, capture cannot start."
    ]);
    guideSection(host, "Run desktop capture, no hardware", [
      "Open Detections, use Set capture source, pick a station, and enter a source such as webcam:0, stream:<web page url>, url:<direct stream>, or file:C:/clip.mp4.",
      "Open Detections, Capture, and press Start for that station. Detections appear below, each with its captured frame; a station with no screening model in place cannot start."
    ]);
    guideSection(host, "Connect a Raspberry Pi field station", [
      "Settings, Stations, Add station: give it a name and environment. A station identifier is generated for you.",
      "On the station card, choose Connect to Pi and copy the shown desktop key.",
      "Flash a Raspberry Pi 5 with the AI HAT+ 2 using Raspberry Pi Imager. In the advanced options, enable SSH with public-key authentication and paste the key.",
      "Boot the Pi on the same network, return to Connect to Pi, enter its address (an IP or a name ending in .local) and user, and connect. It then runs on its own and broadcasts its field hotspot."
    ]);
    guideSection(host, "Add environmental sensors", [
      "Settings, Sensors: expand a station and choose Add channel.",
      "Pick a reference sensor (pH, temperature, salinity, and so on) to fill the driver and quality-control ranges, or choose Custom, then save. Readings appear on the Sensors panel once the station is capturing."
    ]);
    guideSection(host, "Species data and reports", [
      "Settings, Species data credentials: add your free IUCN token to enrich records with Red List conservation status. GBIF taxonomy is anonymous and needs no login.",
      "Settings, Schedules: choose how often reports and the longitudinal pass run. Generate a report any time from the Reports panel."
    ]);

    host.appendChild(el("p", { class: "card-note", text: "Where your data lives: this desktop is the authoritative, long-term store. A Pi keeps only a rolling buffer and syncs to the desktop when they are on the same network." }));
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
