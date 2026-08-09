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
    station: "audtheia.station",
    stationDetections: "audtheia.station.detections",
    stationAudio: "audtheia.station.audio",
    gpsView: "audtheia.gpsView"
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

  var _chartSeq = 0;

  // A simple horizontal bar chart drawn as inline SVG, themed through the CSS
  // variables so it matches every palette. Values is an array of {label,value}.
  //
  // Taxon labels are long and of unpredictable length: a scientific name joined
  // to a common name ("Haemorhous mexicanus_House Finch") is routinely wider than
  // the gutter. SVG text does not wrap and does not clip on its own, so an
  // untreated label runs underneath its own bar and the chart becomes unreadable
  // exactly where the record is richest.
  //
  // Two mechanisms, deliberately overlapping. The label is truncated to a
  // character budget so the reader sees an ellipsis and knows the name continues,
  // and the whole gutter is additionally clipped so that a name whose glyphs are
  // wider than the estimate still cannot reach the bars. The estimate alone would
  // be a guess about font metrics; the clip makes the guarantee structural. The
  // full, untruncated name is always available through a <title>, which is also
  // what a screen reader announces.
  function barChart(values, opts) {
    opts = opts || {};
    var width = 640, rowH = 30, pad = 8;
    // Widened from 170 so ordinary binomials fit outright, with a gap held back
    // so a truncated label never sits flush against the bar it labels.
    var labelW = 210, labelGap = 14;
    var textMax = labelW - labelGap;
    // 13px in the sans stack, from .chart-label. Averaged across mixed-case
    // Latin, which is what a scientific name is; deliberately generous, since
    // over-truncating costs a few characters and under-truncating is caught by
    // the clip below.
    var avgGlyph = 6.8;
    var charBudget = Math.max(4, Math.floor(textMax / avgGlyph));

    var max = values.reduce(function (m, d) { return Math.max(m, Number(d.value) || 0); }, 0) || 1;
    var height = pad * 2 + values.length * rowH;
    var svg = svgEl("svg", {
      viewBox: "0 0 " + width + " " + height, width: "100%", height: height,
      role: "img", "aria-label": opts.title || "bar chart", class: "chart"
    });

    var clipId = "chart-label-clip-" + (++_chartSeq);
    var defs = svgEl("defs", {});
    var clip = svgEl("clipPath", { id: clipId });
    clip.appendChild(svgEl("rect", { x: 0, y: 0, width: textMax, height: height }));
    defs.appendChild(clip);
    svg.appendChild(defs);

    values.forEach(function (d, i) {
      var y = pad + i * rowH;
      var barMax = width - labelW - 60;
      var w = Math.max(2, Math.round((Number(d.value) || 0) / max * barMax));

      var full = String(d.label == null ? "" : d.label);
      var shown = full.length > charBudget ? full.slice(0, charBudget - 1) + "…" : full;
      var label = svgEl("text", {
        x: 0, y: y + 19, class: "chart-label", "clip-path": "url(#" + clipId + ")"
      });
      label.appendChild(document.createTextNode(shown));
      // Present whether or not the label was shortened, so hovering any bar
      // always confirms the taxon rather than only sometimes.
      var title = svgEl("title", {});
      title.appendChild(document.createTextNode(full));
      label.appendChild(title);
      svg.appendChild(label);

      svg.appendChild(svgEl("rect", { x: labelW, y: y + 6, width: w, height: rowH - 14, rx: 4, class: "chart-bar" }));
      var val = svgEl("text", { x: labelW + w + 8, y: y + 19, class: "chart-value" });
      val.appendChild(document.createTextNode(String(d.value)));
      svg.appendChild(val);
    });
    return svg;
  }

  var _plotSeq = 0;

  // A coordinate plot drawn as inline SVG with no map tiles, so it works fully
  // offline. It renders an equirectangular (Plate Carree) plane: longitude on x,
  // latitude on y, both linear, the same projection a world wall map uses. A
  // bundled, generic land outline sits behind a labelled degree grid for
  // orientation, and the real markers go on top. Everything is either a real
  // value or generic reference geography; nothing per observation is invented,
  // and there is no imagery, depth, or boundary data.
  //   points: array of { lat, lon, label, kind, status }
  //   opts:   { view: {minLon, maxLon, minLat, maxLat}, world: {polygons:[...]} | null }
  function coordinatePlot(points, opts) {
    opts = opts || {};
    var view = opts.view || boundsOf(points);
    var maxW = 660, maxH = 420, pad = 40;
    var lonSpan = (view.maxLon - view.minLon) || 0.001;
    var latSpan = (view.maxLat - view.minLat) || 0.001;
    var innerW = maxW - pad * 2, innerH = maxH - pad * 2;
    // Preserve the geographic aspect ratio so continents are not stretched: one
    // degree of longitude and one of latitude get the same number of pixels.
    var ppd = Math.min(innerW / lonSpan, innerH / latSpan);
    var plotW = lonSpan * ppd, plotH = latSpan * ppd;
    var ox = pad + (innerW - plotW) / 2, oy = pad + (innerH - plotH) / 2;

    function px(lon) { return ox + (lon - view.minLon) * ppd; }
    function py(lat) { return oy + (view.maxLat - lat) * ppd; }

    var svg = svgEl("svg", {
      viewBox: "0 0 " + maxW + " " + maxH, width: "100%", height: maxH,
      role: "img", "aria-label": "detection and station locations on a world grid", class: "chart map-plot"
    });

    var clipId = "mapclip" + (++_plotSeq);
    var defs = svgEl("defs", {});
    var clip = svgEl("clipPath", { id: clipId });
    clip.appendChild(svgEl("rect", { x: ox, y: oy, width: plotW, height: plotH }));
    defs.appendChild(clip);
    svg.appendChild(defs);

    // Ocean background.
    svg.appendChild(svgEl("rect", { x: ox, y: oy, width: plotW, height: plotH, class: "map-ocean" }));

    // Generic land outline, clipped to the plot rectangle.
    if (opts.world && opts.world.polygons) {
      var landG = svgEl("g", { "clip-path": "url(#" + clipId + ")" });
      opts.world.polygons.forEach(function (poly) {
        var d = "";
        for (var i = 0; i < poly.length; i++) {
          d += (i === 0 ? "M " : " L ") + px(poly[i][0]).toFixed(1) + " " + py(poly[i][1]).toFixed(1);
        }
        d += " Z";
        landG.appendChild(svgEl("path", { d: d, class: "map-land" }));
      });
      svg.appendChild(landG);
    }

    // Degree grid with edge labels.
    svg.appendChild(graticule(view, px, py, ox, oy, plotW, plotH, clipId));

    // Frame.
    svg.appendChild(svgEl("rect", { x: ox, y: oy, width: plotW, height: plotH, class: "map-frame", fill: "none" }));

    // Markers, clipped so nothing spills past the frame. Detections first, then
    // stations on top, so a station marker is never lost behind a cluster of dots.
    // Names are shown in an on-demand callout (hover, keyboard focus, or click to
    // pin) instead of a permanently drawn label, so labels never overlap when two
    // sites sit close together.
    var pts = svgEl("g", { "clip-path": "url(#" + clipId + ")" });

    // One shared, styled callout drawn on top and OUTSIDE the clip, so it renders
    // in full even next to an edge. It carries the same name and position the old
    // always-drawn label and the browser's default tooltip showed, in the panel's
    // own styling rather than the browser's box.
    var callout = svgEl("g", { class: "map-callout", "aria-hidden": "true" });
    callout.setAttribute("display", "none");
    var calloutBox = svgEl("rect", { class: "map-callout-box", rx: 4, x: 0, y: 0, width: 10, height: 10 });
    var calloutName = svgEl("text", { class: "map-callout-name", x: 0, y: 0 });
    var calloutMeta = svgEl("text", { class: "map-callout-meta", x: 0, y: 0 });
    callout.appendChild(calloutBox);
    callout.appendChild(calloutName);
    callout.appendChild(calloutMeta);

    var pinned = null;  // the marker element whose callout a click has pinned

    function placeCallout(p, ax, ay) {
      calloutName.textContent = p.label || "(unnamed)";
      calloutMeta.textContent = pointKind(p) + "  " + p.lat.toFixed(5) + ", " + p.lon.toFixed(5);
      callout.removeAttribute("display");  // make visible before measuring text
      var padX = 8, padY = 6, gap = 4, nameH = 13, metaH = 11;
      var boxW = Math.max(calloutName.getComputedTextLength(), calloutMeta.getComputedTextLength()) + padX * 2;
      var boxH = padY * 2 + nameH + gap + metaH;
      // Prefer above-and-right of the marker; flip or drop below to stay in frame.
      var bx = ax + 10, by = ay - boxH - 8;
      if (bx + boxW > ox + plotW) { bx = ax - boxW - 10; }
      if (bx < ox) { bx = ox + 2; }
      if (bx + boxW > ox + plotW) { bx = ox + plotW - boxW - 2; }
      if (by < oy) { by = ay + 12; }
      if (by + boxH > oy + plotH) { by = oy + plotH - boxH - 2; }
      calloutBox.setAttribute("x", bx.toFixed(1));
      calloutBox.setAttribute("y", by.toFixed(1));
      calloutBox.setAttribute("width", boxW.toFixed(1));
      calloutBox.setAttribute("height", boxH.toFixed(1));
      calloutName.setAttribute("x", (bx + padX).toFixed(1));
      calloutName.setAttribute("y", (by + padY + nameH - 2).toFixed(1));
      calloutMeta.setAttribute("x", (bx + padX).toFixed(1));
      calloutMeta.setAttribute("y", (by + padY + nameH + gap + metaH - 2).toFixed(1));
    }

    function hideCallout() { callout.setAttribute("display", "none"); }

    function wireMarker(mk, p, ax, ay) {
      mk.setAttribute("tabindex", "0");
      mk.setAttribute("role", "button");
      mk.setAttribute("aria-label",
        (p.label || "unnamed") + ", " + pointKind(p) + ", " + p.lat.toFixed(5) + ", " + p.lon.toFixed(5));
      mk.style.cursor = "pointer";
      mk.addEventListener("mouseenter", function () { if (!pinned) { placeCallout(p, ax, ay); } });
      mk.addEventListener("mouseleave", function () { if (!pinned) { hideCallout(); } });
      mk.addEventListener("focus", function () { if (!pinned) { placeCallout(p, ax, ay); } });
      mk.addEventListener("blur", function () { if (!pinned) { hideCallout(); } });
      mk.addEventListener("click", function (e) {
        e.stopPropagation();
        if (pinned === mk) { pinned = null; hideCallout(); }
        else { pinned = mk; placeCallout(p, ax, ay); }
      });
    }

    points.forEach(function (p) {
      if (p.kind === "station") { return; }
      var cls = "map-point" + (p.status === "station_configured" ? " is-configured" : "");
      var ax = px(p.lon), ay = py(p.lat);
      var c = svgEl("circle", { cx: ax.toFixed(1), cy: ay.toFixed(1), r: 5, class: cls });
      wireMarker(c, p, ax, ay);
      pts.appendChild(c);
    });
    points.forEach(function (p) {
      if (p.kind !== "station") { return; }
      var x = px(p.lon), y = py(p.lat), s = 7;
      var d = "M " + x + " " + (y - s) + " L " + (x + s) + " " + y + " L " + x + " " + (y + s) + " L " + (x - s) + " " + y + " Z";
      var mk = svgEl("path", { d: d, class: "map-station" });
      wireMarker(mk, p, x, y);
      pts.appendChild(mk);
    });
    svg.appendChild(pts);
    svg.appendChild(callout);
    // A click on empty map space dismisses a pinned callout.
    svg.addEventListener("click", function () { pinned = null; hideCallout(); });

    return el("div", { class: "map-wrap" }, [svg, coordinateLegend(points)]);
  }

  function boundsOf(points) {
    if (!points.length) { return { minLon: -180, maxLon: 180, minLat: -90, maxLat: 90 }; }
    var lons = points.map(function (p) { return p.lon; });
    var lats = points.map(function (p) { return p.lat; });
    return {
      minLon: Math.min.apply(null, lons), maxLon: Math.max.apply(null, lons),
      minLat: Math.min.apply(null, lats), maxLat: Math.max.apply(null, lats)
    };
  }

  // The view window for the plot. "world" is the whole globe; "fit" frames the
  // data with enough padding that even a single tight site still shows the
  // coastline around it rather than one lonely dot in empty space.
  function gpsViewWindow(points, mode) {
    if (mode === "world" || !points.length) { return { minLon: -180, maxLon: 180, minLat: -90, maxLat: 90 }; }
    var b = boundsOf(points);
    var padLon = Math.max((b.maxLon - b.minLon) * 0.6, 1.6);
    var padLat = Math.max((b.maxLat - b.minLat) * 0.6, 1.6);
    return {
      minLon: Math.max(-180, b.minLon - padLon), maxLon: Math.min(180, b.maxLon + padLon),
      minLat: Math.max(-90, b.minLat - padLat), maxLat: Math.min(90, b.maxLat + padLat)
    };
  }

  function niceDegStep(span) {
    var steps = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 45];
    for (var i = 0; i < steps.length; i++) { if (span / steps[i] <= 8) { return steps[i]; } }
    return 45;
  }

  function fmtDeg(v, pos, neg, dec) {
    if (Math.abs(v) < 1e-9) { return "0°"; }
    return Math.abs(v).toFixed(dec) + "°" + (v > 0 ? pos : neg);
  }

  // Meridians and parallels at a readable interval for the current span, drawn
  // clipped to the plot with degree labels down the left edge and along the
  // bottom.
  function graticule(view, px, py, ox, oy, plotW, plotH, clipId) {
    var g = svgEl("g", {});
    var lines = svgEl("g", { "clip-path": "url(#" + clipId + ")", class: "map-graticule" });
    var lonStep = niceDegStep(view.maxLon - view.minLon);
    var latStep = niceDegStep(view.maxLat - view.minLat);
    var lonDec = lonStep < 1 ? 2 : (lonStep < 5 ? 1 : 0);
    var latDec = latStep < 1 ? 2 : (latStep < 5 ? 1 : 0);

    var lon = Math.ceil(view.minLon / lonStep) * lonStep;
    for (; lon <= view.maxLon + 1e-9; lon += lonStep) {
      var x = px(lon);
      lines.appendChild(svgEl("line", { x1: x, y1: oy, x2: x, y2: oy + plotH }));
      var xlbl = svgEl("text", { x: x, y: oy + plotH + 13, class: "map-graticule-label", "text-anchor": "middle" });
      xlbl.appendChild(document.createTextNode(fmtDeg(lon, "E", "W", lonDec)));
      g.appendChild(xlbl);
    }
    var lat = Math.ceil(view.minLat / latStep) * latStep;
    for (; lat <= view.maxLat + 1e-9; lat += latStep) {
      var y = py(lat);
      lines.appendChild(svgEl("line", { x1: ox, y1: y, x2: ox + plotW, y2: y }));
      var ylbl = svgEl("text", { x: ox - 6, y: y + 3, class: "map-graticule-label", "text-anchor": "end" });
      ylbl.appendChild(document.createTextNode(fmtDeg(lat, "N", "S", latDec)));
      g.appendChild(ylbl);
    }
    g.insertBefore(lines, g.firstChild);
    return g;
  }

  // The human-readable kind of a plotted point, shared by the map callout text
  // and each marker's accessible label. "station position" is a station's own
  // configured coordinates; "entered position" is a detection whose location came
  // from those configured coordinates rather than a live fix; "measured fix" is a
  // live satellite reading.
  function pointKind(p) {
    return p.kind === "station"
      ? "Station position"
      : (p.status === "station_configured" ? "Entered position" : "Measured fix");
  }

  // A legend that names only the point kinds actually on the plot, so a reader
  // sees exactly what each marker means and, for a detection, whether its
  // position was a measured satellite fix or a position entered for the station.
  function coordinateLegend(points) {
    var present = { measured: false, configured: false, station: false };
    points.forEach(function (p) {
      if (p.kind === "station") { present.station = true; }
      else if (p.status === "station_configured") { present.configured = true; }
      else { present.measured = true; }
    });
    var items = [];
    if (present.measured) { items.push(legendItem("dot", "map-point", "Measured GPS fix")); }
    if (present.configured) { items.push(legendItem("dot", "map-point is-configured", "Entered position (no live fix)")); }
    if (present.station) { items.push(legendItem("diamond", "map-station", "Station (configured coordinates)")); }
    return el("div", { class: "map-legend" }, items);
  }

  function legendItem(shape, cls, label) {
    var sw = svgEl("svg", { width: 16, height: 16, viewBox: "0 0 16 16", class: "map-legend-swatch", "aria-hidden": "true" });
    if (shape === "diamond") { sw.appendChild(svgEl("path", { d: "M 8 2 L 14 8 L 8 14 L 2 8 Z", class: cls })); }
    else { sw.appendChild(svgEl("circle", { cx: 8, cy: 8, r: 5, class: cls })); }
    return el("span", { class: "map-legend-item" }, [sw, el("span", { text: label })]);
  }

  function filterBar(onChange, opts) {
    opts = opts || {};
    // A panel that owns a list context keeps its own station choice, so
    // Detections and Audio can show different stations at the same time. Every
    // other panel shares the one app-wide choice, exactly as before.
    var scope = (opts.ctx && opts.ctx.stationStoreKey) ? opts.ctx : null;
    function chosenStation() { return scope ? scope.stationId : state.stationId; }
    function chooseStation(value) {
      if (scope) { scope.stationId = value; store(scope.stationStoreKey, value); }
      else { state.stationId = value; store(STORE.station, value); }
    }

    var bar = el("div", { class: "filter-bar" });
    var select = el("select", { class: "filter-station", "aria-label": "Station" });
    select.appendChild(el("option", { value: "", text: "All stations" }));
    (state.stations || []).forEach(function (s) {
      var o = el("option", { value: s.id, text: s.station_name || s.id });
      if (s.id === chosenStation()) { o.selected = true; }
      select.appendChild(o);
    });
    select.addEventListener("change", function () {
      chooseStation(select.value);
      onChange();
    });
    bar.appendChild(el("label", { class: "filter-field" }, [el("span", { text: "Station" }), select]));

    // An optional species dropdown, sitting beside the station one. Its options
    // are filled in by the detections loader from the species actually present,
    // so the same control adapts to whatever a deployment has recorded.
    if (opts.species) {
      var spCtx = opts.ctx;
      var sp = el("select", { class: "filter-species", "aria-label": "Species" });
      sp.appendChild(el("option", { value: "", text: "All species" }));
      sp.addEventListener("change", function () {
        spCtx.speciesFilter = sp.value;
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
    srcIn.placeholder = kind === "audio" ? "file:C:/clip.wav, or a YouTube/stream link (blank for none)" : "webcam:0, a YouTube or stream link, or file:C:/clip.mp4";
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
      ? "The desktop hardware-free audio source: a local file (file:C:/clip.wav) or a YouTube/stream link. A .wav plays with no extra setup; mp3, m4a, and flac need ffmpeg on PATH. A quoted path from Copy as path is fine, and blank means no desktop audio. Physical microphones and hydrophones are wired to a Pi and set per station under Sensors."
      : "This runs desktop capture without a Pi. Live detection also needs a desktop model; set its path under Settings, Model paths, and place the file under models/." }));
    form.appendChild(msg);
    form.appendChild(el("div", { class: "form-actions" }, [save, cancel]));
    wrap.appendChild(form);
  }

  // Start and stop desktop capture from the interface, per station, so detection
  // runs without a terminal. `kind` is "video" (default) or "audio"; the audio
  // variant lists stations with an audio source and drives the acoustic endpoints.
  function captureRunControl(kind, reload) {
    kind = kind || "video";
    var wrap = el("div", { class: "capture-control" });
    var btn = el("button", { type: "button", class: "btn", text: "Capture" });
    btn.addEventListener("click", function () { toggleCapturePanel(wrap, kind, reload); });
    wrap.appendChild(btn);
    return wrap;
  }

  function toggleCapturePanel(wrap, kind, reload) {
    var open = wrap.querySelector(".capture-panel");
    if (open) { wrap.removeChild(open); return; }
    openCapturePanel(wrap, kind, reload);
  }

  function openCapturePanel(wrap, kind, reload) {
    kind = kind || "video";
    var isAudio = kind === "audio";
    var existing = wrap.querySelector(".capture-panel");
    if (existing) { wrap.removeChild(existing); }
    var panel = el("div", { class: "capture-panel card" });
    panel.appendChild(el("div", { class: "card-title", text: isAudio ? "Desktop audio capture" : "Desktop capture" }));
    var body = el("div");
    panel.appendChild(body);
    body.appendChild(el("p", { class: "card-note", text: "Loading." }));
    wrap.appendChild(panel);
    Promise.all([apiGet("/settings"), apiGet("/capture/status")]).then(function (res) {
      var cfg = res[0].config || {};
      var running = (isAudio ? res[1].running_audio : res[1].running) || [];
      clear(body);
      var stations = cfg.stations || [];
      if (!stations.length) {
        body.appendChild(el("p", { class: "card-note", text: "No stations yet. Add one under Settings, Stations." }));
        return;
      }
      // Every station is listed; one that has no source of this kind simply has
      // its Start disabled with a hint, rather than being hidden, so any station
      // can be seen and set up from here.
      stations.forEach(function (st) {
        var src = (st.capture && st.capture.source) || {};
        var sourceVal = isAudio ? src.audio : src.video;
        var isRun = running.indexOf(st.station_id) !== -1;
        var toggle = el("button", { type: "button", class: "btn btn-small" + (isRun ? "" : " btn-primary"), text: isRun ? "Stop" : "Start" });
        if (!sourceVal && !isRun) { toggle.disabled = true; }
        toggle.addEventListener("click", function () {
          toggle.disabled = true;
          var url = "/capture/" + encodeURIComponent(st.station_id) + (isAudio ? "/audio" : "") + (isRun ? "/stop" : "/start");
          apiSend(url, "POST").then(function (r) {
            if (r && r.warning) { window.alert(r.warning); }
            openCapturePanel(wrap, kind, reload);
            reload();
          }).catch(function (e) { toggle.disabled = false; window.alert("Could not " + (isRun ? "stop" : "start") + " capture: " + e.message); });
        });
        var hint = isAudio ? "no audio source, use Set audio source" : "no video source, use Set capture source";
        body.appendChild(el("div", { class: "capture-row" }, [
          el("span", { class: "capture-row-name", text: (st.station_name || st.station_id) + " . " + (sourceVal || hint) }),
          isRun ? badge(isAudio ? "listening" : "capturing", "source") : null,
          toggle
        ]));
      });
      body.appendChild(el("p", { class: "form-hint", text: isAudio
        ? "Start opens the audio source and runs this station's acoustic model; detections appear below as sounds are recognized. The model must be placed under models/acoustic/ and set on the station, with its labels file."
        : "Start opens the source and runs detection; detections appear below as they are found. A desktop model must be set for anything to be detected." }));
    }).catch(function (e) { clear(body); body.appendChild(el("p", { class: "card-note", text: "Could not load capture: " + e.message })); });
  }

  // Refill the species dropdown from the names present in the current load,
  // keeping the active choice selected even when nothing matches it this time,
  // so a chosen species does not silently reset on refresh.
  function populateSpecies(ctx, names) {
    var select = document.querySelector(".filter-species");
    if (!select) { return; }
    var current = ctx.speciesFilter || "";
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
  function deleteControl(ctx) {
    var wrap = el("div", { class: "delete-control" });
    var btn = el("button", { type: "button", class: "btn", text: "Delete" });
    btn.addEventListener("click", function () { toggleDeleteMode(wrap, ctx); });
    wrap.appendChild(btn);
    return wrap;
  }

  function toggleDeleteMode(wrap, ctx) {
    ctx.selecting = !ctx.selecting;
    if (!ctx.selecting) { ctx.selection = {}; }
    var btn = wrap.querySelector(".btn");
    if (btn) {
      btn.textContent = ctx.selecting ? "Cancel" : "Delete";
      btn.classList.toggle("btn-danger", ctx.selecting);
    }
    var panel = wrap.querySelector(".delete-panel");
    if (ctx.selecting && !panel) {
      wrap.appendChild(buildDeletePanel(wrap, ctx));
    } else if (!ctx.selecting && panel) {
      wrap.removeChild(panel);
    }
    ctx.reload();
  }

  function buildDeletePanel(wrap, ctx) {
    var unit = ctx.unit || "detection";
    var panel = el("div", { class: "delete-panel card" });
    panel.appendChild(el("div", { class: "card-title", text: "Delete " + unit + "s" }));

    var selectAll = el("button", { type: "button", class: "btn btn-small select-all", text: "Select all" });
    selectAll.addEventListener("click", function () {
      var ids = ctx.ids || [];
      var allSel = ids.length && ids.every(function (id) { return ctx.selection[id]; });
      ctx.selection = {};
      if (!allSel) { ids.forEach(function (id) { ctx.selection[id] = true; }); }
      ctx.reload();
    });

    var count = el("span", { class: "delete-count", text: "0 selected" });

    var del = el("button", { type: "button", class: "btn btn-small btn-danger delete-selected", text: "Delete selected" });
    del.disabled = true;
    del.addEventListener("click", function () {
      var ids = Object.keys(ctx.selection).filter(function (id) { return ctx.selection[id]; });
      if (!ids.length) { return; }
      if (!window.confirm("Delete " + ids.length + " " + unit + (ids.length > 1 ? "s" : "") +
        " and their stored media? This cannot be undone.")) { return; }
      del.disabled = true;
      apiSend(ctx.deleteEndpoint, "POST", { ids: ids }).then(function () {
        ctx.selection = {};
        ctx.selecting = false;
        var b = wrap.querySelector(".btn");
        if (b) { b.textContent = "Delete"; b.classList.remove("btn-danger"); }
        var p = wrap.querySelector(".delete-panel");
        if (p) { wrap.removeChild(p); }
        ctx.reload();
      }).catch(function (e) { del.disabled = false; window.alert("Could not delete: " + e.message); });
    });

    panel.appendChild(el("div", { class: "delete-panel-row" }, [selectAll, count, del]));
    panel.appendChild(el("p", { class: "form-hint", text: "Tick the " + unit + "s to remove, then Delete selected." }));
    return panel;
  }

  // Keep the delete panel's count, the Delete-selected enabled state, and the
  // Select-all label in step with the current selection after every render.
  function updateDeleteUI(ctx) {
    var selected = Object.keys(ctx.selection).filter(function (id) { return ctx.selection[id]; });
    var count = document.querySelector(".delete-count");
    if (count) { count.textContent = selected.length + " selected"; }
    var del = document.querySelector(".delete-selected");
    if (del) { del.disabled = selected.length === 0; }
    var selectAll = document.querySelector(".select-all");
    if (selectAll) {
      var ids = ctx.ids || [];
      var allSel = ids.length && ids.every(function (id) { return ctx.selection[id]; });
      selectAll.textContent = allSel ? "Unselect all" : "Select all";
    }
  }

  // The top-right selection checkbox shared by the Detections and Audio cards.
  // It stops its click from reaching the card body so ticking a card never also
  // opens its audit view, and keeps the delete panel's count in step.
  function attachSelectCheckbox(ctx, obs, cardEl) {
    cardEl.classList.add("selectable");
    var checked = !!ctx.selection[obs.id];
    if (checked) { cardEl.classList.add("selected"); }
    var cb = el("button", {
      type: "button",
      class: "card-select" + (checked ? " checked" : ""),
      "aria-label": "Select this " + (ctx.unit || "item"),
      "aria-pressed": checked ? "true" : "false"
    });
    cb.addEventListener("click", function (ev) {
      ev.stopPropagation();
      if (ctx.selection[obs.id]) {
        delete ctx.selection[obs.id];
        cardEl.classList.remove("selected"); cb.classList.remove("checked"); cb.setAttribute("aria-pressed", "false");
      } else {
        ctx.selection[obs.id] = true;
        cardEl.classList.add("selected"); cb.classList.add("checked"); cb.setAttribute("aria-pressed", "true");
      }
      updateDeleteUI(ctx);
    });
    cardEl.appendChild(cb);
  }

  // A stable colour per species, derived from its name, with no stored state, so the
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

  // Colour is deterministic first (a species tends to the same hue from its name
  // hash), then de-collided: if a new species would land within a small angle of a
  // hue already in use, it is stepped by the golden angle until it is clear, so no
  // two species on screen ever share a near-identical colour.
  var _speciesColorCache = {};
  var _usedHues = [];
  function _hueClear(hue) {
    for (var k = 0; k < _usedHues.length; k++) {
      var d = Math.abs(hue - _usedHues[k]) % 360;
      if (d > 180) { d = 360 - d; }
      if (d < 28) { return false; }
    }
    return true;
  }
  function speciesColor(name) {
    name = String(name == null ? "unknown" : name);
    if (_speciesColorCache[name]) { return _speciesColorCache[name]; }
    var h = 2166136261;
    for (var i = 0; i < name.length; i++) {
      h ^= name.charCodeAt(i);
      h = (h * 16777619) >>> 0;
    }
    var hue = h % 360;
    var guard = 0;
    while (!_hueClear(hue) && guard < 24) { hue = (hue + 137.508) % 360; guard++; }
    _usedHues.push(hue);
    var sat = 0.60 + ((h >>> 9) % 1000) / 1000 * 0.18;
    var light = 0.50 + ((h >>> 17) % 1000) / 1000 * 0.08;
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

  // ----------------------------------------------------------------------
  // Expert corrections
  //
  // A correction is an additional claim standing beside the model's, never an
  // edit to it. The field call and the desktop verdict are shown unchanged
  // above the controls so the reviewer can see exactly what they are agreeing
  // with or overruling, and so a later reader can tell the machine's assertion
  // from the person's. Nothing here displays a confidence for a human
  // judgement: a percentage is a property of a model's output and means
  // nothing attached to an expert.
  // ----------------------------------------------------------------------

  // The name an expert put on this event, or null when nobody has ruled on it.
  function correctedName(obs) {
    var c = obs && obs.correction;
    if (!c) { return null; }
    if (c.verdict === "reject") { return null; }
    return c.corrected_common_name || c.corrected_scientific_name || null;
  }

  // The badges a reviewed event carries, appended to the provenance row so the
  // expert's position is visible without opening the card.
  function correctionBadges(obs) {
    var c = obs && obs.correction;
    if (!c) { return []; }
    if (c.verdict === "reject") { return [badge("rejected · not an organism", "warn")]; }
    if (c.verdict === "relabel") { return [badge("expert identification", "ok")]; }
    return [badge("expert confirmed", "ok")];
  }

  // The species line for a card: the corrected name when there is one, so the
  // reviewed identification is what the reader sees first.
  function displaySpecies(obs, modelSpecies) {
    var name = correctedName(obs);
    if (name) { return name; }
    if (obs && obs.correction && obs.correction.verdict === "reject") {
      return "no organism present";
    }
    return modelSpecies;
  }

  // The confidence cell for a card. A corrected or confirmed event reports an
  // expert identification in place of the model's percentage, because the
  // number no longer describes the claim being made.
  function displayConfidence(obs, modelConfidence) {
    if (obs && obs.correction) { return "expert identification"; }
    return "confidence " + fmtConfidence(modelConfidence);
  }

  // A one-line restatement of a stored correction, used to show the reviewer
  // what they or a colleague said last time.
  function correctionSummary(c) {
    if (!c) { return "not yet reviewed"; }
    var who = c.corrector || "expert";
    var when = c.corrected_at ? fmtTime(c.corrected_at) : "";
    var what;
    if (c.verdict === "reject") { what = "not an organism"; }
    else if (c.verdict === "relabel") {
      what = "corrected to " + (c.corrected_scientific_name || c.corrected_gbif_usage_key);
    } else { what = "confirmed as identified"; }
    return what + " · " + who + (when ? " · " + when : "");
  }

  // The correction control block, shared by the Detections and Audio audits.
  // `modality` tags the row so a call heard and an organism seen stay
  // distinguishable in the review record. `onSaved` lets the calling view
  // refresh itself without a full page reload.
  function correctionPanel(obs, modality, onSaved) {
    var wrap = el("div", { class: "correction-panel" });
    wrap.appendChild(el("div", { class: "card-title", text: "Was this right?" }));

    // What is being judged, laid out before the controls.
    var context = el("div", { class: "correction-context" });
    function ctxRow(k, v) {
      return el("div", { class: "audit-row" }, [
        el("span", { class: "audit-k", text: k }),
        el("span", { class: "audit-v", text: v })
      ]);
    }
    var dets = (modality === "audio") ? (obs.audio_detections || []) : (obs.vision_detections || []);
    var fieldCall = dets.map(taxonName).join(", ") || "no resolved taxon";
    context.appendChild(ctxRow("Field call", fieldCall));
    var v = obs.verification;
    if (v && (v.rfdetr_scientific_name || v.verified)) {
      context.appendChild(ctxRow("Desktop verification",
        v.rfdetr_scientific_name || (v.verified ? "verified, no species recorded" : "not verified")));
    } else {
      context.appendChild(ctxRow("Desktop verification", "not verified"));
    }
    context.appendChild(ctxRow("Expert review", correctionSummary(obs.correction)));
    wrap.appendChild(context);

    var chosen = null;
    var chosenKey = null;
    var status = el("p", { class: "form-hint", text: "" });
    var searchBox = el("div", { class: "correction-search hidden" });

    var buttons = el("div", { class: "correction-actions" });
    var defs = [
      { verdict: "confirm", label: "Correct" },
      { verdict: "relabel", label: "Not this species" },
      { verdict: "reject", label: "Not an organism" }
    ];
    var btns = {};
    defs.forEach(function (d) {
      var b = el("button", { type: "button", class: "btn correction-btn", text: d.label });
      b.addEventListener("click", function () {
        chosen = d.verdict;
        Object.keys(btns).forEach(function (k) { btns[k].classList.remove("selected"); });
        b.classList.add("selected");
        if (d.verdict === "relabel") {
          searchBox.classList.remove("hidden");
          status.textContent = "Search for the correct species, then choose it from the list.";
        } else {
          searchBox.classList.add("hidden");
          chosenKey = null;
          status.textContent = "";
        }
        saveBtn.disabled = !canSave();
      });
      btns[d.verdict] = b;
      buttons.appendChild(b);
    });

    // The species picker. Only a key the backbone returned is ever submitted,
    // so typing a name and walking away cannot put free text into the record.
    var input = el("input", { type: "text", class: "correction-input",
      placeholder: "Type at least two letters, for example junco" });
    var results = el("div", { class: "correction-results" });
    searchBox.appendChild(input);
    searchBox.appendChild(results);

    var searchTimer = null;
    input.addEventListener("input", function () {
      chosenKey = null;
      saveBtn.disabled = !canSave();
      if (searchTimer) { clearTimeout(searchTimer); }
      var term = input.value.trim();
      if (term.length < 2) { clear(results); return; }
      searchTimer = setTimeout(function () {
        apiGet("/species/search" + query({ q: term })).then(function (data) {
          clear(results);
          if (data.index_available === false) {
            results.appendChild(el("p", { class: "form-hint", text:
              "The taxonomic index has not been built yet, so a species cannot be chosen. Confirm and reject still work." }));
            return;
          }
          var list = data.results || [];
          if (!list.length) {
            results.appendChild(el("p", { class: "form-hint", text: "No matching taxon in the shipped backbone." }));
            return;
          }
          list.forEach(function (r) {
            var line = r.canonical_name;
            if (r.status === "SYNONYM" && r.accepted_name) {
              line += "  (synonym of " + r.accepted_name + ")";
            }
            var opt = el("button", { type: "button", class: "correction-result", text: line });
            opt.addEventListener("click", function () {
              chosenKey = r.usage_key;
              input.value = r.canonical_name;
              clear(results);
              status.textContent = "Will record " + (r.accepted_name || r.canonical_name) + ".";
              saveBtn.disabled = !canSave();
            });
            results.appendChild(opt);
          });
        }).catch(function (e) {
          clear(results);
          results.appendChild(el("p", { class: "form-hint", text: "Species search failed: " + e.message }));
        });
      }, 150);
    });

    function canSave() {
      if (!chosen) { return false; }
      if (chosen === "relabel") { return !!chosenKey; }
      return true;
    }

    var saveBtn = el("button", { type: "button", class: "btn btn-primary", text: "Save review" });
    saveBtn.disabled = true;
    saveBtn.addEventListener("click", function () {
      if (!canSave()) { return; }
      saveBtn.disabled = true;
      status.textContent = "Saving.";
      apiSend("/observations/" + encodeURIComponent(obs.id) + "/correct", "POST", {
        verdict: chosen,
        modality: modality,
        gbif_usage_key: (chosen === "relabel") ? chosenKey : null
      }).then(function (data) {
        // Update the record in hand so the modal and the card behind it both
        // reflect the new state without a reload.
        obs.correction = data.correction;
        status.textContent = "Saved · " + correctionSummary(data.correction);
        if (onSaved) { onSaved(data.correction); }
      }).catch(function (e) {
        status.textContent = "Could not save: " + e.message;
        saveBtn.disabled = false;
      });
    });

    wrap.appendChild(buttons);
    wrap.appendChild(searchBox);
    var footer = el("div", { class: "correction-footer" });
    footer.appendChild(saveBtn);
    wrap.appendChild(footer);
    wrap.appendChild(status);
    return wrap;
  }

  // The frame-audit modal: a zoom-style view opened from a detection card. It
  // shows every stored frame of the event with its own box and confidence, and a
  // panel deriving the card's numbers (frame count, duration, confidence,
  // salience) from those frames, so a scientist can verify the stats rather than
  // trust them.
  function openFrameAudit(obs, onCorrected) {
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
    var title = el("div", { class: "audit-title", text: displaySpecies(obs, species) });
    panel.appendChild(title);
    panel.appendChild(el("div", { class: "card-meta", text: obs.event_name || obs.id }));

    var body = el("div", { class: "audit-body" });
    // The review controls sit above the frames, because the question the modal
    // exists to answer is whether the call was right, not what the numbers were.
    body.appendChild(correctionPanel(obs, "vision", function () {
      title.textContent = displaySpecies(obs, species);
      if (onCorrected) { onCorrected(obs); }
    }));
    // The frames load into their own host so that refreshing them never removes
    // the review controls the reviewer may already be part-way through using.
    var framesHost = el("div", { class: "audit-frames" });
    framesHost.appendChild(el("p", { class: "card-note", text: "Loading frames." }));
    body.appendChild(framesHost);
    panel.appendChild(body);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    apiGet("/detections/" + encodeURIComponent(obs.id) + "/frames").then(function (data) {
      clear(framesHost);
      var frames = data.frames || [];
      framesHost.appendChild(auditDerivation(obs, frames));
      // Any field-skill flags that fired on this event, shown as derived
      // readings that stand beside the measurement, never as a correction to it.
      var skillFlags = data.skill_flags || [];
      if (skillFlags.length) {
        var flagWrap = el("div", { class: "audit-derivation" });
        flagWrap.appendChild(el("div", { class: "card-title", text: "Field-skill flags" }));
        flagWrap.appendChild(el("p", { class: "form-hint", text:
          "Each flag is a derived reading of this event's measured values, recorded by a field skill. It stands beside the measurement and never alters it." }));
        var flagRow = el("div", { class: "badge-row" });
        skillFlags.forEach(function (f) { flagRow.appendChild(badge(f.skill_title, "source")); });
        flagWrap.appendChild(flagRow);
        framesHost.appendChild(flagWrap);
      }
      if (!frames.length) {
        framesHost.appendChild(el("p", { class: "empty-state", text: "No stored frames were found for this event on disk." }));
      } else {
        // A curation summary that repaints as frames are reviewed, above the
        // strip, so the effect of a verdict is visible without scrolling.
        var summaryHost = el("div", { class: "audit-curation" });
        function paintCuration(sum) {
          clear(summaryHost);
          summaryHost.appendChild(curationView(data.distribution || [], sum || data.review_summary));
        }
        paintCuration(data.review_summary);
        framesHost.appendChild(summaryHost);
        framesHost.appendChild(auditFrameStrip(frames, species, obs.id, paintCuration));
      }
    }).catch(function (e) {
      clear(framesHost);
      framesHost.appendChild(el("p", { class: "empty-state", text: "Could not load frames: " + e.message }));
    });
  }

  // The expert-curation view: the per-frame species distribution, and what an
  // expert's frame verdicts have done to the trusted set. It never restates a
  // measured number as changed; the measured stats live in auditDerivation above
  // and stay as captured. An 'inaccurate' frame is subtracted here and nowhere
  // else.
  function curationView(distribution, summary) {
    var wrap = el("div", { class: "audit-derivation" });
    wrap.appendChild(el("div", { class: "card-title", text: "Per-frame review" }));

    if (distribution.length) {
      var parts = distribution.map(function (d) { return d.count + " " + d.class_name; });
      wrap.appendChild(el("div", { class: "audit-row" }, [
        el("span", { class: "audit-k", text: "Frames by species" }),
        el("span", { class: "audit-v", text: parts.join(", ") })
      ]));
    }
    if (summary && summary.multiple_candidates) {
      wrap.appendChild(el("p", { class: "form-hint", text:
        "This track was read as more than one species across its frames. If it is one organism the model second-guessed, relabel the whole event above and mark the wrong frames Inaccurate; if two organisms really were present, mark each frame accordingly." }));
    }
    if (summary) {
      var total = summary.total_frames || 0;
      var kept = summary.curated_frame_count != null ? summary.curated_frame_count : total;
      var trust = summary.trust != null ? Math.round(summary.trust * 100) + "%" : "not reviewed";
      wrap.appendChild(el("div", { class: "audit-row" }, [
        el("span", { class: "audit-k", text: "Expert-curated" }),
        el("span", { class: "audit-v", text: kept + " of " + total + " frames kept  (" +
          (summary.inaccurate || 0) + " marked inaccurate, " + (summary.accurate || 0) + " confirmed accurate)" })
      ]));
      wrap.appendChild(el("div", { class: "audit-row" }, [
        el("span", { class: "audit-k", text: "Event trust" }),
        el("span", { class: "audit-v", text: trust })
      ]));
    }
    wrap.appendChild(el("p", { class: "form-hint", text:
      "Marking a frame Inaccurate subtracts it from this curated view and from the frame count the longitudinal pass and analytics trust; an event whose every frame is Inaccurate, or one you reject above, is left out of the pass entirely. The measured numbers above never change, so what the model did stays on the record." }));
    return wrap;
  }

  // The two per-frame verdict buttons. A second click on the active verdict
  // clears it, so a frame can return to unreviewed without leaving a false
  // record. The measured frame is never altered; this only appends a verdict.
  function frameReviewButtons(f, obsId, cell, onReviewed, painters) {
    var rowEl = el("div", { class: "frame-review" });
    var accBtn = el("button", { type: "button", class: "frame-review-btn acc", text: "Accurate" });
    var badBtn = el("button", { type: "button", class: "frame-review-btn bad", text: "Inaccurate" });
    function paint() {
      accBtn.classList.toggle("is-on", f.review === "accurate");
      badBtn.classList.toggle("is-on", f.review === "inaccurate");
      accBtn.setAttribute("aria-pressed", f.review === "accurate" ? "true" : "false");
      badBtn.setAttribute("aria-pressed", f.review === "inaccurate" ? "true" : "false");
      cell.classList.toggle("frame-inaccurate", f.review === "inaccurate");
      cell.classList.toggle("frame-accurate", f.review === "accurate");
    }
    // Let a "mark all accurate" action set this frame and repaint it, so the
    // whole strip reflects the bulk verdict without rebuilding.
    if (painters) { painters.push(function () { f.review = "accurate"; paint(); }); }
    function send(target) {
      var verdict = (f.review === target) ? "cleared" : target;
      accBtn.disabled = badBtn.disabled = true;
      apiSend("/detections/" + encodeURIComponent(obsId) + "/frames/" + f.index + "/review", "POST",
        { verdict: verdict }).then(function (res) {
          f.review = (verdict === "cleared") ? null : verdict;
          paint();
          if (onReviewed) { onReviewed(res.review_summary); }
        }).catch(function (e) {
          cell.appendChild(el("div", { class: "frame-review-err", text: e.message }));
        }).then(function () { accBtn.disabled = badBtn.disabled = false; });
    }
    accBtn.addEventListener("click", function () { send("accurate"); });
    badBtn.addEventListener("click", function () { send("inaccurate"); });
    paint();
    rowEl.appendChild(accBtn);
    rowEl.appendChild(badBtn);
    return rowEl;
  }

  // Model trust, an inferred reliability score for a single detection. It is the
  // detection evidence D (the same quantity salience uses) times the model's
  // expert-judged accuracy for the species it called. It is distinct from the
  // per-frame "Event trust" in the curation view below, which is the share of
  // frames kept after review. It is never a measurement, so it is always shown
  // labelled as inference and tagged with the model it belongs to. When the
  // species has no expert reviews under that model it is not computable, and it
  // says "not yet rated" rather than showing a false number. (The response field
  // is still named event_trust internally.)
  function eventTrustText(et) {
    if (!et) { return null; }
    if (et.multiple_species) {
      var names = (et.species_labels || []).join(", ");
      return "multiple species" + (names ? " (" + names + ")" : "") +
        "  -  a single score would represent only one; review each species to rate them (inferred)";
    }
    if (et.computable) {
      return fmtNum(et.value, 2) + "  = detection evidence " + fmtNum(et.detection_evidence, 2) +
        " × accuracy " + fmtNum(et.accuracy, 2) + " for " + (et.species_label || "this species") +
        "  (model " + (et.model_version || "version not recorded") + ", inferred)";
    }
    return "not yet rated  (" + (et.reason || "not computable") + ", inferred)";
  }

  function eventTrustAuditRow(et) {
    if (!et) { return null; }
    return el("div", { class: "audit-row" }, [
      el("span", { class: "audit-k", text: "Model trust" }),
      el("span", { class: "audit-v", text: eventTrustText(et) })
    ]);
  }

  // The compact chip shown on a Detections or Audio card. Computable trust reads
  // as a value; an unrated event reads plainly as such, muted, never as a zero.
  // The full derivation is carried in the title so a hover discloses the model.
  function eventTrustChip(et) {
    if (!et) { return null; }
    if (et.multiple_species) {
      var multi = el("span", { class: "trust-chip muted", text: "model trust: multiple species" });
      multi.title = et.reason || "more than one species in this event";
      return multi;
    }
    if (et.computable) {
      var chip = el("span", { class: "trust-chip", text: "model trust " + fmtNum(et.value, 2) });
      chip.title = "Inferred: detection evidence " + fmtNum(et.detection_evidence, 2) +
        " × accuracy " + fmtNum(et.accuracy, 2) + " for " + (et.species_label || "this species") +
        " under model " + (et.model_version || "version not recorded");
      return chip;
    }
    var muted = el("span", { class: "trust-chip muted", text: "model trust: not yet rated" });
    muted.title = et.reason || "not yet computable";
    return muted;
  }

  function auditDerivation(obs, frames) {
    var wrap = el("div", { class: "audit-derivation" });
    var n = frames.length;
    var confs = frames.map(function (f) { return Number(f.confidence) || 0; });
    var maxConf = confs.length ? Math.max.apply(null, confs) : null;
    var times = frames.map(function (f) { return f.captured_at; }).filter(Boolean);
    function row(label, value, ok) {
      var v = el("span", { class: "audit-v", text: value });
      if (ok === true) { v.appendChild(el("span", { class: "audit-ok", text: "  matches" })); }
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
    var etRow = eventTrustAuditRow(obs.event_trust);
    if (etRow) { wrap.appendChild(etRow); }
    wrap.appendChild(el("p", { class: "form-hint", text:
      "Each frame below is a saved detection with its own confidence and box, so the frame count and the true duration are directly verifiable. Confidence is the peak across frames; salience is computed from the whole record at capture. Model trust is inferred, not measured: it multiplies the detection evidence by this model's expert-judged accuracy for the species, and it never changes a stored value. It is a different quantity from the per-frame Event trust in the curation section." }));
    return wrap;
  }

  function auditFrameStrip(frames, caption, obsId, onReviewed) {
    var wrap = el("div", { class: "audit-strip-wrap" });
    var painters = [];
    // The caption doubles as a control: a "mark all Accurate" link lets a
    // reviewer accept the whole event at once and then only mark the few wrong
    // frames Inaccurate (a later per-frame verdict wins). It reads as plain text
    // set apart from the buttons, per the review layout.
    var cap = el("div", { class: "card-note" }, [
      el("span", { text: frames.length + " frames · scroll to review · click a frame to enlarge · mark each Accurate or Inaccurate · " })
    ]);
    var markAll = el("span", { class: "link-inline", text: "mark all Accurate", role: "button", tabindex: "0" });
    function doMarkAll() {
      if (!obsId || !frames.length) { return; }
      if (!window.confirm("Mark all " + frames.length + " frames as accurate? You can still mark individual frames Inaccurate afterward.")) { return; }
      markAll.textContent = "marking all accurate...";
      apiSend("/detections/" + encodeURIComponent(obsId) + "/frames/review-all", "POST", { verdict: "accurate" })
        .then(function (res) {
          painters.forEach(function (fn) { fn(); });
          if (onReviewed) { onReviewed(res.review_summary); }
          markAll.textContent = "all marked Accurate";
        })
        .catch(function (e) { markAll.textContent = "could not mark all: " + e.message; });
    }
    markAll.addEventListener("click", doMarkAll);
    markAll.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); doMarkAll(); } });
    if (obsId && frames.length) { cap.appendChild(markAll); }
    wrap.appendChild(cap);
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
        (f.confidence != null ? "  ·  " + Math.round(Number(f.confidence) * 100) + "%" : "") +
        (f.class_name ? "  ·  " + f.class_name : "") }));
      if (obsId != null && f.index != null) {
        cell.appendChild(frameReviewButtons(f, obsId, cell, onReviewed, painters));
      }
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
    pollTimer: null
  };

  // Per-list-panel state. Detections and Audio share the same machinery (a
  // species filter, a paged window, and a multi-select delete mode) but each
  // keeps its own slice so switching tabs never carries one panel's filter,
  // page, or selection into the other. Each context also names the endpoints it
  // reads and the loader that redraws it, so the shared helpers stay generic.
  //   speciesFilter : the chosen species (server-side filter)
  //   ids           : the ids currently shown, so Select all knows its scope
  //   selecting     : whether the delete selection mode is on
  //   selection     : the set of ids ticked for deletion
  //   pageSize      : page size (PAGE_ALL = every match)
  //   pageOffset    : current offset
  //   sig           : a signature so changing filters resets to page one while a
  //                   plain page turn does not
  //   stationId     : this panel's own station choice, so Detections and Audio
  //                   can sit on different stations at the same time. It starts
  //                   at every station and is remembered per panel.
  function makeListCtx(opts) {
    return {
      stationId: store(opts.stationStoreKey) || "",
      stationStoreKey: opts.stationStoreKey,
      speciesFilter: "", ids: [], selecting: false, selection: {},
      pageSize: 100, pageOffset: 0, sig: null,
      listEndpoint: opts.listEndpoint,
      speciesEndpoint: opts.speciesEndpoint,
      deleteEndpoint: opts.deleteEndpoint,
      unit: opts.unit,
      reload: opts.reload
    };
  }
  state.det = makeListCtx({
    listEndpoint: "/detections", speciesEndpoint: "/detections/species",
    deleteEndpoint: "/detections/delete", unit: "detection",
    stationStoreKey: STORE.stationDetections,
    reload: function () { loaders.detections(); }
  });
  state.aud = makeListCtx({
    listEndpoint: "/audio", speciesEndpoint: "/audio/species",
    deleteEndpoint: "/detections/delete", unit: "acoustic detection",
    stationStoreKey: STORE.stationAudio,
    reload: function () { loaders.audio(); }
  });

  // ----------------------------------------------------------------------
  // 5. Theme control
  // ----------------------------------------------------------------------

  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") || DEFAULT_THEME;
  }

  // Apply a theme to the page and remember it. By default the choice is also
  // saved on the hub so it survives a restart; pass { persist: false } for a
  // programmatic apply (startup, or echoing a value just read from the hub) that
  // should not write back.
  function applyTheme(name, opts) {
    if (!THEMES[name]) { name = DEFAULT_THEME; }
    document.documentElement.setAttribute("data-theme", name);
    store(STORE.theme, name);
    if (THEMES[name].mode === "dark") { store(STORE.lastDark, name); }
    else { store(STORE.lastLight, name); }
    var picker = $("#appearance-theme");
    if (picker) { picker.value = name; }
    if (!opts || opts.persist !== false) { persistTheme(); }
  }

  // Remember the theme on the hub, not just in this browser, so the choice is the
  // same the next time Audtheia is opened and on any device that opens it. A hub
  // that is a field node, or one that cannot be reached, keeps only the browser
  // copy and simply does not persist, so the choice is never lost either way.
  function persistTheme() {
    var changes = [
      { scope: "global", field: "ui_theme", value: currentTheme() },
      { scope: "global", field: "ui_last_dark", value: store(STORE.lastDark) || null },
      { scope: "global", field: "ui_last_light", value: store(STORE.lastLight) || null }
    ];
    apiSend("/settings/update", "POST", { changes: changes }).catch(function () { /* the browser copy still holds the choice */ });
  }

  function toggleMode() {
    var mode = THEMES[currentTheme()].mode;
    if (mode === "dark") { applyTheme(store(STORE.lastLight) || "forest"); }
    else { applyTheme(store(STORE.lastDark) || "ocean"); }
  }

  function initTheme() {
    // Apply the browser's remembered choice at once and without saving, so the
    // interface never flashes a default before a preference loads.
    applyTheme(store(STORE.theme) || currentTheme() || DEFAULT_THEME, { persist: false });
    var toggle = $("[data-theme-toggle]");
    if (toggle) { toggle.addEventListener("click", toggleMode); }
    loadServerTheme();
  }

  // Prefer the theme saved on the hub when there is one, so a fresh browser or a
  // second device opens with the same look. The server value, when present, is
  // applied without saving it straight back.
  function loadServerTheme() {
    apiGet("/settings").then(function (s) {
      var ui = (s.config && s.config.ui) || {};
      if (ui.last_dark && THEMES[ui.last_dark]) { store(STORE.lastDark, ui.last_dark); }
      if (ui.last_light && THEMES[ui.last_light]) { store(STORE.lastLight, ui.last_light); }
      if (ui.theme && THEMES[ui.theme] && ui.theme !== currentTheme()) {
        applyTheme(ui.theme, { persist: false });
      }
    }).catch(function () { /* offline or no saved theme: the browser copy stands */ });
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
    // The longitudinal pass keeps its own small timer; leaving the panel stops it.
    if (state.dreamTimer) { window.clearTimeout(state.dreamTimer); state.dreamTimer = null; }
  }

  // The live views (detections, the longitudinal status) re-read on the poll;
  // the reference views load once when opened.
  // Brain is deliberately not a live panel. It is reference and status, and
  // rebuilding all of it every few seconds threw the reader back to the top of
  // the page and flickered. The one part that genuinely changes while you watch,
  // the longitudinal pass, refreshes itself in place while a pass is running.
  var LIVE_PANELS = { detections: true, audio: true, analytics: true, sensors: true };

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
      state.pollTimer = window.setInterval(function () {
        // Rebuilding the list while a clip is playing would tear out the <audio>
        // element and stop it, so a live refresh waits until nothing is playing.
        var playing = Array.prototype.some.call(
          document.querySelectorAll("audio"),
          function (a) { return !a.paused && !a.ended; }
        );
        if (playing) { return; }
        loaders[name]();
      }, POLL_INTERVAL_MS);
    }
  }

  // Brain is a reference panel and deliberately does not poll, so its sub-panels
  // are re-read whenever their pill is selected. That way an expert correction
  // made on Detections or Audio is reflected in the audit the moment the reader
  // returns to it, without a separate refresh control that would be ambiguous
  // among three pills. The reload is skipped unless Brain is the active panel, so
  // restoring a stored pill at startup does not fetch a hidden panel's data.
  var SUBPANEL_LOADERS = {
    "brain-models": function () { loadBrainModels(); },
    "brain-learning": function () { loadBrainLearning(); },
    "brain-skills": function () { loadBrainSkills(); }
  };

  function activateSubpanel(name) {
    $all(".subpanel").forEach(function (p) { p.hidden = p.getAttribute("data-subpanel") !== name; });
    $all(".subnav-item").forEach(function (b) {
      if (b.getAttribute("data-subpanel") === name) { b.setAttribute("aria-current", "page"); }
      else { b.removeAttribute("aria-current"); }
    });
    store(STORE.subpanel, name);
    var reload = SUBPANEL_LOADERS[name];
    if (reload && state.activePanel === "brain") { reload(); }
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
    var ctx = state.det;
    var host = region("detections-list");
    var filters = region("detections-filters");
    if (filters && !filters.hasChildNodes()) {
      filters.appendChild(filterBar(loaders.detections, { species: true, ctx: ctx }));
      filters.appendChild(captureSourceControl("video", loaders.detections));
      filters.appendChild(captureRunControl("video", loaders.detections));
      filters.appendChild(deleteControl(ctx));
    }
    // Reset to the first page whenever the station, species, or page size
    // changes; a plain page turn (prev/next) keeps the same signature.
    var sig = ctx.stationId + "|" + ctx.speciesFilter + "|" + ctx.pageSize;
    if (sig !== ctx.sig) { ctx.pageOffset = 0; ctx.sig = sig; }

    // Fill the species dropdown from the full server-side list (every recorded
    // species), not just the current page.
    apiGet(ctx.speciesEndpoint + query({ station_id: ctx.stationId }))
      .then(function (list) { populateSpecies(ctx, list || []); })
      .catch(function () {});

    apiGet(ctx.listEndpoint + query({
      station_id: ctx.stationId,
      species: ctx.speciesFilter,
      limit: ctx.pageSize,
      offset: ctx.pageOffset
    })).then(function (resp) {
      clear(host);
      var rows = (resp && resp.items) || [];
      var total = (resp && resp.total) || 0;
      ctx.ids = rows.map(function (obs) { return obs.id; });

      if (!rows.length) {
        setState(host, "empty-state", (ctx.speciesFilter || ctx.stationId)
          ? "No detections match these filters."
          : "No detections yet. Use Set capture source to run desktop detection, or connect a Pi under Settings, Stations.");
        updateDeleteUI(ctx);
        return;
      }

      var grid = el("div", { class: "card-grid" + (ctx.selecting ? " selecting" : "") });
      rows.forEach(function (obs) {
        var v = obs.verification;
        var species = (obs.vision_detections || []).map(taxonName).join(", ") || "no resolved taxon";
        var badges = provenanceBadges(obs.data_source, obs.qc_state);
        if (v && v.verified) { badges.push(badge("verified", "ok")); }
        else { badges.push(badge("not verified", "muted")); }
        correctionBadges(obs).forEach(function (b) { badges.push(b); });
        var meta = [
          obs.event_name || obs.id,
          fmtTime(obs.first_seen),
          "trigger: " + (obs.trigger_source || "unknown")
        ].join(" · ");
        var card = el("article", { class: "card" });

        if (ctx.selecting) { attachSelectCheckbox(ctx, obs, card); }

        if (obs.representative_frame) {
          card.appendChild(detectionFrame(obs.representative_frame, species, obs.vision_detections,
            (function (o) { return function () { openFrameAudit(o, ctx.reload); }; })(obs)));
          card.appendChild(el("div", { class: "frame-note", text: "highest-confidence frame of this event" }));
        }
        card.appendChild(el("div", { class: "card-meta", text: meta }));
        card.appendChild(el("div", { class: "card-title", text: displaySpecies(obs, species) }));
        card.appendChild(el("div", { class: "card-stats", text:
          "tracked across " + fmtNum(obs.frame_count) + " frames" +
          "  ·  " + fmtNum(obs.duration, 1) + "s" +
          "  ·  " + displayConfidence(obs, obs.screening_confidence) +
          "  ·  salience " + fmtNum(obs.salience_provisional, 2) }));
        var trustChip = eventTrustChip(obs.event_trust);
        if (trustChip) { card.appendChild(el("div", { class: "card-trust" }, [trustChip])); }
        card.appendChild(el("div", { class: "badge-row" }, badges));
        grid.appendChild(card);
      });
      host.appendChild(grid);
      host.appendChild(pager(ctx, total));
      updateDeleteUI(ctx);
    }).catch(function (e) { setState(host, "empty-state", "Could not load detections: " + e.message); });
  };

  // The bottom-of-list pager: a page-size selector (20/40/80/100/200/All), a
  // "start–end of total" readout, and Prev/Next. Changing the size or turning a
  // page re-runs the panel's loader, which fetches just that page from the
  // server. Shared by Detections and Audio through the panel context.
  function pager(ctx, total) {
    var wrap = el("div", { class: "pager" });
    var isAll = ctx.pageSize >= PAGE_ALL;
    var size = ctx.pageSize;
    var start = total === 0 ? 0 : ctx.pageOffset + 1;
    var end = isAll ? total : Math.min(ctx.pageOffset + size, total);

    var sizeSel = el("select", { class: "pager-size", "aria-label": "Items per page" });
    [20, 40, 80, 100, 200].forEach(function (n) {
      var o = el("option", { value: String(n), text: String(n) });
      if (!isAll && n === size) { o.selected = true; }
      sizeSel.appendChild(o);
    });
    var allOpt = el("option", { value: "all", text: "All" });
    if (isAll) { allOpt.selected = true; }
    sizeSel.appendChild(allOpt);
    sizeSel.addEventListener("change", function () {
      ctx.pageSize = sizeSel.value === "all" ? PAGE_ALL : parseInt(sizeSel.value, 10);
      ctx.reload();
    });
    wrap.appendChild(el("label", { class: "filter-field" }, [el("span", { text: "Per page" }), sizeSel]));

    wrap.appendChild(el("span", { class: "pager-info", text: total ? (start + "–" + end + " of " + total) : "0 of 0" }));

    var prev = el("button", { type: "button", class: "btn btn-small", text: "Prev" });
    prev.disabled = isAll || ctx.pageOffset <= 0;
    prev.addEventListener("click", function () {
      ctx.pageOffset = Math.max(0, ctx.pageOffset - size);
      ctx.reload();
    });
    var next = el("button", { type: "button", class: "btn btn-small", text: "Next" });
    next.disabled = isAll || end >= total;
    next.addEventListener("click", function () {
      ctx.pageOffset = ctx.pageOffset + size;
      ctx.reload();
    });
    wrap.appendChild(prev);
    wrap.appendChild(next);
    return wrap;
  }

  // Audio: acoustic detections tied to events, with the true clip duration.
  // Mirrors Detections (a station + species filter, a paged window, multi-select
  // delete, and an audit view) over its own panel context so the two never mix.
  loaders.audio = function () {
    var ctx = state.aud;
    var host = region("audio-list");
    var filters = region("audio-filters");
    if (filters && !filters.hasChildNodes()) {
      filters.appendChild(filterBar(loaders.audio, { species: true, ctx: ctx }));
      filters.appendChild(captureSourceControl("audio", loaders.audio));
      filters.appendChild(captureRunControl("audio", loaders.audio));
      filters.appendChild(deleteControl(ctx));
    }
    var sig = ctx.stationId + "|" + ctx.speciesFilter + "|" + ctx.pageSize;
    if (sig !== ctx.sig) { ctx.pageOffset = 0; ctx.sig = sig; }

    apiGet(ctx.speciesEndpoint + query({ station_id: ctx.stationId }))
      .then(function (list) { populateSpecies(ctx, list || []); })
      .catch(function () {});

    apiGet(ctx.listEndpoint + query({
      station_id: ctx.stationId,
      species: ctx.speciesFilter,
      limit: ctx.pageSize,
      offset: ctx.pageOffset
    })).then(function (resp) {
      clear(host);
      var rows = (resp && resp.items) || [];
      var total = (resp && resp.total) || 0;
      ctx.ids = rows.map(function (obs) { return obs.id; });

      if (!rows.length) {
        setState(host, "empty-state", (ctx.speciesFilter || ctx.stationId)
          ? "No acoustic detections match these filters."
          : "No acoustic detections yet. Set an audio source, or connect a Pi with a microphone or hydrophone.");
        updateDeleteUI(ctx);
        return;
      }

      var grid = el("div", { class: "card-grid" + (ctx.selecting ? " selecting" : "") });
      rows.forEach(function (a) {
        var species = (a.audio_detections || []).map(taxonName).join(", ") || "unclassified sound";
        var badges = provenanceBadges(a.data_source, a.qc_state);
        var av = a.verification;
        if (av && av.verified) { badges.push(badge("verified", "ok")); }
        else { badges.push(badge("not verified", "muted")); }
        correctionBadges(a).forEach(function (b) { badges.push(b); });
        var aConfs = (a.audio_detections || [])
          .map(function (d) { return Number(d.confidence); })
          .filter(function (x) { return !isNaN(x); });
        var aPeak = aConfs.length ? Math.max.apply(null, aConfs) : null;
        var card = el("article", { class: "card audio-card" });

        if (ctx.selecting) { attachSelectCheckbox(ctx, a, card); }

        // The head (meta, species, stats, badges) opens the audio audit; the
        // player below stays outside it so pressing play never opens the modal.
        var head = el("div", { class: "audio-open", role: "button", tabindex: "0",
          "aria-label": "Open audit for this acoustic detection" });
        head.appendChild(el("div", { class: "card-meta", text: (a.event_name || a.id) + " · " + fmtTime(a.first_seen) }));
        head.appendChild(el("div", { class: "card-title", text: displaySpecies(a, species) }));
        head.appendChild(el("div", { class: "card-stats", text:
          "true duration: " + fmtNum(a.audio_true_duration_seconds, 1) + "s" +
          (a.audio_capped ? " (stored clip capped)" : "") +
          "  ·  " + displayConfidence(a, aPeak) +
          "  ·  salience " + fmtNum(a.salience_provisional, 2) }));
        var audioTrustChip = eventTrustChip(a.event_trust);
        if (audioTrustChip) { head.appendChild(el("div", { class: "card-trust" }, [audioTrustChip])); }
        head.appendChild(el("div", { class: "badge-row" }, badges));
        (function (obs) {
          head.addEventListener("click", function () { openAudioAudit(obs, ctx.reload); });
          head.addEventListener("keydown", function (e) {
            if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openAudioAudit(obs, ctx.reload); }
          });
        })(a);
        card.appendChild(head);

        if (a.audio_clip_path) {
          card.appendChild(el("audio", { class: "audio-clip", controls: true, preload: "none", src: API + "/media" + query({ path: a.audio_clip_path }) }));
        } else {
          card.appendChild(el("div", { class: "card-note", text: "No stored clip for this event." }));
        }
        grid.appendChild(card);
      });
      host.appendChild(grid);
      host.appendChild(pager(ctx, total));
      updateDeleteUI(ctx);
    }).catch(function (e) { setState(host, "empty-state", "Could not load audio: " + e.message); });
  };

  // The audio-audit modal: the acoustic analogue of the frame audit. It replays
  // the stored clip and lists every acoustic detection with its own confidence,
  // then derives the card's numbers (call count, true duration, peak confidence,
  // salience) from those detections so a scientist can verify rather than trust.
  function openAudioAudit(obs, onCorrected) {
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

    var dets = obs.audio_detections || [];
    var species = dets.map(taxonName).join(", ") || "unclassified sound";
    var title = el("div", { class: "audit-title", text: displaySpecies(obs, species) });
    panel.appendChild(title);
    panel.appendChild(el("div", { class: "card-meta", text: obs.event_name || obs.id }));

    var body = el("div", { class: "audit-body" });
    // The same three verdicts as the visual path, tagged audio so a call heard
    // and an organism seen stay distinguishable in the review record.
    body.appendChild(correctionPanel(obs, "audio", function () {
      title.textContent = displaySpecies(obs, species);
      if (onCorrected) { onCorrected(obs); }
    }));
    body.appendChild(audioAuditDerivation(obs, dets));
    if (obs.audio_clip_path) {
      body.appendChild(el("div", { class: "card-note", text: "Stored clip · the exact audio these calls were recognized from" }));
      body.appendChild(el("audio", { class: "audio-clip audit-clip", controls: true, preload: "none",
        src: API + "/media" + query({ path: obs.audio_clip_path }) }));
    } else {
      body.appendChild(el("p", { class: "empty-state", text: "No stored clip was kept for this event." }));
    }
    body.appendChild(audioDetectionList(dets));
    panel.appendChild(body);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);
  }

  function audioAuditDerivation(obs, dets) {
    var wrap = el("div", { class: "audit-derivation" });
    var n = dets.length;
    var confs = dets.map(function (d) { return Number(d.confidence) || 0; });
    var maxConf = confs.length ? Math.max.apply(null, confs) : null;
    function row(label, value, ok) {
      var v = el("span", { class: "audit-v", text: value });
      if (ok === true) { v.appendChild(el("span", { class: "audit-ok", text: "  matches" })); }
      return el("div", { class: "audit-row" }, [el("span", { class: "audit-k", text: label }), v]);
    }
    wrap.appendChild(el("div", { class: "card-title", text: "How these numbers were derived" }));
    wrap.appendChild(row("Acoustic detections", n + " call" + (n === 1 ? "" : "s") + " recognized  (= " + n + " listed below)"));
    wrap.appendChild(row("True duration", fmtNum(obs.audio_true_duration_seconds, 1) + " s" +
      (obs.audio_capped ? "  (the stored clip is capped; this is the full recognized length)" : "")));
    if (maxConf != null) {
      wrap.appendChild(row("Confidence", Math.round(maxConf * 100) + "%  (highest of the " + n + " per-call values below)"));
    }
    wrap.appendChild(row("Model", obs.acoustic_model_version || "unstated"));
    var v = obs.verification;
    wrap.appendChild(row("Verification", (v && v.verified)
      ? "cleared for the dream pass by the acoustic-confidence gate (peak ≥ the acoustic floor)"
      : "not cleared, peak below the acoustic floor, so it shapes baselines but not the generative phase"));
    wrap.appendChild(row("Salience", fmtNum(obs.salience_provisional, 2) +
      "  = D · (0.5·N + 0.5·R), Shannon-surprisal novelty & rarity (docs/salience.md)"));
    var etRow = eventTrustAuditRow(obs.event_trust);
    if (etRow) { wrap.appendChild(etRow); }
    wrap.appendChild(el("p", { class: "form-hint", text:
      "Each call below is a stored acoustic detection with its own confidence, so the count and the peak confidence are directly verifiable against the clip. Salience is computed from the whole record at capture. Model trust is inferred, not measured: it multiplies the detection evidence by this model's expert-judged accuracy for the species, and it never changes a stored value. It is a different quantity from the per-frame Event trust in the curation section." }));
    return wrap;
  }

  function audioDetectionList(dets) {
    var wrap = el("div", { class: "audit-strip-wrap" });
    if (!dets.length) {
      wrap.appendChild(el("p", { class: "empty-state", text: "No individual acoustic detections were stored for this event." }));
      return wrap;
    }
    wrap.appendChild(el("div", { class: "card-note", text: dets.length + " recognized call" + (dets.length === 1 ? "" : "s") }));
    var list = el("div", { class: "audio-det-list" });
    dets.forEach(function (d) {
      var conf = (d.confidence == null) ? null : Math.round(Number(d.confidence) * 100);
      var col = speciesColor(taxonName(d));
      var swatch = el("span", { class: "audio-det-swatch" });
      swatch.style.backgroundColor = col.bg;
      var barWrap = el("div", { class: "audio-det-bar" });
      var bar = el("div", { class: "audio-det-bar-fill" });
      bar.style.width = (conf == null ? 0 : conf) + "%";
      bar.style.backgroundColor = col.bg;
      barWrap.appendChild(bar);
      list.appendChild(el("div", { class: "audio-det-row" }, [
        swatch,
        el("span", { class: "audio-det-name", text: taxonName(d) }),
        barWrap,
        el("span", { class: "audio-det-conf", text: conf == null ? "n/a" : conf + "%" })
      ]));
    });
    wrap.appendChild(list);
    return wrap;
  }

  // GPS: a self-contained coordinate plot, no external map tiles. It overlays two
  // real things: located detections (each a measured fix or, for a fixed station,
  // an entered position) and each station's own configured coordinates.
  loaders.gps = function () {
    var host = region("gps-map");
    var filters = region("gps-filters");
    if (filters && !filters.hasChildNodes()) { filters.appendChild(filterBar(loaders.gps)); }
    Promise.all([
      apiGet("/gps" + query({ station_id: state.stationId, limit: 1000 })),
      apiGet("/settings"),
      ensureWorldLand()
    ]).then(function (res) {
      var rows = res[0] || [];
      var cfg = (res[1] && res[1].config) || {};
      var world = res[2];

      var detections = rows.filter(function (r) {
        return r.gps_latitude !== null && r.gps_longitude !== null;
      }).map(function (r) {
        return {
          lat: Number(r.gps_latitude), lon: Number(r.gps_longitude),
          label: r.event_name || r.observation_id, kind: "detection", status: r.gps_status
        };
      });

      // Station markers come from each station's own configured coordinates,
      // used only when both a latitude and a longitude are set, and narrowed to
      // the chosen station when one is selected in the filter.
      var stations = (cfg.stations || []).filter(function (st) {
        if (state.stationId && st.station_id !== state.stationId) { return false; }
        var loc = st.location || {};
        return loc.latitude !== null && loc.latitude !== undefined &&
          loc.longitude !== null && loc.longitude !== undefined;
      }).map(function (st) {
        var loc = st.location || {};
        return {
          lat: Number(loc.latitude), lon: Number(loc.longitude),
          label: st.station_name || st.station_id, kind: "station", status: "station_configured"
        };
      });

      var points = stations.concat(detections);
      var measured = detections.filter(function (d) { return d.status !== "station_configured"; }).length;
      var entered = detections.length - measured;

      // Draw for the chosen view mode, and let the toggle redraw from the same
      // data without re-reading the record.
      function render(mode) {
        clear(host);
        host.appendChild(gpsViewToggle(mode, function (next) { store(STORE.gpsView, next); render(next); }));
        host.appendChild(coordinatePlot(points, { view: gpsViewWindow(points, mode), world: world }));
        var parts = [];
        if (measured) { parts.push(fmtNum(measured) + " measured " + (measured === 1 ? "fix" : "fixes")); }
        if (entered) { parts.push(fmtNum(entered) + " entered " + (entered === 1 ? "position" : "positions")); }
        if (stations.length) { parts.push(fmtNum(stations.length) + " station " + (stations.length === 1 ? "marker" : "markers")); }
        var lead = parts.length ? parts.join(", ") + ". " : "No located detections yet, and no station coordinates entered. Add a station's position under Settings, Stations. ";
        host.appendChild(el("p", { class: "card-note", text: lead +
          "Markers are real values only (measured satellite fixes and coordinates entered for a station); the land outline behind them is a generic offline world map for orientation, not per-site imagery." }));
      }

      render(store(STORE.gpsView) || "fit");
    }).catch(function (e) { setState(host, "empty-state", "Could not load spatial data: " + e.message); });
  };

  // Load the bundled, generic world land outline once. It is a static asset
  // served from the same origin as the page, so it needs no internet; if it is
  // ever absent the plot simply draws its grid and markers without the land.
  function ensureWorldLand() {
    if (state.worldLand !== undefined) { return Promise.resolve(state.worldLand); }
    return fetch("world-land.json").then(function (r) { return r.ok ? r.json() : null; })
      .then(function (w) { state.worldLand = w; return w; })
      .catch(function () { state.worldLand = null; return null; });
  }

  // A small segmented control to switch the GPS map between framing the data and
  // showing the whole globe.
  function gpsViewToggle(current, onPick) {
    var wrap = el("div", { class: "map-view-toggle" });
    [["fit", "Fit to data"], ["world", "Whole world"]].forEach(function (o) {
      var b = el("button", { type: "button", class: "btn btn-small" + (current === o[0] ? " btn-primary" : ""), text: o[1] });
      b.addEventListener("click", function () { onPick(o[0]); });
      wrap.appendChild(b);
    });
    return wrap;
  }

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
    // Both modalities are read, so a species that was only ever heard still gets
    // a profile alongside the ones that were seen.
    Promise.all([
      apiGet("/brain/models"),
      apiGet("/brain/memory" + query({ station_id: state.stationId })),
      apiGet("/detections" + query({ station_id: state.stationId, limit: 500 })),
      apiGet("/audio" + query({ station_id: state.stationId, limit: 500 }))
    ]).then(function (res) {
      var models = res[0], memory = res[1];
      var detections = (res[2] && res[2].items) || [];
      var audioEvents = (res[3] && res[3].items) || [];
      clear(host);

      // The desktop hub. Its own stations come first, because they are what is
      // configured and what captures observations; the desktop's own models
      // (verification, the language model) come after, because they act on what
      // the stations produce. A station is placed by what it is configured to do,
      // never assumed to be out in a field.
      host.appendChild(el("h3", { text: "This computer" }));
      host.appendChild(el("p", { class: "settings-desc", text:
        "The desktop hub: the stations you run here, and the models that re-judge and interpret each observation after it has been captured. This computer keeps the authoritative record." }));

      var deskModels = models.desktop_models || {};
      var stations = models.stations || [];
      var desktopStations = stations.filter(function (st) {
        var d = stationDeployment(st);
        return d.desktop || !d.field;
      });
      var fieldStations = stations.filter(function (st) { return stationDeployment(st).field; });

      host.appendChild(el("h4", { text: "Stations run from this computer" }));
      host.appendChild(el("p", { class: "settings-desc", text:
        "Each station you run on the desktop, with its own models: the ones that decide what becomes an observation in the first place. A station with no Pi device stays here." }));
      if (!desktopStations.length) {
        host.appendChild(el("p", { class: "card-note", text: "No station is set up to run on this computer yet." }));
      } else {
        var deskGrid = el("div", { class: "card-grid station-model-grid" });
        desktopStations.forEach(function (st) { deskGrid.appendChild(stationModelCard(st, models.files)); });
        host.appendChild(deskGrid);
      }

      // The desktop's own models run after capture, on the observations the
      // stations above (and any Pi field station below) produce: verification
      // re-scores saved frames, and the language model interprets and drives the
      // longitudinal pass. They belong here, below the stations they act on.
      host.appendChild(el("h4", { text: "Vision verification" }));
      host.appendChild(el("p", { class: "settings-desc", text:
        "Runs on the desktop after capture. It re-scores the saved frames of each observation, for every station whether it runs here or on a Pi." }));
      host.appendChild(el("div", { class: "info-block" }, [modelEntry(
        "Verification model",
        "on this computer",
        "Re-scores the saved frames of an event for publication-grade accuracy. It can overrule a station's call, and it adds the interpretive points such as ecological role and rarity, which are always labelled as inference and never stored as measurement.",
        deskModels.visual_rfdetr, models.files)]));

      // The heading and its description sit above the panel, exactly as the
      // "Vision verification" section does, so this block carries the same
      // comfortable spacing rather than crowding its title against the top edge.
      host.appendChild(el("h4", { text: "Language model" }));
      host.appendChild(el("p", { class: "settings-desc", text:
        "Runs here on your computer. It powers the longitudinal pass and the interpretation text. It does not write your reports: a report is assembled from the stored record, and anything this model produced is labelled in it as inference." }));
      var llmBlock = el("div", { class: "info-block" });
      var llmHost = el("div", { class: "llm-manager" });
      llmBlock.appendChild(llmHost);
      host.appendChild(llmBlock);
      renderLlmManager(llmHost);

      // Pi field stations: only stations configured with a Raspberry Pi device,
      // whose screening model is compiled for the Pi accelerator (a .hef). A
      // station appears here only when it carries that model.
      host.appendChild(el("h3", { text: "Pi Field Stations" }));
      host.appendChild(el("p", { class: "settings-desc", text:
        "Only stations configured with a Raspberry Pi field device appear here. Their screening model runs out on the Pi's accelerator, deciding what becomes an observation before anything reaches this computer." }));
      if (!fieldStations.length) {
        host.appendChild(el("p", { class: "card-note", text: "No station is configured with a Pi device yet." }));
      } else {
        var fieldGrid = el("div", { class: "card-grid station-model-grid" });
        fieldStations.forEach(function (st) { fieldGrid.appendChild(stationModelCard(st, models.files)); });
        host.appendChild(fieldGrid);
      }

      host.appendChild(el("h3", { text: "Site memory" }));
      host.appendChild(el("div", { class: "info-block" }, [siteMemoryView(memory)]));

      host.appendChild(el("h3", { text: "Species profiles" }));
      host.appendChild(speciesProfiles(detections, audioEvents));
    }).catch(function (e) { setState(host, "empty-state", "Could not load models and memory: " + e.message); });
  }

  // Where a station runs, read from how it is configured rather than assumed.
  // Mirrors the backend classifier: a field deployment has a screening model
  // compiled for its accelerator; a desktop station has a desktop screening
  // model or a desktop capture source; a station may be both, or neither.
  function stationDeployment(st) {
    if (st && st.deployment) { return st.deployment; }
    var m = (st && st.models) || {};
    var src = (st && st.capture && st.capture.source) || {};
    var field = !!((m.visual_pi || {}).path);
    var desktop = !!((m.visual_desktop || {}).path) || !!src.video || !!src.audio;
    return { field: field, desktop: desktop, configured: field || desktop };
  }

  function deploymentBadge(dep) {
    dep = dep || {};
    if (dep.field && dep.desktop) { return badge("Runs on this computer and a field station", "ok"); }
    if (dep.desktop) { return badge("Runs on this computer", "ok"); }
    if (dep.field) { return badge("Runs on a field station", "ok"); }
    return badge("Not yet configured to run anywhere", "muted");
  }

  // One station's own models, in the order they act on a moment: the camera's
  // screener first, then the microphone or hydrophone's listener.
  function stationModelCard(st, files) {
    var m = st.models || {};
    var card = el("article", { class: "card" }, [
      el("div", { class: "card-title", text: st.station_name || st.station_id }),
      el("div", { class: "badge-row card-deployment-row" }, [deploymentBadge(stationDeployment(st))])
    ]);
    card.appendChild(modelEntry(
      "Vision screening",
      "on the station's accelerator",
      "Checks every frame the camera produces. A frame with nothing in it is discarded straight away, so only real detections ever reach storage.",
      m.visual_pi, files));

    var acoustic = m.acoustic || {};
    card.appendChild(modelEntry(
      "Acoustic recognition",
      "on the station's processor",
      "Listens to the audio stream and recognises sounds. A recognised sound opens an observation of its own, so the station hears as well as sees.",
      acoustic, files));

    // The desktop screening row is always shown, set or not: a station run on
    // this computer screens with this model, so a missing one is a state to
    // report plainly rather than a row to hide.
    card.appendChild(modelEntry(
      "Desktop screening",
      "stands in when there is no field hardware",
      "Used only when you run this station's capture on your computer, from a video file or a webcam, instead of on a Raspberry Pi.",
      m.visual_desktop, files));
    return card;
  }

  // One model, described in plain language first and by its file second. A
  // citation is kept but folded away, so credit is preserved without burying the
  // rest of the panel under it.
  function modelEntry(title, where, description, entry, files) {
    var wrap = el("div", { class: "model-entry" });
    // The title sits on its own line, and the "where it runs" badge below it
    // rather than pushed out to the right, so the layout stays clean and reads
    // the same no matter how many stations are shown.
    wrap.appendChild(el("div", { class: "model-entry-head" }, [
      el("span", { class: "form-label", text: title })
    ]));
    if (where) { wrap.appendChild(el("div", { class: "badge-row model-entry-where" }, [badge(where, "muted")])); }
    wrap.appendChild(el("p", { class: "card-note", text: description }));

    // A version without a path is not a model, so presence of a path is the only
    // test. Stated as its own status rather than another line of prose, so it is
    // not mistaken for part of the description above it.
    if (!entry || !entry.path) {
      wrap.appendChild(el("p", { class: "model-status is-absent", text: "No model set" }));
      return wrap;
    }
    var kv = el("div", { class: "kv-list" });
    // A model path can be long, so it stacks under its label and wraps rather
    // than running past the edge of the card.
    if (entry.path) { kv.appendChild(modelKvRow("File", entry.path, "model-file")); }
    kv.appendChild(modelKvRow("Version", entry.version || "not stated"));
    wrap.appendChild(kv);

    // A configured path is intent, not proof the file arrived. Several models are
    // downloaded or exported by hand after setup, so a path pointing at nothing
    // must not look like a model that is ready to run.
    if (entry.path) {
      var info = (files || {})[entry.path];
      if (info && info.present) {
        wrap.appendChild(el("p", { class: "model-status is-present", text: "File present, " + fmtBytes(info.size_bytes) }));
      } else if (info) {
        wrap.appendChild(el("p", { class: "model-status is-absent", text: "No file at this path yet" }));
      }
    }

    if (entry.citation) {
      var cite = el("details", { class: "subgroup" }, [el("summary", { text: "Citation" })]);
      cite.appendChild(el("p", { class: "card-meta model-citation", text: entry.citation }));
      wrap.appendChild(cite);
    }
    return wrap;
  }

  function modelKvRow(k, v, extraClass) {
    return el("div", { class: "kv-row" + (extraClass ? " " + extraClass : "") }, [
      el("span", { class: "kv-key", text: k }),
      el("span", { class: "kv-val", text: v })
    ]);
  }

  // Site memory explained where it is shown, rather than assumed. This is the
  // yardstick the salience score is measured against, so it is worth stating
  // plainly what it holds and that it never touches the observation archive.
  function siteMemoryView(memory) {
    var wrap = el("div");
    var count = memory.baseline_count || (memory.site_baselines || []).length;
    wrap.appendChild(el("p", { class: "settings-desc", text:
      "Site memory is a site's long-term sense of what is normal. The longitudinal pass compresses the record into compact statistical cells, one for each station, recurring period (every June, for example), taxon group and measured signal. Each cell holds the middle value, how much it usually varies, its range, and how many readings went into it." }));
    wrap.appendChild(el("p", { class: "card-note", text:
      "That is the yardstick for \"is this unusual?\": a new observation is scored against its matching cell. It is permanent and is never pruned. A pass only ever trims its own working notes, never your observation archive." }));
    wrap.appendChild(metricRow([metricCard("Baseline cells", fmtNum(count))]));
    if (!count) {
      wrap.appendChild(el("p", { class: "card-note", text:
        "There are none yet because no longitudinal pass has run. They fill in after the first pass." }));
    }
    wrap.appendChild(deferredNote("Editing or annotating memory writes desktop-owned records, which arrives with the memory write path."));
    return wrap;
  }

  // The desktop language model that powers the dream pass and interpretation:
  // what is installed, which one is active, and controls to select a model or
  // learn how to drop a new one in. A model change applies on the next start.
  function renderLlmManager(host) {
    clear(host);
    host.appendChild(el("p", { class: "card-note", text: "Loading language model." }));
    apiGet("/brain/llm").then(function (info) {
      clear(host);
      // The honest readiness line, and the exact remedy when something is wrong,
      // so the manager states the fix rather than only that the model is absent.
      var statusOk = info.status === "model_present";
      host.appendChild(el("p", { class: statusOk ? "card-note" : "form-message", text:
        (info.status_message || (info.runtime_available ? "The model runtime is installed." : "The model runtime is not installed.")) }));
      if (info.remedy && !statusOk) {
        host.appendChild(el("p", { class: "form-hint", text: info.remedy }));
      }

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

  // One profile card per taxon, built from BOTH modalities, so a species that
  // was only ever heard appears beside the ones that were seen. Sightings and
  // calls are counted separately, because a photograph and a recorded call are
  // different kinds of evidence and should not be silently added together.
  function speciesProfiles(visionEvents, audioEvents) {
    var byTaxon = {};
    var stationNames = {};
    (state.stations || []).forEach(function (s) { stationNames[s.id] = s.station_name || s.id; });

    function add(obs, det, modality) {
      var key = taxonName(det);
      if (!key) { return; }
      if (!byTaxon[key]) {
        byTaxon[key] = {
          name: key, scientific: det.scientific_name || "",
          vision: 0, audio: 0, sum: 0, scored: 0, best: 0, last: null, sites: {}
        };
      }
      var t = byTaxon[key];
      t[modality] += 1;
      if (det.confidence !== null && det.confidence !== undefined) {
        t.sum += Number(det.confidence);
        t.scored += 1;
        t.best = Math.max(t.best, Number(det.confidence));
      }
      if (!t.last || (obs.first_seen && obs.first_seen > t.last)) { t.last = obs.first_seen; }
      if (obs.station_id) { t.sites[obs.station_id] = true; }
    }

    (visionEvents || []).forEach(function (obs) {
      (obs.vision_detections || []).forEach(function (d) { add(obs, d, "vision"); });
    });
    (audioEvents || []).forEach(function (obs) {
      (obs.audio_detections || []).forEach(function (d) { add(obs, d, "audio"); });
    });

    var taxa = Object.keys(byTaxon).map(function (k) { return byTaxon[k]; })
      .sort(function (a, b) { return (b.vision + b.audio) - (a.vision + a.audio); });
    if (!taxa.length) { return el("p", { class: "card-note", text: "No resolved taxa in the record yet." }); }

    var grid = el("div", { class: "card-grid" });
    taxa.forEach(function (t) {
      var sites = Object.keys(t.sites).map(function (id) { return stationNames[id] || id; });
      var modes = el("div", { class: "badge-row" });
      if (t.vision) { modes.appendChild(badge(fmtNum(t.vision) + (t.vision === 1 ? " sighting" : " sightings"), "source")); }
      if (t.audio) { modes.appendChild(badge(fmtNum(t.audio) + (t.audio === 1 ? " call" : " calls"), "source")); }
      grid.appendChild(el("article", { class: "card" }, [
        el("div", { class: "card-title", text: t.name }),
        (t.scientific && t.scientific !== t.name) ? el("div", { class: "card-meta", text: t.scientific }) : null,
        modes,
        el("div", { class: "card-stats", text:
          "mean confidence: " + fmtConfidence(t.scored ? t.sum / t.scored : null) +
          "  best: " + fmtConfidence(t.best) }),
        sites.length ? el("div", { class: "card-meta", text: "recorded at: " + sites.join(", ") }) : null,
        el("div", { class: "card-meta", text: "last recorded: " + fmtTime(t.last) })
      ]));
    });
    return grid;
  }

  // Brain, Learning: the live longitudinal status with pause and resume, the
  // candidate patterns it has produced, and an audit summary.
  // Paint the candidate-pattern grid into its own region, so a run-now refresh
  // can repaint just this without rebuilding the whole panel.
  function paintPatterns(host, learning) {
    clear(host);
    var patterns = (learning && learning.patterns) || [];
    if (!patterns.length) { host.appendChild(el("p", { class: "card-note", text: "No candidate patterns yet." })); return; }
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

  function loadBrainLearning() {
    var host = region("brain-learning");
    clear(host);
    host.appendChild(el("p", { class: "empty-state", text: "Loading learning and audit history." }));

    function auditDesc() {
      return el("p", { class: "settings-desc", text:
        "Evidence of how the system behaved, counted from the stored record. Every figure here is a count or an average over rows already written, never a new measurement and never an interpretation." });
    }

    Promise.all([
      apiGet("/dream/status"),
      apiGet("/brain/learning"),
      apiGet("/analytics" + query({ station_id: state.stationId })),
      apiGet("/brain/audit" + query({ station_id: state.stationId }))
    ]).then(function (res) {
      var status = res[0], learning = res[1], analytics = res[2], audit = res[3];
      clear(host);

      host.appendChild(el("h3", { text: "Longitudinal pass" }));
      host.appendChild(el("p", { class: "card-note", text:
        "It reads the whole verified record for patterns, each offered as a candidate hypothesis. The trend and correlation detectors work on environmental sensor readings, so a station with sensors gives the pass its fullest picture; species co-occurrence and the site baseline still build without them." }));
      var dreamHost = el("div");
      host.appendChild(dreamHost);
      paintDreamStatus(dreamHost, status);

      // Candidate patterns sit directly under the pass that produces them, so the
      // run and its results read as one thing.
      host.appendChild(el("h3", { text: "Candidate patterns" }));
      var patternsHost = el("div");
      host.appendChild(patternsHost);
      paintPatterns(patternsHost, learning);

      host.appendChild(el("h3", { text: "Run now" }));
      var runHost = el("div");
      host.appendChild(runHost);

      host.appendChild(el("h3", { text: "Species data" }));
      var speciesHost = el("div");
      host.appendChild(speciesHost);
      renderSpeciesData(speciesHost);

      host.appendChild(el("h3", { text: "Audit" }));
      host.appendChild(auditDesc());
      var auditHost = el("div");
      host.appendChild(auditHost);
      auditHost.appendChild(auditView(audit, analytics));

      // Per-species model accuracy sits directly above the retraining export it
      // informs: the weakest species are the ones worth exporting and retraining.
      host.appendChild(el("h3", { text: "Model accuracy by species" }));
      var trustHost = el("div");
      host.appendChild(trustHost);
      renderModelTrust(trustHost, audit.model_trust);

      host.appendChild(el("h3", { text: "Retraining exports" }));
      var retrainHost = el("div");
      host.appendChild(retrainHost);
      renderRetraining(retrainHost);

      // A run-now action refreshes only the regions its run can change: the
      // pass status, the audit, and the candidate patterns. It never rebuilds
      // the run-now controls, so the outcome message a button just wrote stays
      // on screen. A refresh failure is swallowed rather than allowed to wipe
      // that outcome.
      function refreshData() {
        return Promise.all([
          apiGet("/dream/status"),
          apiGet("/brain/learning"),
          apiGet("/analytics" + query({ station_id: state.stationId })),
          apiGet("/brain/audit" + query({ station_id: state.stationId }))
        ]).then(function (r2) {
          clear(dreamHost); paintDreamStatus(dreamHost, r2[0]);
          clear(auditHost); auditHost.appendChild(auditView(r2[3], r2[2]));
          renderModelTrust(trustHost, r2[3].model_trust);
          paintPatterns(patternsHost, r2[1]);
        }).catch(function () { /* keep the run-now outcome visible on a refresh error */ });
      }

      renderRunNowControls(runHost, refreshData);
    }).catch(function (e) { setState(host, "empty-state", "Could not load learning: " + e.message); });
  }

  // Retraining exports: gather the weakest and most disputed detections into a
  // folder a person can correct and train from. The counts are previewed first,
  // from the same selection the export itself uses, so nothing is written until
  // someone has seen what they are about to get.
  function renderRetraining(host) {
    clear(host);
    var block = el("div", { class: "info-block" });
    block.appendChild(el("p", { class: "settings-desc", text:
      "A monitoring system earns its accuracy back by learning from the cases it handled worst. This gathers the weakest detections, the ones the desktop verifier disputed, and anything the station could not classify, and writes them out with their labels and boxes already in place, ready to be corrected and trained on." }));
    block.appendChild(el("p", { class: "card-note", text:
      "The labels in an export are the models' own guesses, chosen precisely because they are doubtful. Treat every one as a question to answer, not as truth. A detection that turns out to be an organism the model has never been trained on is the most valuable item in the folder." }));

    var thresholdInput = el("input", { type: "number", class: "form-input", step: "0.05", min: "0", max: "1", value: "0.45" });
    var counts = el("div", { class: "kv-list" });
    var message = el("p", { class: "form-message" });

    function refresh() {
      clear(counts);
      counts.appendChild(el("p", { class: "card-note", text: "Counting candidates." }));
      apiGet("/brain/retraining/candidates" + query({
        station_id: state.stationId, confidence_below: thresholdInput.value || 0.45
      })).then(function (c) {
        clear(counts);
        [["vision", "Visual frames to review", visionBtn], ["acoustic", "Audio clips to review", audioBtn]].forEach(function (row) {
          var group = c[row[0]] || {};
          var total = group.detections || 0;
          var reasons = Object.keys(group.by_reason || {}).map(function (k) {
            return humanize(k) + " " + fmtNum(group.by_reason[k]);
          }).join(", ");
          counts.appendChild(modelKvRow(row[1], total
            ? fmtNum(total) + " from " + fmtNum(group.events || 0) + " events" + (reasons ? " (" + reasons + ")" : "")
            : "nothing to write"));
          // Nothing to export means the button is not offered, rather than
          // offering it and answering with a refusal.
          row[2].disabled = !total;
        });
      }).catch(function (e) { setState(counts, "card-note", "Could not count candidates: " + e.message); });
    }

    function runExport(kind, force) {
      message.textContent = "Writing the " + kind + " package.";
      apiSend("/brain/retraining/export", "POST", {
        kind: kind,
        station_id: state.stationId || null,
        confidence_below: Number(thresholdInput.value) || 0.45,
        force: !!force
      }).then(function (r) {
        if (r.already_exists) {
          // The same detections were exported before, so nothing was rewritten.
          clear(message);
          message.appendChild(el("span", { text:
            "These exact " + fmtNum(r.detections) + " detections were already exported to " + r.path +
            ". Nothing was written again. " }));
          message.appendChild(el("button", {
            type: "button", class: "btn btn-small", text: "Export a fresh copy anyway",
            onclick: function () { runExport(kind, true); }
          }));
          return;
        }
        var made = kind === "vision"
          ? fmtNum(r.images) + " images with " + fmtNum(r.annotations) + " boxes"
          : fmtNum(r.clips) + " clips across " + fmtNum(r.labels) + " labels";
        message.textContent = "Wrote " + made + " to " + r.path +
          (r.missing_media ? " (" + fmtNum(r.missing_media) + " files were missing from storage)" : "");
      }).catch(function (e) { message.textContent = "Could not export: " + e.message; });
    }

    var refreshBtn = el("button", { type: "button", class: "btn", text: "Count candidates" });
    refreshBtn.addEventListener("click", refresh);
    var visionBtn = el("button", { type: "button", class: "btn btn-primary", text: "Export visual package" });
    visionBtn.addEventListener("click", function () { runExport("vision"); });
    var audioBtn = el("button", { type: "button", class: "btn btn-primary", text: "Export audio package" });
    audioBtn.addEventListener("click", function () { runExport("acoustic"); });

    block.appendChild(el("label", { class: "form-field" }, [
      el("span", { class: "form-label", text: "Treat a detection as weak below this confidence" }),
      thresholdInput,
      el("span", { class: "form-hint", text: "Detections the desktop verifier disputed, and records the station could not classify, are always included whatever this is set to." })
    ]));
    block.appendChild(counts);
    block.appendChild(el("div", { class: "form-actions" }, [refreshBtn, visionBtn, audioBtn]));
    block.appendChild(message);
    block.appendChild(el("p", { class: "form-hint", text:
      "Each package holds a README with the steps to follow. A retrained desktop model is pointed at from the settings; a retrained field model must first be compiled to the accelerator's own format, which is a one-time build step on an x86 Linux machine and never something a deployed station does." }));

    host.appendChild(block);
    refresh();
  }

  // The audit view: evidence of how the system behaved, all of it counted from
  // rows already stored. Where a stage has not run yet it says so plainly,
  // rather than showing a zero that could be misread as a result.
  function auditView(audit, analytics) {
    var wrap = el("div");
    var verdicts = audit.verification || {};
    var confidence = audit.confidence || {};
    var byModality = confidence.by_modality || {};

    // "Verified share" is the automated desktop verifier only. When it has not run
    // the card says so rather than showing 0.0%, because an absence and a zero are
    // different claims; expert identifications are counted on their own line below.
    var expertSummary = audit.expert || {};
    wrap.appendChild(metricRow([
      metricCard("Events", fmtNum(audit.events)),
      metricCard("Species", fmtNum(analytics && analytics.species_richness)),
      metricCard("Auto-verified share", verdicts.with_verdict
        ? fmtPct(audit.events ? (verdicts.verified || 0) / audit.events : 0)
        : "none run"),
      metricCard("Expert IDs", fmtNum(expertSummary.targets || 0))
    ]));

    // Expert identifications are human verdicts. They are shown on their own line
    // and never folded into the automated "Verified share" above, because a
    // person's judgement and a model's re-score are different kinds of claim. This
    // is what keeps a 0.0% automated share from reading as "nothing is trusted"
    // when a reviewer has already confirmed detections by hand.
    var expert = audit.expert || {};
    var expBlock = el("div", { class: "info-block" });
    expBlock.appendChild(el("div", { class: "card-title", text: "Expert identifications" }));
    expBlock.appendChild(el("p", { class: "settings-desc", text:
      "What a human reviewer has confirmed, relabelled, or rejected. These are counted here as their own producer and are never merged into the automated verifier's share above, because a person's identification and a model's re-score are different claims with different provenance." }));
    if (expert.available === false) {
      expBlock.appendChild(el("p", { class: "card-note", text:
        "This record predates the corrections store, so no expert identification can be counted here yet." }));
    } else if (!expert.targets) {
      expBlock.appendChild(el("p", { class: "card-note", text:
        "No expert identification has been recorded yet. Confirm, relabel, or reject a detection on Detections or Audio and it is counted here." }));
    } else {
      var ekv = el("div", { class: "kv-list" });
      ekv.appendChild(modelKvRow("Events with an expert identification", fmtNum(expert.observations_with_correction)));
      ekv.appendChild(modelKvRow("Confirmed", fmtNum(expert.confirm)));
      ekv.appendChild(modelKvRow("Relabelled", fmtNum(expert.relabel)));
      ekv.appendChild(modelKvRow("Rejected", fmtNum(expert.reject)));
      expBlock.appendChild(ekv);
      expBlock.appendChild(el("p", { class: "card-note", text:
        "The first line counts events; Confirmed, Relabelled, and Rejected count individual corrected targets, so they can differ when one event carries more than one verdict." }));
    }
    wrap.appendChild(expBlock);

    // How often the desktop verifier reached the same conclusion as the field.
    var verBlock = el("div", { class: "info-block" });
    verBlock.appendChild(el("div", { class: "card-title", text: "Desktop verification" }));
    verBlock.appendChild(el("p", { class: "settings-desc", text:
      "How often the desktop verifier, re-scoring an event's saved frames, reached the same conclusion as the field station's screening model. This is the automated verifier only; a human reviewer's identifications are counted separately under Expert identifications above. A disagreement is not an error: it is the desktop correcting a fast field call, and it is recorded rather than hidden." }));
    if (!verdicts.with_verdict) {
      verBlock.appendChild(el("p", { class: "card-note", text:
        "No desktop verification has run yet, so there is nothing to compare. These figures fill in once verification runs over the synced record." }));
    } else {
      var vkv = el("div", { class: "kv-list" });
      vkv.appendChild(modelKvRow("Events with a verdict", fmtNum(verdicts.with_verdict)));
      vkv.appendChild(modelKvRow("Cleared for the longitudinal pass", fmtNum(verdicts.verified)));
      vkv.appendChild(modelKvRow("Agreed with the field call", fmtNum(verdicts.agree)));
      vkv.appendChild(modelKvRow("Overruled the field call", fmtNum(verdicts.disagree)));
      vkv.appendChild(modelKvRow("No field label to compare", fmtNum(verdicts.not_comparable)));
      vkv.appendChild(modelKvRow("Frames scored", fmtNum(verdicts.frames_scored)));
      vkv.appendChild(modelKvRow("Frame agreement", verdicts.frame_agreement_fraction === null ||
        verdicts.frame_agreement_fraction === undefined ? "not applicable" : fmtPct(verdicts.frame_agreement_fraction)));
      vkv.appendChild(modelKvRow("Mean verifier confidence", fmtConfidence(verdicts.mean_verifier_confidence)));
      verBlock.appendChild(vkv);
    }
    wrap.appendChild(verBlock);

    // What the field station's deterministic checks did with each record.
    var qc = audit.qc || {};
    var qcBlock = el("div", { class: "info-block" });
    qcBlock.appendChild(el("div", { class: "card-title", text: "Quality control" }));
    qcBlock.appendChild(el("p", { class: "settings-desc", text:
      "What the field station's deterministic checks did with each record. A deferred record is one the station could not classify on its own and passed to the desktop, with its reason recorded." }));
    qcBlock.appendChild(el("p", { class: "card-note", text:
      "\"Passed\" means the checks finalized the record; \"pending\" means capture saved it but quality control has not run on it yet. Use \"Finalize quality control\" in the Run now controls above to clear pending records now." }));
    var states = countsToBars(qc.by_state);
    if (states.length) { qcBlock.appendChild(barChart(states, { title: "events by quality state" })); }
    var reasons = countsToBars(qc.by_reason);
    if (reasons.length) {
      qcBlock.appendChild(el("p", { class: "card-note", text: "Why records were deferred:" }));
      qcBlock.appendChild(barChart(reasons, { title: "deferral reasons" }));
    } else {
      qcBlock.appendChild(el("p", { class: "card-note", text: "No record has been deferred." }));
    }
    wrap.appendChild(qcBlock);

    // How sure the models were, kept separate by modality.
    var confBlock = el("div", { class: "info-block" });
    confBlock.appendChild(el("div", { class: "card-title", text: "Detection confidence" }));
    confBlock.appendChild(el("p", { class: "settings-desc", text:
      "How sure the models were about each recognised detection, kept separate by modality because a camera's score and an acoustic score are not the same measurement." }));
    var ckv = el("div", { class: "kv-list" });
    ["vision", "audio"].forEach(function (modality) {
      var s = byModality[modality] || { n: 0 };
      ckv.appendChild(modelKvRow(humanize(modality) + " detections", s.n
        ? fmtNum(s.n) + " scored, mean " + fmtConfidence(s.mean) +
          ", range " + fmtConfidence(s.min) + " to " + fmtConfidence(s.max)
        : "none recorded"));
    });
    confBlock.appendChild(ckv);
    var buckets = countsToBars(confidence.buckets, true);
    if (buckets.length) { confBlock.appendChild(barChart(buckets, { title: "confidence distribution" })); }
    wrap.appendChild(confBlock);

    // Events per month with that month's mean confidence, so drift is visible.
    var trend = audit.trend || [];
    if (trend.length) {
      var tBlock = el("div", { class: "info-block" });
      tBlock.appendChild(el("div", { class: "card-title", text: "Over time" }));
      tBlock.appendChild(el("p", { class: "settings-desc", text:
        "Events per month with that month's mean detection confidence, so a drift in how certain the models are over a deployment is visible rather than hidden inside one overall average." }));
      var tkv = el("div", { class: "kv-list" });
      trend.forEach(function (row) {
        tkv.appendChild(modelKvRow(row.period, fmtNum(row.events) + " events, mean confidence " +
          fmtConfidence(row.mean_confidence) + ", " + fmtNum(row.verified) + " verified"));
      });
      tBlock.appendChild(tkv);
      wrap.appendChild(tBlock);
    }

    // What produced each row, which is what makes a result reproducible later.
    // Two different kinds of provenance are shown apart, because they are set in
    // different ways: model VERSIONS a person types in Settings, and reference
    // DATES that the fetch stamps automatically. Mixing them read as if a date
    // were a version, which was confusing.
    var versions = audit.versions || {};
    function versionRows(host, rows) {
      var kv = el("div", { class: "kv-list" });
      rows.forEach(function (pair) {
        var counts = versions[pair[0]] || {};
        var parts = Object.keys(counts).map(function (k) { return k + " (" + fmtNum(counts[k]) + ")"; });
        kv.appendChild(modelKvRow(pair[1], parts.length ? parts.join(", ") : "none recorded"));
      });
      host.appendChild(kv);
    }

    var verBlock2 = el("div", { class: "info-block" });
    verBlock2.appendChild(el("div", { class: "card-title", text: "Provenance of the stored rows" }));
    verBlock2.appendChild(el("p", { class: "settings-desc", text:
      "What produced each stored row, so a result can be reproduced later. Two different kinds are shown separately: the model versions you set, and the dates the offline reference data was fetched." }));

    verBlock2.appendChild(el("div", { class: "card-meta", text: "Model versions" }));
    verBlock2.appendChild(el("p", { class: "form-hint", text:
      "The version of each model that produced the row. You set these per model in Settings; \"not stated\" means the version was blank when the row was produced. Worth filling in before a deployment you intend to publish." }));
    versionRows(verBlock2, [
      ["screening_model_version", "Field screening model"],
      ["acoustic_model_version", "Acoustic model"],
      ["rfdetr_version", "Desktop verifier"]
    ]);

    verBlock2.appendChild(el("div", { class: "card-meta", text: "Reference-data fetch dates" }));
    verBlock2.appendChild(el("p", { class: "form-hint", text:
      "These are not versions you type. Each is the date the GBIF or IUCN reference data was current when it was fetched, stamped so a result stays reproducible against that data vintage. \"not stated\" means the record was captured before that data was fetched, or its species name did not match GBIF. Run Species data, Fetch reference data (and correct any misspelled names) to fill them; they update on their own." }));
    versionRows(verBlock2, [
      ["gbif_snapshot_date", "GBIF backbone snapshot date"],
      ["iucn_fetch_date", "IUCN status fetch date"]
    ]);
    wrap.appendChild(verBlock2);

    if (audit.note) { wrap.appendChild(el("p", { class: "card-note", text: audit.note })); }
    return wrap;
  }

  // Per-species model accuracy: the inferred model-quality layer. Every value is
  // derived from expert reviews and is tagged inference, keyed to a model
  // version, and never written into the measured record. A low accuracy for a
  // species is a fine-tuning target, so the table is sorted with the weakest
  // species first, right beside the retraining export that acts on them.
  function renderModelTrust(host, mt) {
    clear(host);
    var block = el("div", { class: "info-block" });
    var head = el("div", { class: "card-title" }, [
      el("span", { text: "Model accuracy by species" }),
      badge("inferred", "source")
    ]);
    block.appendChild(head);
    block.appendChild(el("p", { class: "settings-desc", text:
      "How often each model was right about a species, judged by your own expert reviews. This is inferred, not measured: it is computed from confirmed, relabelled, and rejected detections and is never written back onto the record, and it changes no salience value or pass result. A low accuracy means the model is weak on that species, so the weakest species are listed first as fine-tuning targets. Every figure is tied to the model version that produced the call." }));

    if (!mt || mt.empty) {
      block.appendChild(el("p", { class: "card-note", text: (mt && mt.reason) ||
        "No expert reviews yet, so no model accuracy can be computed. Confirm, relabel, or reject detections on Detections or Audio and this fills in." }));
      host.appendChild(block);
      return;
    }

    if (mt.scope) {
      block.appendChild(el("p", { class: "form-hint", text: "Scope: " + mt.scope + "." }));
    }

    // Group the species rows by the model that produced them, so each model's
    // rollups sit above only its own species. The rows arrive already sorted with
    // the lowest accuracy first, and grouping preserves that order within a model.
    var groups = {};
    var order = [];
    (mt.species || []).forEach(function (r) {
      var key = r.model_version || "";
      if (!groups[key]) { groups[key] = []; order.push(key); }
      groups[key].push(r);
    });

    order.forEach(function (key) {
      var rollup = (mt.models || {})[key] || {};
      var modelLabel = rollup.model_version || key || "version not recorded";
      var modelBlock = el("div", { class: "trust-model" });
      modelBlock.appendChild(el("div", { class: "card-meta", text: "Model: " + modelLabel }));
      modelBlock.appendChild(metricRow([
        metricCard("Overall (micro)", rollup.micro == null ? "-" : fmtPct(rollup.micro)),
        metricCard("Per-species (macro)", rollup.macro == null ? "-" : fmtPct(rollup.macro)),
        metricCard("Species reviewed", fmtNum(rollup.species)),
        metricCard("Reviews", fmtNum(rollup.reviewed))
      ]));
      modelBlock.appendChild(el("p", { class: "form-hint", text:
        "Overall weights every reviewed detection equally; per-species averages the species so a weakness on a rarely-seen one is not hidden by a common one. A gap between them points to an uneven model." }));

      var table = el("table", { class: "trust-table" });
      var thead = el("thead");
      thead.appendChild(el("tr", {}, [
        el("th", { text: "Species" }),
        el("th", { class: "num", text: "Reviews" }),
        el("th", { class: "num", text: "Confirmed" }),
        el("th", { class: "num", text: "Relabelled" }),
        el("th", { class: "num", text: "Rejected" }),
        el("th", { class: "num", text: "Accuracy" }),
        el("th", { class: "num", text: "Cautious low" }),
        el("th", { text: "Confused with" })
      ]));
      table.appendChild(thead);
      var tbody = el("tbody");
      groups[key].forEach(function (r) {
        // Small-n rows are shown but de-emphasized: the prior tempers them, yet a
        // handful of reviews is weaker evidence than many, and the eye should know.
        var smallN = (r.reviewed || 0) < 5;
        var confused = Object.keys(r.confused_with || {})
          .map(function (name) { return name + " (" + r.confused_with[name] + ")"; })
          .join(", ") || "-";
        tbody.appendChild(el("tr", { class: smallN ? "muted-row" : "" }, [
          el("td", { text: r.species_label || r.species_key }),
          el("td", { class: "num", text: fmtNum(r.reviewed) }),
          el("td", { class: "num", text: fmtNum(r.confirms) }),
          el("td", { class: "num", text: fmtNum(r.relabels) }),
          el("td", { class: "num", text: fmtNum(r.rejects) }),
          el("td", { class: "num", text: r.accuracy == null ? "-" : fmtPct(r.accuracy) }),
          el("td", { class: "num", text: r.wilson_lower == null ? "-" : fmtPct(r.wilson_lower) }),
          el("td", { text: confused })
        ]));
      });
      table.appendChild(tbody);
      modelBlock.appendChild(table);
      block.appendChild(modelBlock);
    });

    block.appendChild(el("p", { class: "form-hint", text:
      "\"Accuracy\" is a Laplace-smoothed precision, so a single review reads as neither 0 nor 100 percent; \"Cautious low\" is a conservative lower bound for the same species. The species at the top are the fine-tuning targets: export the weak and disputed cases below and retrain to raise them, keyed to a model version so the improvement is provable." }));
    host.appendChild(block);
  }

  // Turn a {label: count} map into sorted bar-chart rows. Fixed-order maps such
  // as a confidence histogram keep their natural order instead of being sorted
  // by size, because the order is the meaning.
  function countsToBars(counts, keepOrder) {
    var rows = Object.keys(counts || {}).map(function (k) {
      return { label: keepOrder ? k : humanize(k), value: counts[k] };
    });
    return keepOrder ? rows : rows.sort(function (a, b) { return b.value - a.value; });
  }

  function meanAcrossModalities(byModality) {
    var sum = 0, n = 0;
    Object.keys(byModality || {}).forEach(function (m) {
      var s = byModality[m];
      if (s && s.n) { sum += s.mean * s.n; n += s.n; }
    });
    return n ? sum / n : null;
  }

  // On-demand controls for the stages the capture-time scheduler would otherwise
  // run: the longitudinal pass, quality control, verification, and reports. The
  // scheduler only advances while capture is running and counts elapsed
  // capture-thread time, so a desktop that captures in short bursts never reaches
  // a weekly cadence; these run each stage now and report what it did. The scope
  // follows the station filter, so a control run while viewing one station acts on
  // that station, and one run while viewing all stations acts on all of them.
  function renderRunNowControls(host, onAfterRun) {
    clear(host);
    var block = el("div", { class: "info-block" });
    block.appendChild(el("p", { class: "settings-desc", text:
      "Run a processing stage now instead of waiting for the capture-time schedule. Each reports what it did, or that nothing was eligible. The longitudinal pass reads the whole verified record; the others follow the station filter above." }));
    var row = el("div", { class: "control-row" });
    var result = el("p", { class: "card-note", text: "" });

    function scopeQuery() {
      return state.stationId ? query({ station_id: state.stationId }) : "";
    }

    function runControl(label, path, describe, body) {
      var btn = el("button", { type: "button", class: "btn", text: label });
      btn.addEventListener("click", function () {
        var buttons = row.querySelectorAll("button");
        Array.prototype.forEach.call(buttons, function (b) { b.disabled = true; });
        result.textContent = label + ": working...";
        function reenable() { Array.prototype.forEach.call(buttons, function (b) { b.disabled = false; }); }
        apiSend(path + scopeQuery(), "POST", body || {}).then(function (r) {
          // Write the outcome first, then refresh only the data regions. The
          // refresh no longer rebuilds this control block, so the message stays
          // on screen instead of being wiped by a full panel rebuild.
          result.textContent = describe(r) || (r && r.note) || (label + ": done.");
          if (typeof onAfterRun === "function") {
            var done = onAfterRun();
            if (done && typeof done.then === "function") { done.then(reenable, reenable); }
            else { reenable(); }
          } else {
            reenable();
          }
        }).catch(function (e) {
          reenable();
          result.textContent = label + " could not run: " + e.message;
        });
      });
      return btn;
    }

    row.appendChild(runControl("Run longitudinal pass", "/dream/run", function (r) {
      var msg = "Longitudinal pass " + (r.status || "ran") + ": " +
        fmtNum(r.observations_consolidated) + " consolidated, " +
        fmtNum(r.salience_scored) + " scored, " +
        fmtNum(r.patterns_emitted) + " candidate pattern(s). " +
        (r.narration_available ? "" : "Language model unavailable, so no narration. ") +
        "Candidate patterns are hypotheses, never findings.";
      // When the pass found nothing, say plainly why, from the counted record.
      if (r.patterns_emitted === 0 && r.diagnostics && r.diagnostics.reason) {
        msg += " " + r.diagnostics.reason;
      }
      return msg;
    }));
    row.appendChild(runControl("Finalize quality control", "/qc/run", function (r) {
      return r.finalized ? "Finalized quality control on " + fmtNum(r.finalized) + " record(s)."
        : "No record was awaiting quality control.";
    }));
    row.appendChild(runControl("Run verification", "/verify/run", function (r) {
      return r.verified ? "Re-scored and gated " + fmtNum(r.verified) + " observation(s)."
        : "No eligible observation, or none in the desktop verifier's target group.";
    }));
    row.appendChild(runControl("Generate report (PDF)", "/reports/run", function (r) {
      var made = (r.formats || []).join(", ");
      return made ? "Wrote a " + made + " report to the reports folder (bundle " + (r.bundle || "") + "). See the Reports tab."
        : "No report output was produced.";
    }, { formats: ["pdf"] }));

    block.appendChild(row);
    block.appendChild(result);
    host.appendChild(block);
  }

  // The two guided species-data setup steps: building the taxonomic index that
  // relabelling searches, and fetching per-species GBIF and IUCN reference data.
  // Each runs in the background on the desktop; this reads its status and, while a
  // job runs, repolls so progress updates without a manual reload. The steps
  // replace the command-line scripts so a non-programmer never needs a terminal.
  function renderSpeciesData(host) {
    clear(host);
    var block = el("div", { class: "info-block" });
    block.appendChild(el("p", { class: "settings-desc", text:
      "Prepare the species data Audtheia uses offline at run time. Each step runs in the background and reports progress here; you can leave this page while one runs." }));
    var indexHost = el("div");
    var targetHost = el("div", { class: "species-step" });
    var refHost = el("div", { class: "species-step" });
    block.appendChild(indexHost);
    block.appendChild(targetHost);
    block.appendChild(refHost);
    host.appendChild(block);

    function stillHere(node) { return state.activePanel === "brain" && document.body.contains(node); }

    // Poll a config push to the Pi until it finishes, then report the outcome.
    // A push reuses the provisioning status endpoint, so the states are the same
    // ones the connect flow reports.
    function pollPush(sid, btn, msg) {
      apiGet("/stations/" + encodeURIComponent(sid) + "/provision/status").then(function (s) {
        if (s.state === "running") {
          if (stillHere(msg)) { window.setTimeout(function () { pollPush(sid, btn, msg); }, 2000); }
          return;
        }
        btn.disabled = false;
        if (s.state === "succeeded") {
          msg.textContent = "Configuration pushed to the Pi; it applies on the station's next start.";
        } else {
          msg.textContent = "The push did not finish (" + (s.state || "unknown") + "). It is safe to try again.";
        }
      }).catch(function (e) { btn.disabled = false; msg.textContent = "Could not read push status: " + e.message; });
    }

    // The target-species editor: add or remove the species each station is
    // looking for. The reference fetch covers these and the field model is
    // trained on them, so this is where a person declares them without editing a
    // configuration file by hand.
    function paintTargets() {
      apiGet("/settings").then(function (r) {
        clear(targetHost);
        var stations = ((r.config || {}).stations) || [];
        targetHost.appendChild(el("div", { class: "card-title", text: "Target species (what each station is looking for)" }));
        targetHost.appendChild(el("p", { class: "form-hint", text:
          "The species a station targets. The reference fetch covers these, and a field model is trained on them. Use the correct scientific name so it matches GBIF." }));
        if (!stations.length) { targetHost.appendChild(el("p", { class: "card-note", text: "No stations configured yet." })); return; }
        stations.forEach(function (st) {
          var sid = st.station_id;
          var targets = st.target_species || [];
          // Each station is a foldable, softly shadowed card, so a long list of
          // stations stays compact and scannable. It opens by default when it
          // has no targets yet, so a first-time setup is not hidden.
          var summaryText = (st.station_name || sid) + "  (" +
            (targets.length ? fmtNum(targets.length) + (targets.length === 1 ? " target species" : " target species") : "no targets yet") + ")";
          var body = el("div", { class: "fold-body" });

          var chips = el("div", { class: "badge-row" });
          if (!targets.length) { chips.appendChild(el("span", { class: "card-note", text: "none yet" })); }
          targets.forEach(function (name) {
            var chip = el("button", { type: "button", class: "btn btn-small", text: name + "  ×", title: "Remove " + name });
            chip.addEventListener("click", function () {
              chip.disabled = true;
              apiSend("/settings/stations/" + encodeURIComponent(sid) + "/target-species/" + encodeURIComponent(name), "DELETE")
                .then(function () { paintTargets(); paintRef(); })
                .catch(function (e) { chip.disabled = false; window.alert("Could not remove: " + e.message); });
            });
            chips.appendChild(chip);
          });
          body.appendChild(chips);

          var input = el("input", { type: "text", class: "form-input", placeholder: "scientific name, for example Aplysina fistularis" });
          var add = el("button", { type: "button", class: "btn btn-small", text: "Add species" });
          function doAdd() {
            var name = input.value.trim();
            if (!name) { return; }
            add.disabled = true;
            apiSend("/settings/stations/" + encodeURIComponent(sid) + "/target-species", "POST", { name: name })
              .then(function () { input.value = ""; add.disabled = false; paintTargets(); paintRef(); })
              .catch(function (e) { add.disabled = false; window.alert("Could not add: " + e.message); });
          }
          add.addEventListener("click", doAdd);
          input.addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); doAdd(); } });
          body.appendChild(el("div", { class: "control-row" }, [input, add]));

          // Push the current configuration down to this station's Pi. A config
          // edit reaches a field station only when it is pushed, so this sends
          // the updated settings over the station's already-authorized key.
          var pushMsg = el("p", { class: "form-hint", text:
            "Editing here updates the desktop. For a Pi field station, push the change down so the Pi applies it on its next start. A station whose Pi has not been connected yet will say so." });
          var push = el("button", { type: "button", class: "btn btn-small", text: "Push changes to Pi" });
          push.addEventListener("click", function () {
            push.disabled = true;
            pushMsg.textContent = "Pushing the configuration to the Pi.";
            apiSend("/stations/" + encodeURIComponent(sid) + "/push-config", "POST", {})
              .then(function () { pollPush(sid, push, pushMsg); })
              .catch(function (e) { push.disabled = false; pushMsg.textContent = e.message; });
          });
          body.appendChild(el("div", { class: "control-row" }, [push]));
          body.appendChild(pushMsg);

          // Cards start folded, so a long station list opens as a compact,
          // scannable set of one-line summaries; a person opens the one they
          // want. The open state is not stored, so nothing is persisted.
          var card = el("details", { class: "fold-card" }, [
            el("summary", { text: summaryText }),
            body
          ]);
          targetHost.appendChild(card);
        });
      }).catch(function (e) { clear(targetHost); targetHost.appendChild(el("p", { class: "card-note", text: "Could not read target species: " + e.message })); });
    }

    function paintIndex() {
      apiGet("/species/index/status").then(function (s) {
        clear(indexHost);
        var job = s.job || {};
        indexHost.appendChild(el("div", { class: "card-title", text: "Taxonomic index (species search for relabelling)" }));
        var line;
        if (s.index_present) {
          line = "Built: " + (s.index_names != null ? fmtNum(s.index_names) + " names" : "present") +
            ". Relabelling a detection can search species.";
        } else if (!s.backbone_present) {
          line = "Not built, and the backbone file is missing. Fetch the backbone during setup, then build the index.";
        } else {
          line = "Not built. Relabelling to a corrected species needs this; confirm, reject and per-frame review already work without it.";
        }
        indexHost.appendChild(el("p", { class: "card-note", text: line }));
        if (job.status === "running") {
          var pct = job.progress != null ? Math.round(job.progress * 100) + "%  " : "";
          indexHost.appendChild(el("p", { class: "card-note", text: "Building: " + pct + (job.message || "working") }));
        } else {
          if (job.status === "done" && job.result) {
            indexHost.appendChild(el("p", { class: "card-note", text: "Last build indexed " + fmtNum(job.result.names) + " names." }));
          } else if (job.status === "error") {
            indexHost.appendChild(el("p", { class: "card-note", text: "Last build did not finish: " + job.error }));
          }
          if (s.backbone_present) {
            var label = s.index_present ? "Rebuild index" : "Build index";
            var btn = el("button", { type: "button", class: "btn", text: label });
            btn.addEventListener("click", function () {
              btn.disabled = true;
              apiSend("/species/index/build" + (s.index_present ? query({ force: true }) : ""), "POST", {})
                .then(function () { paintIndex(); })
                .catch(function (e) { btn.disabled = false; indexHost.appendChild(el("p", { class: "card-note", text: e.message })); });
            });
            indexHost.appendChild(btn);
            indexHost.appendChild(el("p", { class: "form-hint", text:
              "Building reads the whole backbone once and takes several minutes." }));
          }
        }
        if (job.status === "running" && stillHere(indexHost)) { window.setTimeout(paintIndex, 3000); }
      }).catch(function (e) {
        clear(indexHost);
        indexHost.appendChild(el("p", { class: "card-note", text: "Could not read index status: " + e.message }));
      });
    }

    function paintRef() {
      apiGet("/species/reference/status").then(function (s) {
        clear(refHost);
        var job = s.job || {};
        var targets = s.target_species || [];
        var detected = s.detected_species || [];
        var toFetch = targets.length + detected.length;
        refHost.appendChild(el("div", { class: "card-title", text: "Reference data (names, occurrence, conservation status)" }));
        refHost.appendChild(el("p", { class: "card-note", text:
          fmtNum(s.references_stored) + " species on file. " + targets.length + " target species configured, and " +
          detected.length + " species already detected in your record. The fetch covers both." }));
        refHost.appendChild(el("p", { class: "form-hint", text:
          "GBIF naming and the global occurrence count need no account. The IUCN token only adds the Red List conservation status: " +
          (s.iucn_token_present
            ? "a token is set, so status will be fetched."
            : "no token is set, so status stays blank until you add one in Settings.") }));
        if (job.status === "running") {
          refHost.appendChild(el("p", { class: "card-note", text: "Fetching: " + (job.message || "working") }));
        } else {
          if (job.status === "done" && job.result) {
            var r = job.result;
            refHost.appendChild(el("p", { class: "card-note", text:
              "Last fetch: " + fmtNum(r.fetched) + " stored, " + fmtNum(r.cached) + " already on file, " +
              fmtNum(r.unmatched) + " not matched by GBIF, " + fmtNum(r.failed) + " failed." +
              (r.stamped_existing ? "  " + fmtNum(r.stamped_existing) + " existing record(s) stamped with snapshot dates." : "") }));
          } else if (job.status === "error") {
            refHost.appendChild(el("p", { class: "card-note", text: "Last fetch did not finish: " + job.error }));
          }
          var btn = el("button", { type: "button", class: "btn", text: "Fetch reference data" });
          if (!toFetch) { btn.disabled = true; }
          btn.addEventListener("click", function () {
            btn.disabled = true;
            apiSend("/species/reference/fetch", "POST", {})
              .then(function () { paintRef(); })
              .catch(function (e) { btn.disabled = false; refHost.appendChild(el("p", { class: "card-note", text: e.message })); });
          });
          refHost.appendChild(btn);
          refHost.appendChild(el("p", { class: "form-hint", text: toFetch
            ? "This reaches out to GBIF and IUCN once per species and needs an internet connection; it is the one online step. It covers your configured target species and the species already in your record, and stamps snapshot dates on matching records. A misspelled name will not match GBIF, so correct labels first for those to fill."
            : "Nothing to fetch yet: add target species to a station in Settings, or capture some detections, then the fetch covers them." }));
        }
        if (job.status === "running" && stillHere(refHost)) { window.setTimeout(paintRef, 3000); }
      }).catch(function (e) {
        clear(refHost);
        refHost.appendChild(el("p", { class: "card-note", text: "Could not read reference status: " + e.message }));
      });
    }

    paintIndex();
    paintTargets();
    paintRef();
  }

  // Keep the longitudinal pass current without rebuilding the whole panel: only
  // this block is redrawn, and only while a pass is actually running, so the page
  // never jumps or flickers under someone who is reading it.
  function paintDreamStatus(hostEl, status) {
    clear(hostEl);
    hostEl.appendChild(dreamStatusView(status));
    if (state.dreamTimer) { window.clearTimeout(state.dreamTimer); state.dreamTimer = null; }
    var active = status && status.active;
    if (!active || active.status !== "running") { return; }
    state.dreamTimer = window.setTimeout(function () {
      if (state.activePanel !== "brain" || !document.body.contains(hostEl)) { return; }
      apiGet("/dream/status")
        .then(function (next) { paintDreamStatus(hostEl, next); })
        .catch(function () { /* a missed poll simply leaves the last status shown */ });
    }, POLL_INTERVAL_MS);
  }

  function dreamStatusView(status) {
    var active = status.active;
    var wrap = el("div", { class: "dream-status" });
    if (!active) {
      var passes = status.passes || [];
      if (!passes.length) {
        wrap.appendChild(el("p", { class: "card-note", text: "No longitudinal pass has run yet." }));
        return wrap;
      }
      // Summarize the most recent pass with its real figures, so the box reports
      // what the last run did rather than only a phase word.
      var p = passes[0];
      wrap.appendChild(el("div", { class: "card-stats", text:
        "Last pass: " + (p.status || "unknown") +
        "   phase " + (p.phase_reached || "unknown") +
        "   " + fmtNum(p.cycles_completed) + " cycle(s)" +
        "   " + fmtNum(p.work_budget_consumed) + " record(s) consolidated" }));
      wrap.appendChild(el("p", { class: "card-meta", text: "ran " + fmtTime(p.ended_at || p.started_at) }));
      wrap.appendChild(el("p", { class: "form-hint", text:
        "This box reports the run itself. Any hypotheses the pass proposed appear below under Candidate patterns, and the site baselines it builds appear under Models and Memory, Site memory. With few events a pass consolidates the record and scores salience but proposes no statistical patterns yet; those need enough events to clear the evidence thresholds." }));
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

  // Starter skills a person can insert and then edit. Each one is already on the
  // correct tier, so the collection teaches the field-versus-desktop rule by
  // example rather than by asking someone to read it first. Every field skill
  // below is a pure function of a value measured in the record it is looking at;
  // every desktop skill reasons about ecology or across records, and so is
  // recorded as labelled inference.
  var SKILL_TEMPLATES = [
    {
      title: "Low-confidence detection review",
      trigger_condition: "The screening model's confidence for a detection is below 0.45.",
      instruction: "Flag the observation as low confidence so it is reviewed before the record is relied on, and so its frames become candidates for retraining.",
      tier: "deterministic_flag",
      why: "Catches shaky calls, including organisms the model was never trained on."
    },
    {
      title: "Thermal stress watch",
      trigger_condition: "Water temperature is above 29 degrees Celsius.",
      instruction: "Flag the observation as a possible thermal stress event.",
      tier: "deterministic_flag",
      why: "A measured threshold on a sensor channel, so the station can decide it alone."
    },
    {
      title: "Unusually long event",
      trigger_condition: "The event lasted longer than 120 seconds.",
      instruction: "Flag the observation for behavioural review.",
      tier: "deterministic_flag",
      why: "Long events often hold behaviour worth watching rather than a simple pass-through."
    },
    {
      title: "Night-time activity",
      trigger_condition: "The event's timestamp falls between 20:00 and 04:00 UTC.",
      instruction: "Flag the observation as a night-time detection.",
      tier: "deterministic_flag",
      why: "Separates nocturnal activity without asserting anything about why."
    },
    {
      title: "Possible new arrival",
      trigger_condition: "A taxon is recorded that does not appear anywhere in this station's first 30 days of record.",
      instruction: "Note it as a possible new arrival at this site, and state what it would take to confirm that.",
      tier: "interpretive",
      why: "Needs memory of other records, so it belongs on the desktop."
    },
    {
      title: "Off-target taxon",
      trigger_condition: "A detection resolves to a taxon that is not in this station's target species list.",
      instruction: "Note that the identification falls outside what the model was trained for, and that it may be a misclassification worth checking.",
      tier: "interpretive",
      why: "The likeliest way a misclassified or unknown organism surfaces for a human."
    },
    {
      title: "Substrate competition",
      trigger_condition: "A sponge is detected in the same event as coral.",
      instruction: "Note the potential competition for substrate between them.",
      tier: "interpretive",
      why: "An ecological reading, so it is stored as inference and never as measurement."
    },
    {
      title: "Heard but never seen",
      trigger_condition: "A taxon has acoustic detections at a site but no visual detections there.",
      instruction: "Note the gap between the two modalities and suggest whether camera placement or timing could explain it.",
      tier: "interpretive",
      why: "Compares across records and modalities, which only the desktop can do."
    }
  ];

  // What a skill is, and what it honestly cannot do. Folded away so it is there
  // for a first-time reader without standing in the way afterwards.
  function skillsGuidance() {
    var box = el("details", { class: "info-block skills-guidance" }, [
      el("summary", { text: "What a skill is, and what it cannot do" })
    ]);
    box.appendChild(el("p", { class: "settings-desc", text:
      "A skill is a reusable rule you write once and Audtheia applies to every observation. It has a trigger (when to use it) and an instruction (what to do), and it runs in one of two places." }));

    var tiers = el("div", { class: "kv-list" });
    tiers.appendChild(modelKvRow("Field, deterministic flag",
      "Runs on the station. May use only values measured in the record in front of it: confidence, sensor readings, duration, timestamps, counts. Its output is a measured flag."));
    tiers.appendChild(modelKvRow("Desktop, interpretive",
      "Runs on your computer. May reason about ecology and across many records. Everything it writes is stored as labelled inference, never as measurement."));
    box.appendChild(tiers);

    box.appendChild(el("p", { class: "form-label", text: "Limits worth knowing before you write one" }));
    var limits = el("ul", { class: "skills-limits" });
    [
      "A skill cannot make a model recognise a species it was never trained on. To teach it something new, flag the examples with a skill and then retrain on them.",
      "A field skill has no memory of other observations and no ecological knowledge. If your rule needs either, it belongs on the desktop.",
      "Skills annotate and flag. They never change what is captured, and they never delete anything.",
      "No skill can reach the internet. Audtheia runs offline by design.",
      "Write triggers that are specific and testable. \"When something looks unusual\" cannot be evaluated; \"when confidence is below 0.45\" can."
    ].forEach(function (line) { limits.appendChild(el("li", { text: line })); });
    box.appendChild(limits);
    return box;
  }

  // The starter collection, offered as cards that open a prefilled form.
  function showSkillTemplates(host, tier) {
    skillFormOpen = true;
    clear(host);
    host.appendChild(el("p", { class: "settings-desc", text:
      "Pick a starting point, then edit it to match your site. Each one is already set to the tier it must run on." }));
    var grid = el("div", { class: "card-grid" });
    SKILL_TEMPLATES.forEach(function (t) {
      grid.appendChild(el("article", { class: "card" }, [
        el("div", { class: "badge-row" }, [
          badge(t.tier === "interpretive" ? "desktop" : "field", "source"),
          badge(t.tier === "interpretive" ? "interpretive" : "deterministic flag", "status")
        ]),
        el("div", { class: "card-title", text: t.title }),
        el("div", { class: "card-stats", text: "When: " + t.trigger_condition }),
        el("div", { class: "card-stats", text: "Do: " + t.instruction }),
        el("div", { class: "card-meta", text: t.why }),
        el("div", { class: "card-actions" }, [
          el("button", {
            type: "button", class: "btn btn-small btn-primary", text: "Use this",
            // No id, so the form opens as a new skill with these values filled in.
            onclick: function () { showSkillForm(host, tier, { title: t.title, trigger_condition: t.trigger_condition, instruction: t.instruction, tier: t.tier }); }
          })
        ])
      ]));
    });
    host.appendChild(grid);
    host.appendChild(el("div", { class: "form-actions" }, [
      el("button", { type: "button", class: "btn", text: "Back to skills", onclick: function () { renderSkills(host, tier); } })
    ]));
  }

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
      filters.appendChild(el("button", {
        type: "button", class: "btn", text: "Start from an example",
        onclick: function () { showSkillTemplates(host, sel.value); }
      }));
    }
    // A background refresh must not close a form the person is filling in.
    if (!skillFormOpen) { renderSkills(host, ""); }
  }

  // The narrow condition vocabulary the field engine compiles, mirrored here so
  // the builder can only offer choices the engine accepts. Keeping it in one
  // place means the interface and the engine never drift apart.
  var CONDITION_SOURCES = [
    ["observation", "An observation value"],
    ["detection", "A detection confidence"],
    ["channel", "A sensor channel reading"],
    ["time", "The time of day"]
  ];
  var CONDITION_FIELDS = {
    observation: [
      ["screening_confidence", "screening confidence"],
      ["duration", "event duration (seconds)"],
      ["frame_count", "number of frames"],
      ["audio_true_duration_seconds", "audio duration (seconds)"],
      ["salience_provisional", "provisional salience"]
    ],
    detection: [["confidence", "confidence"]],
    time: [["hour_utc", "hour of day (UTC, 0 to 23)"]]
  };
  var CONDITION_OPS = [
    ["lt", "is below"], ["lte", "is at most"], ["gt", "is above"],
    ["gte", "is at least"], ["between", "is between"], ["outside", "is outside"]
  ];
  var CONDITION_AGGREGATES = [["max", "the strongest"], ["min", "the weakest"]];

  function opLabel(op) {
    for (var i = 0; i < CONDITION_OPS.length; i += 1) { if (CONDITION_OPS[i][0] === op) { return CONDITION_OPS[i][1]; } }
    return op;
  }

  // A stored condition rendered in plain words, so a saved skill reads as a rule
  // rather than as raw JSON.
  function describeCondition(cond) {
    if (!cond) { return null; }
    var subject;
    if (cond.source === "observation") { subject = "the observation's " + labelFor(CONDITION_FIELDS.observation, cond.field); }
    else if (cond.source === "detection") { subject = "the " + labelFor(CONDITION_AGGREGATES, cond.aggregate || "max") + " detection " + labelFor(CONDITION_FIELDS.detection, cond.field); }
    else if (cond.source === "channel") { subject = "the " + cond.field + " channel reading"; }
    else if (cond.source === "time") { subject = "the hour of day (UTC)"; }
    else { subject = cond.field; }
    var value = Array.isArray(cond.value) ? cond.value.join(" and ") : cond.value;
    return subject + " " + opLabel(cond.op) + " " + value;
  }

  function labelFor(pairs, key) {
    for (var i = 0; i < pairs.length; i += 1) { if (pairs[i][0] === key) { return pairs[i][1]; } }
    return key;
  }

  function parseStoredCondition(raw) {
    if (!raw) { return null; }
    if (typeof raw === "object") { return raw; }
    try { return JSON.parse(raw); } catch (e) { return null; }
  }

  function renderSkills(host, tier) {
    // Showing the list means no form is open, so the background refresh may
    // repaint the panel again.
    skillFormOpen = false;
    clear(host);
    host.appendChild(el("p", { class: "empty-state", text: "Loading skills." }));
    apiGet("/brain/skills" + query({ tier: tier })).then(function (skills) {
      clear(host);
      host.appendChild(skillsGuidance());
      host.appendChild(el("p", { class: "settings-desc", text:
        "A field skill runs on the station during quality control: when its condition holds for an event, it records a flag that stands beside the measurement. New captures are flagged automatically; use Apply skills to existing records to run the current field skills over events already captured. An interpretive skill runs on the desktop and is applied when the language model is available; its output is recorded as labeled inference." }));
      host.appendChild(el("button", {
        type: "button", class: "btn", text: "Apply skills to existing records",
        onclick: function () { applySkills(host, tier); }
      }));

      if (!skills.length) {
        host.appendChild(el("p", { class: "empty-state", text:
          "No skills defined yet. Use Start from an example to insert one and edit it, or New skill to write your own." }));
        return;
      }
      var grid = el("div", { class: "card-grid" });
      skills.forEach(function (s) {
        var cond = parseStoredCondition(s.condition);
        var flagged = Number(s.flagged_events || 0);
        grid.appendChild(el("article", { class: "card" }, [
          el("div", { class: "badge-row" }, [
            badge(s.tier === "interpretive" ? "desktop" : "field", "source"),
            badge(s.tier, "status"),
            (s.tier === "deterministic_flag")
              ? badge(flagged === 1 ? "1 event flagged" : fmtNum(flagged) + " events flagged", flagged ? "source" : "muted")
              : null
          ]),
          el("div", { class: "card-title", text: s.title }),
          el("div", { class: "card-stats", text: "When: " + s.trigger_condition }),
          el("div", { class: "card-stats", text: "Do: " + s.instruction }),
          (s.tier === "deterministic_flag")
            ? el("div", { class: "card-meta", text: cond ? ("Runs when: " + describeCondition(cond)) : "No condition set yet, so this skill records nothing until one is added." })
            : null,
          el("div", { class: "card-actions" }, [
            el("button", { type: "button", class: "btn btn-small", text: "Edit", onclick: function () { showSkillForm(host, tier, s); } }),
            el("button", { type: "button", class: "btn btn-small", text: "Delete", onclick: function () { deleteSkill(host, tier, s); } })
          ])
        ]));
      });
      host.appendChild(grid);
    }).catch(function (e) { setState(host, "empty-state", "Could not load skills: " + e.message); });
  }

  function applySkills(host, tier) {
    var msg = el("p", { class: "form-message", text: "Applying field skills to existing records." });
    host.appendChild(msg);
    apiSend("/brain/skills/apply" + (state.stationId ? query({ station_id: state.stationId }) : ""), "POST", {})
      .then(function (r) {
        msg.textContent = "Scanned " + fmtNum(r.scanned) + " record(s); " + fmtNum(r.flags) +
          " skill flag(s) now stand on the record. Open a detection to see its flags.";
        // Refresh the counts on the cards, keeping the outcome message visible.
        window.setTimeout(function () { renderSkills(host, tier); }, 1200);
      })
      .catch(function (e) { msg.textContent = "Could not apply skills: " + e.message; });
  }

  // The create and edit form. With no existing skill it authors a new one;
  // given one it edits it in place, keeping the skill's identity.
  function showSkillForm(host, tier, existing) {
    // Mark a form as open so the background refresh does not repaint over it.
    skillFormOpen = true;
    clear(host);
    // An existing skill carries an id, so it is edited in place. A starter
    // template carries values but no id, so the same form opens prefilled and
    // creates a new skill instead of overwriting anything.
    var isEdit = !!(existing && existing.id);

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

    // The condition builder: the same narrow vocabulary the field engine
    // compiles, offered as dropdowns so a non-programmer can author a runnable
    // rule without writing JSON or touching a config file.
    var existingCond = existing ? parseStoredCondition(existing.condition) : null;

    var sourceSel = el("select", { class: "form-input", "aria-label": "Condition source" });
    CONDITION_SOURCES.forEach(function (o) {
      var opt = el("option", { value: o[0], text: o[1] });
      if (existingCond && existingCond.source === o[0]) { opt.selected = true; }
      sourceSel.appendChild(opt);
    });

    var fieldSel = el("select", { class: "form-input", "aria-label": "Condition field" });
    var channelInput = el("input", { type: "text", class: "form-input", placeholder: "channel name, for example water_temperature" });
    if (existingCond && existingCond.source === "channel") { channelInput.value = existingCond.field || ""; }

    var aggSel = el("select", { class: "form-input", "aria-label": "Which detection" });
    CONDITION_AGGREGATES.forEach(function (o) {
      var opt = el("option", { value: o[0], text: o[1] });
      if (existingCond && (existingCond.aggregate || "max") === o[0]) { opt.selected = true; }
      aggSel.appendChild(opt);
    });

    var opSel = el("select", { class: "form-input", "aria-label": "Comparison" });
    CONDITION_OPS.forEach(function (o) {
      var opt = el("option", { value: o[0], text: o[1] });
      if (existingCond && existingCond.op === o[0]) { opt.selected = true; }
      opSel.appendChild(opt);
    });

    var v1 = el("input", { type: "number", step: "any", class: "form-input", "aria-label": "Value" });
    var v2 = el("input", { type: "number", step: "any", class: "form-input", "aria-label": "Second value" });
    if (existingCond) {
      if (Array.isArray(existingCond.value)) { v1.value = existingCond.value[0]; v2.value = existingCond.value[1]; }
      else if (existingCond.value !== undefined && existingCond.value !== null) { v1.value = existingCond.value; }
    }

    var aggField = field("Which detection", aggSel);
    var fieldField = field("Field", fieldSel);
    var channelField = field("Channel name", channelInput);
    var v2Field = field("and", v2);

    function populateFields() {
      var src = sourceSel.value;
      clear(fieldSel);
      var opts = CONDITION_FIELDS[src] || [];
      opts.forEach(function (o) {
        var opt = el("option", { value: o[0], text: o[1] });
        if (existingCond && existingCond.field === o[0]) { opt.selected = true; }
        fieldSel.appendChild(opt);
      });
      fieldField.style.display = (src === "observation" || src === "detection") ? "" : "none";
      channelField.style.display = (src === "channel") ? "" : "none";
      aggField.style.display = (src === "detection") ? "" : "none";
    }

    function syncOpValues() {
      var twoValued = (opSel.value === "between" || opSel.value === "outside");
      v2Field.style.display = twoValued ? "" : "none";
    }

    sourceSel.addEventListener("change", populateFields);
    opSel.addEventListener("change", syncOpValues);

    var builder = el("div", { class: "info-block" }, [
      el("p", { class: "form-hint", text:
        "A field skill runs only from this structured condition, never from the prose above, because a station that acted on a sentence would be interpreting. Leave the value empty to save the skill without a runnable condition yet." }),
      field("When (source)", sourceSel),
      fieldField, channelField, aggField,
      field("Comparison", opSel),
      field("Value", v1),
      v2Field
    ]);

    function syncTierVisibility() {
      builder.style.display = (tierSelect.value === "deterministic_flag") ? "" : "none";
    }
    tierSelect.addEventListener("change", syncTierVisibility);

    function buildCondition() {
      if (tierSelect.value !== "deterministic_flag") { return { ok: true, value: null }; }
      var src = sourceSel.value;
      var fld = (src === "channel") ? channelInput.value.trim() : (src === "time" ? "hour_utc" : fieldSel.value);
      var primary = v1.value.trim();
      // An empty value means no runnable condition yet, which is allowed.
      if (primary === "") { return { ok: true, value: null }; }
      if (!fld) { return { ok: false, error: "Choose a field for the condition, or clear the value to save without one." }; }
      var op = opSel.value;
      var value;
      if (op === "between" || op === "outside") {
        var a = parseFloat(v1.value), b = parseFloat(v2.value);
        if (isNaN(a) || isNaN(b)) { return { ok: false, error: "A between or outside comparison needs two numbers." }; }
        value = [a, b];
      } else {
        var n = parseFloat(v1.value);
        if (isNaN(n)) { return { ok: false, error: "The value must be a number." }; }
        value = n;
      }
      var cond = { source: src, field: fld, op: op, value: value };
      if (src === "detection") { cond.aggregate = aggSel.value; }
      return { ok: true, value: cond };
    }

    var message = el("p", { class: "form-message" });
    var save = el("button", { type: "button", class: "btn btn-primary", text: isEdit ? "Save changes" : "Create skill" });
    save.addEventListener("click", function () {
      var cond = buildCondition();
      if (!cond.ok) { message.textContent = cond.error; return; }
      var body = {
        title: titleInput.value.trim(),
        trigger_condition: triggerInput.value.trim(),
        instruction: instructionInput.value.trim(),
        tier: tierSelect.value,
        condition: cond.value
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
      el("p", { class: "form-hint", text: "A skill flags and annotates; it cannot teach a model to recognise something it was never trained on, and it never alters or deletes what was captured." }),
      builder,
      message,
      el("div", { class: "form-actions" }, [save, cancel])
    ]));

    // Set the initial visibility and field lists to match the loaded values.
    populateFields();
    syncOpValues();
    syncTierVisibility();
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
      var dir = r.reports_dir || "";
      // The separator the machine uses, taken from its own path, so the on-disk
      // location reads naturally on Windows or on the Pi.
      var sep = dir.indexOf("\\") !== -1 ? "\\" : "/";
      if (dir) {
        host.appendChild(el("p", { class: "form-hint", text:
          "Reports are saved on this computer at: " + dir + ". You can open a file below, or open that folder directly." }));
      }
      if (!bundles.length) { host.appendChild(el("p", { class: "empty-state", text: "No reports generated yet." })); return; }
      bundles.forEach(function (b) {
        var files = el("div", { class: "file-row" });
        (b.files || []).forEach(function (f) {
          // Show a readable file name, but link through the full path so it opens.
          files.appendChild(el("a", { class: "file-link", href: API + "/reports/file" + query({ path: b.name + "/" + f }),
            text: (f === "report.pdf" ? "Open PDF report" : f), target: "_blank", rel: "noopener" }));
        });
        if (!(b.files || []).length) {
          files.appendChild(el("p", { class: "card-note", text:
            "This report has no downloadable files. If you asked for a PDF and it is missing, the model runtime or fpdf2 may have failed; check the language model status, or regenerate." }));
        }
        var del = el("button", { type: "button", class: "btn btn-small", text: "Delete report" });
        del.addEventListener("click", function () {
          if (!window.confirm('Delete the report "' + b.name + '"? This removes only this report bundle, not any captured data.')) { return; }
          del.disabled = true;
          apiSend("/reports/" + encodeURIComponent(b.name), "DELETE")
            .then(function () { loaders.reports(); })
            .catch(function (e) { del.disabled = false; window.alert("Could not delete the report: " + e.message); });
        });
        host.appendChild(el("article", { class: "card" }, [
          el("div", { class: "card-title", text: b.name }),
          el("div", { class: "card-meta", text: "generated: " + fmtTime(b.modified_utc) }),
          el("div", { class: "card-meta", text: "saved at: " + (dir ? dir + sep + b.name : b.name) }),
          files,
          el("div", { class: "card-actions" }, [del])
        ]));
      });
    }).catch(function (e) { setState(host, "empty-state", "Could not load reports: " + e.message); });
  };

  function reportForm() {
    var form = el("form", { class: "report-form" });
    var start = el("input", { type: "date", "aria-label": "Start date" });
    var end = el("input", { type: "date", "aria-label": "End date" });
    // Format choice: pick the polished PDF, the raw CSV data bundle, or both.
    // PDF is on by default because it is the report a person reads and shares.
    var pdfBox = el("input", { type: "checkbox", checked: true, "aria-label": "PDF report" });
    pdfBox.checked = true;
    var csvBox = el("input", { type: "checkbox", "aria-label": "CSV data bundle" });
    var submit = el("button", { type: "submit", class: "btn btn-primary", text: "Generate report" });
    var note = el("span", { class: "form-note" });
    form.appendChild(el("label", { class: "filter-field" }, [el("span", { text: "From" }), start]));
    form.appendChild(el("label", { class: "filter-field" }, [el("span", { text: "To" }), end]));
    form.appendChild(el("label", { class: "filter-field" }, [pdfBox, el("span", { text: "PDF report" })]));
    form.appendChild(el("label", { class: "filter-field" }, [csvBox, el("span", { text: "CSV data" })]));
    form.appendChild(submit);
    form.appendChild(note);
    form.appendChild(el("span", { class: "form-hint", text: "The PDF is the readable report, with a summary and charts. CSV is the raw data for your own analysis. Choose either or both." }));
    form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      var formats = [];
      if (pdfBox.checked) { formats.push("pdf"); }
      if (csvBox.checked) { formats.push("csv"); }
      if (!formats.length) { note.textContent = "Choose at least one format: PDF, CSV, or both."; return; }
      submit.disabled = true;
      note.textContent = "Requesting generation.";
      apiSend("/reports", "POST", {
        station_id: state.stationId || null,
        start: start.value || null,
        end: end.value || null,
        formats: formats
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
    "settings-stations": "The stations this computer manages, with their environment and sensors. A station may run on this computer, on field hardware, or both.",
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
      renderModelsEditor(cfg);
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
    host.appendChild(el("p", { class: "card-note", text: "Your theme is saved and used the next time you open Audtheia, on this and any other device that opens this hub." }));
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
      } else if (e.kind === "numberOrNull") {
        var ns = String(raw).trim();
        if (ns === "") {
          val = null;
          if (e.original === null || e.original === undefined || e.original === "") { continue; }
        } else {
          val = Number(ns);
          if (isNaN(val)) { msg.textContent = "A number field has an invalid value."; return null; }
          if (e.original != null && e.original !== "" && Number(e.original) === val) { continue; }
        }
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
      var card = el("details", { class: "config-card" }, [
        el("summary", { class: "station-summary" }, [
          el("span", { class: "station-summary-name", text: st.station_name || st.station_id }),
          el("span", { class: "badge-row station-summary-badges" }, [deploymentBadge(stationDeployment(st))])
        ])
      ]);
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
    var loc = st.location || {};
    if (loc.latitude !== null && loc.latitude !== undefined && loc.longitude !== null && loc.longitude !== undefined) {
      var posText = Number(loc.latitude).toFixed(4) + ", " + Number(loc.longitude).toFixed(4);
      if (loc.elevation !== null && loc.elevation !== undefined) { posText += " (" + loc.elevation + " m)"; }
      body.appendChild(fieldRow("Fixed position", posText));
    }
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
    var vidIn = el("input", { type: "text", class: "form-input", value: src.video || "",
      placeholder: "webcam:0" });
    editors.push({ scope: "station", field: "capture_source_video", station_id: sid, original: src.video || "", get: function () { return vidIn.value; }, kind: "textOrNull" });
    body.appendChild(editField("Capture source (video)", vidIn, "webcam:0, url:rtsp://..., stream:<page url>, or file:C:/clip.mp4. Leave blank for none (Optional)."));

    var audIn = el("input", { type: "text", class: "form-input", value: src.audio || "",
      placeholder: "file:C:/clip.wav" });
    editors.push({ scope: "station", field: "capture_source_audio", station_id: sid, original: src.audio || "", get: function () { return audIn.value; }, kind: "textOrNull" });
    body.appendChild(editField("Capture source (audio)", audIn, "file:C:/clip.wav, url:<direct stream>, stream:<page url>, or a plain path. Leave blank for none (Optional)."));

    var sensors = st.sensors || {};
    [["camera", "sensor_camera_enabled", "Camera"], ["audio", "sensor_audio_enabled", "Audio"], ["gps", "sensor_gps_enabled", "GPS"]].forEach(function (t) {
      var on = !!(sensors[t[0]] && sensors[t[0]].enabled);
      var f = switchField(t[2], on);
      editors.push({ scope: "station", field: t[1], station_id: sid, original: on, get: f.get, kind: "bool" });
      body.appendChild(f.row);
    });

    // Fixed station position. For a station at a known spot with no live
    // receiver, an entered position is recorded on each event, kept distinct
    // from a live satellite fix. Each field may be left blank to clear it.
    var loc = st.location || {};
    var locGroup = el("details", { class: "subgroup edit-advanced" }, [el("summary", { text: "Fixed station position (optional)" })]);
    locGroup.appendChild(el("p", { class: "form-hint", text: "For a station at a known, surveyed spot with no live receiver, enter its position. Decimal degrees, WGS84. Latitude -90 to 90 (north positive, south negative); longitude -180 to 180 (east positive, west negative). Example: 18.2100, -67.1500. Leave a field blank to clear it. A position entered here is recorded as an entered position, kept separate from a live satellite fix." }));
    function locEditor(field, labelText, val) {
      var input = el("input", { type: "number", class: "form-input", step: "any", value: (val === null || val === undefined) ? "" : String(val) });
      editors.push({ scope: "station", field: field, station_id: sid, original: (val === null || val === undefined) ? "" : val, get: function () { return input.value; }, kind: "numberOrNull" });
      locGroup.appendChild(editField(labelText, input));
    }
    locEditor("station_latitude", "Latitude", loc.latitude);
    locEditor("station_longitude", "Longitude", loc.longitude);
    locEditor("station_elevation", "Elevation (meters, optional)", loc.elevation);
    body.appendChild(locGroup);

    // This station's models are not edited here. They live under Settings, Model
    // paths, where every model a station uses is shown together with whether its
    // file is actually present, so a model is never set from a form that cannot
    // show that.
    body.appendChild(el("p", { class: "form-hint", text:
      "This station's models are set under Settings, Model paths, where each one is shown with its version, its citation, and whether the file is present." }));

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

  // ---- Settings, Model paths --------------------------------------------
  //
  // Every model the system can be pointed at, in one place, grouped by the
  // station that uses it. A station's models are per-station configuration, so
  // the panel is built from the configured stations rather than from a fixed
  // list of files. Each entry carries the same measured-versus-configured
  // distinction the Brain panel uses: a path that is set but whose file is not
  // on disk is never shown as ready, and a slot with no path says plainly that
  // no model is set rather than displaying a guess.
  function renderModelsEditor(cfg) {
    var host = region("settings-models");
    if (!host) { return; }
    cfg = cfg || {};
    clear(host);
    host.appendChild(el("p", { class: "settings-desc", text:
      "Every model this system uses, grouped by the station that uses it. Audtheia ships with no species models of its own: you supply models trained on the species each station was deployed to watch, and point each slot at the file. A slot with no file is reported as having no model set." }));
    var loading = el("p", { class: "empty-state", text: "Reading model files." });
    host.appendChild(loading);

    // The presence of each file is read from the backend rather than assumed,
    // so a path that points at nothing cannot be mistaken for a working model.
    apiGet("/brain/models").then(function (models) {
      if (loading.parentNode === host) { host.removeChild(loading); }
      var files = models.files || {};
      var desktop = cfg.desktop_models || {};
      var verifierPath = (desktop.visual_rfdetr || {}).path || null;
      var stations = cfg.stations || [];

      host.appendChild(el("h4", { text: "This computer" }));
      host.appendChild(modelBox({
        title: "Desktop hub",
        subtitle: "Runs after an observation has been captured, for every station",
        build: function (box) {
          box.appendChild(modelEntry(
            "Vision verification",
            "on this computer",
            "Re-scores the saved frames of an event to a publication standard. It can overrule a station's call.",
            desktop.visual_rfdetr, files));
          box.appendChild(el("p", { class: "card-note", text:
            "The language model is chosen in Brain, under Models and Memory." }));
        },
        edit: function (box, editors) {
          var rf = desktop.visual_rfdetr || {};
          modelPathFields(editors, box, "global", null, "Verification model (ONNX)",
            "visual_rfdetr_path", "visual_rfdetr_version", "visual_rfdetr_citation", rf,
            "The high-accuracy model that re-scores saved frames on this computer, for every station. This is not the model that runs during capture: each station's screening model is set on that station's card below. Leave empty to set no model.",
            "models/visual/my_verification_model.onnx");
        }
      }));

      // Stations run from this computer belong with the Desktop hub, under This
      // computer: a station with a desktop model or source, and any not yet
      // configured to run anywhere, is managed here. Only a station with a model
      // compiled for a Pi accelerator is listed as a Pi field station below.
      // station-model-grid divides each row between the cards that exist, so two
      // stations read as two wide cards rather than a pair pushed into a corner.
      var desktopStations = stations.filter(function (st) {
        var d = stationDeployment(st);
        return d.desktop || !d.field;
      });
      var fieldStations = stations.filter(function (st) { return stationDeployment(st).field; });

      host.appendChild(el("h5", { class: "form-group-title", text: "Stations run from this computer" }));
      if (!desktopStations.length) {
        host.appendChild(el("p", { class: "card-note", text: "No station is set up to run on this computer yet." }));
      } else {
        var deskGrid = el("div", { class: "card-grid station-model-grid" });
        desktopStations.forEach(function (st) { deskGrid.appendChild(stationModelBox(st, files, verifierPath)); });
        host.appendChild(deskGrid);
      }

      host.appendChild(el("h4", { text: "Pi Field Stations" }));
      host.appendChild(el("p", { class: "settings-desc", text:
        "Only stations configured with a Raspberry Pi field device appear here, with the screening model compiled for the Pi accelerator." }));
      if (!fieldStations.length) {
        host.appendChild(el("p", { class: "card-note", text: "No station is configured with a Pi device yet." }));
      } else {
        var fieldGrid = el("div", { class: "card-grid station-model-grid" });
        fieldStations.forEach(function (st) { fieldGrid.appendChild(stationModelBox(st, files, verifierPath)); });
        host.appendChild(fieldGrid);
      }
    }).catch(function (e) {
      if (loading.parentNode === host) { host.removeChild(loading); }
      host.appendChild(el("p", { class: "card-note", text: "Could not read the model files: " + e.message }));
    });
  }

  // A model box in the same shape as the Brain panel's station cards: a read
  // view with an Edit affordance that swaps in a form, so every configuration
  // surface in the application is edited the same way.
  function modelBox(spec) {
    var card = el("article", { class: "card" }, [
      el("div", { class: "card-title", text: spec.title }),
      spec.subtitle ? el("p", { class: "card-meta", text: spec.subtitle }) : null
    ]);
    var body = el("div");
    card.appendChild(body);

    function read() {
      clear(body);
      spec.build(body);
      if (!state.settingsCanEdit) {
        body.appendChild(el("p", { class: "card-note", text:
          "Models are configured on the desktop. This node is not the desktop." }));
        return;
      }
      var edit = el("button", { type: "button", class: "btn", text: "Edit" });
      edit.addEventListener("click", form);
      body.appendChild(el("div", { class: "form-actions" }, [edit]));
    }

    function form() {
      clear(body);
      var editors = [];
      var msg = el("p", { class: "form-message" });
      spec.edit(body, editors);
      body.appendChild(msg);
      var save = el("button", { type: "button", class: "btn btn-primary", text: "Save models" });
      var cancel = el("button", { type: "button", class: "btn", text: "Cancel", onclick: read });
      save.addEventListener("click", function () {
        var changes = collectChanges(editors, msg);
        if (changes === null) { return; }
        if (!changes.length) { read(); return; }
        save.disabled = true; msg.textContent = "Saving.";
        saveSettings(changes).then(settingsSaved).catch(function (e) {
          save.disabled = false; msg.textContent = "Could not save: " + e.message;
        });
      });
      body.appendChild(el("div", { class: "form-actions" }, [save, cancel]));
    }

    read();
    return card;
  }

  // One station's models: what screens in the field, what screens on the desktop
  // when there is no field hardware, and what listens. The hub's verification
  // model is named here too, because whether it differs from this station's
  // desktop screening model decides whether verification means anything.
  function stationModelBox(st, files, verifierPath) {
    var sid = st.station_id;
    var m = st.models || {};
    var acoustic = m.acoustic || {};

    return modelBox({
      title: st.station_name || sid,
      subtitle: [st.environment_type, st.habitat].filter(Boolean).map(humanize).join(", "),
      build: function (box) {
        // Placed by what it is configured to do, not assumed to be in a field.
        // A station created and run on this computer reads as running here.
        box.appendChild(el("div", { class: "badge-row card-deployment-row" }, [deploymentBadge(stationDeployment(st))]));

        box.appendChild(modelEntry(
          "Field screening (.hef)",
          "on the station's accelerator",
          "Checks every frame the camera produces. A frame with nothing in it is discarded straight away, so only real detections reach storage.",
          m.visual_pi, files));

        box.appendChild(modelEntry(
          "Desktop screening (.onnx)",
          "stands in when there is no field hardware",
          "Used only when this station's capture is run on this computer, from a video file or a webcam, instead of on a field station. This one watches video; a model that listens goes in the acoustic slot below.",
          m.visual_desktop, files));

        // Screening and verification by the same weights is not verification.
        // Stated on the card rather than left to be noticed, because an audit
        // that compares a model against itself reports agreement that carries
        // no evidence.
        var dtPath = (m.visual_desktop || {}).path;
        if (dtPath && verifierPath && String(dtPath) === String(verifierPath)) {
          box.appendChild(el("p", { class: "model-status is-absent", text:
            "This is the same file as the desktop verification model. In desktop mode this station would be screened and then re-scored by identical weights, so its agreement figures are not independent evidence." }));
        }

        box.appendChild(modelEntry(
          "Acoustic model",
          "on the station's processor",
          "Listens to the audio stream. A recognised sound opens an observation of its own, so the station hears as well as sees. Whatever this station is deployed to hear, birds, bats, cetaceans or reef fish, its model goes here.",
          acoustic, files));

        // A classifier reports a class number and the labels file names it.
        // Without one every detection reads as a bare index, which looks like
        // an identifier and is not one, so the gap is stated on the card.
        if (acoustic.path && !acoustic.labels_path) {
          box.appendChild(el("p", { class: "model-status is-absent", text:
            "No labels file is set for this model, so detections are recorded by class number rather than by name. Set the labels file in Edit." }));
        }
      },
      edit: function (box, editors) {
        // Both screening models belong to the station, not to a machine: they
        // are the same job, done wherever this station's capture is running.
        // Saying so above each group answers the question the old layout
        // raised, which was why a desktop model is being set inside a station.
        box.appendChild(el("h5", { class: "form-group-title", text: "When this station runs on its own hardware" }));
        modelPathFields(editors, box, "station", sid, "Field screening model (.hef)",
          "visual_pi_path", "visual_pi_version", "visual_pi_citation", m.visual_pi,
          "Compiled for the station's accelerator. Leave empty to set no model.",
          "models/visual/pi/my_screening_model.hef");

        box.appendChild(el("h5", { class: "form-group-title", text: "When you run this station's capture on this computer" }));
        modelPathFields(editors, box, "station", sid, "Desktop screening model (.onnx)",
          "visual_desktop_path", "visual_desktop_version", "visual_desktop_citation", m.visual_desktop,
          "This station's screening model for desktop mode, which is why it is set here and not under This computer. It replaces the field screening model above when there is no field hardware. Point it at a different model from the verification model, otherwise verification re-scores with identical weights and proves nothing.",
          "models/visual/desktop/my_screening_model.onnx");

        box.appendChild(el("h5", { class: "form-group-title", text: "What this station listens with" }));
        acousticModelFields(editors, box, sid, acoustic);
      }
    });
  }

  // Path, version and citation for one model slot. Every path field clears to
  // null when emptied, so a model can be unset as deliberately as it is set,
  // and no field is ever pre-filled with a filename the system merely guessed.
  function modelPathFields(editors, host, scope, sid, label, pathField, versionField, citationField, entry, hint, example) {
    entry = entry || {};
    function push(field, input, original) {
      var e = { scope: scope, field: field, original: original == null ? "" : original, get: function () { return input.value; }, kind: "textOrNull" };
      if (sid) { e.station_id = sid; }
      editors.push(e);
    }
    // The example lives in the placeholder, never as a stored value. A shipped
    // path that names a file nobody has reads as a model that is set and then
    // fails at capture; shown here it is plainly a form to fill in.
    var pathIn = el("input", { type: "text", class: "form-input", value: entry.path || "",
      placeholder: example || "No model set" });
    push(pathField, pathIn, entry.path);
    host.appendChild(editField(label, pathIn, hint));

    // Confirm the file at the moment the path is entered, without a trip to
    // Brain: whether it is present, its size (so a zero-byte or partial download
    // shows), and, for a visual slot whose type is fixed by its runtime, whether
    // the file is the kind the slot loads.
    var pathStatus = el("p", { class: "form-hint model-probe-status" });
    host.appendChild(pathStatus);
    function confirmPath() {
      var p = String(pathIn.value || "").trim();
      if (!p) { pathStatus.textContent = ""; pathStatus.className = "form-hint model-probe-status"; return; }
      pathStatus.textContent = "Checking this path...";
      pathStatus.className = "form-hint model-probe-status";
      apiSend("/models/probe", "POST", { path: p, field: pathField })
        .then(function (r) { renderVisualProbe(r, pathStatus); })
        .catch(function () {
          pathStatus.textContent = "Could not check this path.";
          pathStatus.className = "form-hint model-probe-status is-absent";
        });
    }
    pathIn.addEventListener("change", confirmPath);
    if (String(pathIn.value || "").trim()) { confirmPath(); }

    var verIn = el("input", { type: "text", class: "form-input", value: entry.version || "" });
    push(versionField, verIn, entry.version);
    host.appendChild(editField(label + " version", verIn, "Optional. Record which version of the model this is, so an observation can name the model that actually ran."));

    var citeIn = el("input", { type: "text", class: "form-input", value: entry.citation || "" });
    push(citationField, citeIn, entry.citation);
    host.appendChild(editField(label + " citation", citeIn, "Optional. Credit the model's source."));
  }

  // The one acoustic model a station listens with, edited as one flat form:
  // the model file, its labels file, and the audio shape the file expects. The
  // shape is read from the file when it can be, and what is read is kept apart
  // from what is proposed, so a guessed sample rate is never shown as measured.
  // No model family is named, and no file type is required: a .tflite file or a
  // TensorFlow SavedModel folder are both accepted.
  function acousticModelFields(editors, host, sid, acoustic) {
    acoustic = acoustic || {};

    var pathIn = el("input", { type: "text", class: "form-input", value: acoustic.path || "",
      placeholder: "models/acoustic/my_model_file (a .tflite file or a SavedModel folder)" });
    editors.push({ scope: "station", field: "acoustic_path", station_id: sid, original: acoustic.path || "", get: function () { return pathIn.value; }, kind: "textOrNull" });
    host.appendChild(editField("Acoustic model", pathIn,
      "Any acoustic model this station can run, in whatever form you trained or downloaded it. No file type is required."));
    var pathStatus = el("p", { class: "form-hint model-probe-status" });
    host.appendChild(pathStatus);

    var labelsIn = el("input", { type: "text", class: "form-input", value: acoustic.labels_path || "",
      placeholder: "models/acoustic/my_labels.txt" });
    editors.push({ scope: "station", field: "acoustic_labels_path", station_id: sid, original: acoustic.labels_path || "", get: function () { return labelsIn.value; }, kind: "textOrNull" });
    host.appendChild(editField("Labels file", labelsIn,
      "One label per line in class order, or a JSON list or map. Without it every detection is recorded as a class number instead of a name."));

    var rateIn = el("input", { type: "number", step: "1", class: "form-input",
      value: (acoustic.sample_rate === null || acoustic.sample_rate === undefined) ? "" : String(acoustic.sample_rate),
      placeholder: "e.g. 48000" });
    editors.push({ scope: "station", field: "acoustic_sample_rate", station_id: sid, original: (acoustic.sample_rate === null || acoustic.sample_rate === undefined) ? "" : acoustic.sample_rate, get: function () { return rateIn.value; }, kind: "numberOrNull" });
    host.appendChild(editField("Sample rate (Hz)", rateIn,
      "The rate the model expects. This is not stored in the model file and cannot be read from it, so it is entered or confirmed once, never guessed."));

    var winIn = el("input", { type: "number", step: "any", class: "form-input",
      value: (acoustic.window_seconds === null || acoustic.window_seconds === undefined) ? "" : String(acoustic.window_seconds),
      placeholder: "e.g. 3.0" });
    editors.push({ scope: "station", field: "acoustic_window_seconds", station_id: sid, original: (acoustic.window_seconds === null || acoustic.window_seconds === undefined) ? "" : acoustic.window_seconds, get: function () { return winIn.value; }, kind: "numberOrNull" });
    host.appendChild(editField("Window length (seconds)", winIn,
      "The length of audio the model scores at once. Derived from the file's sample count and the sample rate when both are known."));

    var keyIn = el("input", { type: "text", class: "form-input", value: acoustic.output_key || "",
      placeholder: "e.g. label" });
    editors.push({ scope: "station", field: "acoustic_output_key", station_id: sid, original: acoustic.output_key || "", get: function () { return keyIn.value; }, kind: "textOrNull" });
    host.appendChild(editField("Output key", keyIn,
      "Only for a model with more than one output head: the name of the head to read. Leave empty for a single-output model."));

    var verIn = el("input", { type: "text", class: "form-input", value: acoustic.version || "" });
    editors.push({ scope: "station", field: "acoustic_version", station_id: sid, original: acoustic.version || "", get: function () { return verIn.value; }, kind: "textOrNull" });
    host.appendChild(editField("Acoustic model version", verIn, "Optional. Record which version of the model this is, so an observation can name the model that ran."));

    var citeIn = el("input", { type: "text", class: "form-input", value: acoustic.citation || "" });
    editors.push({ scope: "station", field: "acoustic_citation", station_id: sid, original: acoustic.citation || "", get: function () { return citeIn.value; }, kind: "textOrNull" });
    host.appendChild(editField("Acoustic model citation", citeIn, "Optional. Credit the model's source."));

    function probeAcoustic() {
      var p = String(pathIn.value || "").trim();
      if (!p) { pathStatus.textContent = ""; pathStatus.className = "form-hint model-probe-status"; return; }
      pathStatus.textContent = "Checking this path...";
      pathStatus.className = "form-hint model-probe-status";
      apiSend("/models/probe", "POST", { path: p, kind: "acoustic", labels_path: String(labelsIn.value || "").trim() })
        .then(function (r) { renderAcousticProbe(r, pathStatus, rateIn, winIn); })
        .catch(function () {
          pathStatus.textContent = "Could not check this path.";
          pathStatus.className = "form-hint model-probe-status is-absent";
        });
    }
    pathIn.addEventListener("change", probeAcoustic);
    labelsIn.addEventListener("change", probeAcoustic);
    // Confirm whatever is already configured, so an existing model shows its
    // status the moment the form opens rather than only after a trip to Brain.
    if (String(pathIn.value || "").trim()) { probeAcoustic(); }
  }

  // A visual model path confirmation: present or not, its size, and whether the
  // file is the type the slot's runtime loads. The type check applies only to
  // visual slots, which are fixed by their runtime; it never runs for acoustic.
  function renderVisualProbe(r, statusEl) {
    if (!r || !r.present) {
      statusEl.textContent = "No file is present yet at this path.";
      statusEl.className = "form-hint model-probe-status is-absent";
      return;
    }
    var parts = ["File present" + (r.size_bytes !== null && r.size_bytes !== undefined ? ", " + fmtBytes(r.size_bytes) : "")];
    var wrong = false;
    if (r.suffix_ok === false) {
      wrong = true;
      parts.push("but this is not one of " + (r.expected_suffixes || []).join(" or ") + ", which this slot loads. Check it is in the slot you meant");
    }
    statusEl.textContent = parts.join(". ");
    statusEl.className = "form-hint model-probe-status" + (wrong ? " is-absent" : " is-present");
  }

  // Turn a probe result into a one-line confirmation, filling the shape fields
  // from the file when they are still empty. What was read from the file and
  // what is only proposed are said separately, so a proposed rate is never
  // presented as measured.
  function renderAcousticProbe(r, statusEl, rateIn, winIn) {
    if (!r || !r.present) {
      statusEl.textContent = "No file is present yet at this path.";
      statusEl.className = "form-hint model-probe-status is-absent";
      return;
    }
    var parts = ["File present" + (r.size_bytes !== null && r.size_bytes !== undefined ? ", " + fmtBytes(r.size_bytes) : "")];
    var ac = r.acoustic || {};
    var read = ac.read || {};
    var proposed = ac.proposed || {};
    var derived = ac.derived || {};
    if (read.input_samples !== null && read.input_samples !== undefined) {
      parts.push("window read from file: " + read.input_samples + " samples");
    }
    if (read.class_count !== null && read.class_count !== undefined) {
      parts.push(read.class_count + " classes read from file");
    }
    if (proposed.sample_rate !== null && proposed.sample_rate !== undefined) {
      parts.push("proposed sample rate " + proposed.sample_rate + " Hz (not read from the file; confirm or change)");
      if (rateIn && !String(rateIn.value || "").trim()) { rateIn.value = String(proposed.sample_rate); }
      if (winIn && !String(winIn.value || "").trim() && derived.window_seconds !== null && derived.window_seconds !== undefined) {
        winIn.value = String(derived.window_seconds);
      }
    }
    if (r.labels) {
      if (!r.labels.present) { parts.push("labels file not found at the labels path"); }
      else if (r.labels.matches_class_count === false) { parts.push("labels count " + r.labels.count + " does not match the model's class count"); }
      else if (r.labels.count !== null && r.labels.count !== undefined) { parts.push(r.labels.count + " labels"); }
    }
    if (ac.error) { parts.push("could not read the audio shape from this file"); }
    statusEl.textContent = parts.join(". ");
    statusEl.className = "form-hint model-probe-status" + (ac.error ? " is-absent" : "");
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

    if (state.settingsCanEdit) {
      host.appendChild(renderDataDirectory(host, s));
      host.appendChild(renderArchiveControl(host, s));
    }
  }

  // Choose where captured data is stored (for example an external drive). It
  // applies to new captures; data already on disk is not moved.
  function renderDataDirectory(statusHost, s) {
    var wrap = el("div", { class: "subgroup" });
    wrap.appendChild(el("h4", { text: "Data directory" }));
    wrap.appendChild(el("p", { class: "settings-desc", text:
      "Where captured images and clips are stored. Point this at a large or external drive to keep the working drive free. It applies to new captures; data already saved is not moved." }));
    wrap.appendChild(el("p", { class: "card-note", text: "Currently: " + ((s.data && s.data.path) || "not set") }));
    var input = el("input", { type: "text", class: "form-input", placeholder: "for example D:/AudtheiaData", value: (s.data && s.data.path) || "" });
    var msg = el("p", { class: "form-message" });
    var save = el("button", { type: "button", class: "btn btn-small", text: "Save data directory" });
    save.addEventListener("click", function () {
      var path = input.value.trim();
      if (!path) { msg.textContent = "Enter a folder path."; return; }
      save.disabled = true; msg.textContent = "Saving.";
      apiSend("/settings/data-directory", "POST", { path: path })
        .then(function (r) { msg.textContent = r.note || "Saved."; save.disabled = false; apiGet("/storage").then(function (ns) { renderStorageStatus(statusHost, ns); }); })
        .catch(function (e) { save.disabled = false; msg.textContent = "Could not save: " + e.message; });
    });
    wrap.appendChild(el("div", { class: "control-row" }, [input, save]));
    wrap.appendChild(msg);
    return wrap;
  }

  // Export captured frames to a chosen folder, optionally freeing the originals
  // to reclaim space while keeping the observation record.
  function renderArchiveControl(statusHost, s) {
    var wrap = el("div", { class: "subgroup" });
    wrap.appendChild(el("h4", { text: "Export and free space" }));
    wrap.appendChild(el("p", { class: "settings-desc", text:
      "Copy captured frames (with a metadata sidecar) to another folder, for training or long-term keeping. Optionally free the originals afterward to reclaim space; the observation record, with its counts, taxa, verdicts and salience, always stays. Copying is verified before anything is freed, and the record is never deleted." }));
    var start = el("input", { type: "date", "aria-label": "From date" });
    var end = el("input", { type: "date", "aria-label": "To date" });
    var dest = el("input", { type: "text", class: "form-input", placeholder: "destination folder, for example E:/AudtheiaArchive" });
    var reclaim = el("input", { type: "checkbox", "aria-label": "Free the originals after copying" });
    var msg = el("p", { class: "form-message" });
    var run = el("button", { type: "button", class: "btn btn-small", text: "Export" });
    run.addEventListener("click", function () {
      var target = dest.value.trim();
      if (!target) { msg.textContent = "Enter a destination folder."; return; }
      if (reclaim.checked && !window.confirm("This will copy the selected frames to the destination and then delete the originals from Audtheia to free space. The observation records are kept. Continue?")) { return; }
      run.disabled = true; msg.textContent = reclaim.checked ? "Exporting and freeing space." : "Exporting.";
      apiSend("/storage/archive", "POST", {
        start: start.value || null, end: end.value || null,
        target_dir: target, reclaim: !!reclaim.checked
      }).then(function (r) {
        run.disabled = false; msg.textContent = r.note || "Done.";
        apiGet("/storage").then(function (ns) { renderStorageStatus(statusHost, ns); });
      }).catch(function (e) { run.disabled = false; msg.textContent = "Could not export: " + e.message; });
    });
    wrap.appendChild(el("div", { class: "control-row" }, [
      el("label", { class: "filter-field" }, [el("span", { text: "From" }), start]),
      el("label", { class: "filter-field" }, [el("span", { text: "To" }), end]),
      dest
    ]));
    wrap.appendChild(el("label", { class: "filter-field" }, [reclaim, el("span", { text: "Free the originals after copying (reclaim space)" })]));
    wrap.appendChild(el("div", { class: "control-row" }, [run]));
    wrap.appendChild(msg);
    return wrap;
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

  // A short, built-in walkthrough, rebuilt as a foldable accordion. Each phase is
  // a card that is collapsed by default, so a first-time user opens one thing at a
  // time and is never shown everything at once. A step that carries a terminal
  // command shows it in a monospace block with a one-click copy button. The copy
  // reuses the same hidden-textarea plus execCommand path as the Connect to Pi
  // key, so it works offline over http with no clipboard-permission prompt.
  function guideCopy(text, btn) {
    var scratch = el("textarea", { class: "guide-copy-scratch", "aria-hidden": "true" });
    scratch.value = text;
    document.body.appendChild(scratch);
    scratch.select();
    try {
      document.execCommand("copy");
      var was = btn.textContent;
      btn.textContent = "Copied";
      window.setTimeout(function () { btn.textContent = was; }, 1500);
    } catch (e) { /* selection is left for a manual copy */ }
    document.body.removeChild(scratch);
  }

  // A command block: one or more command lines shown as monospace text with a
  // Copy button that copies the exact lines.
  function guideCommand(lines) {
    var text = lines.join("\n");
    var btn = el("button", { type: "button", class: "btn btn-small guide-copy", text: "Copy" });
    btn.addEventListener("click", function () { guideCopy(text, btn); });
    return el("div", { class: "guide-cmd" }, [
      el("pre", { class: "guide-cmd-text" }, [el("code", { text: text })]),
      btn
    ]);
  }

  // Build one collapsible phase from { title, steps, note }. A step is a plain
  // string, or an object { text, cmds, examples } that attaches a copyable command
  // block or a set of inline example tokens beneath the step.
  function guidePhase(phase) {
    var ol = el("ol", { class: "guide-steps" });
    phase.steps.forEach(function (s) {
      if (typeof s === "string") { ol.appendChild(el("li", { text: s })); return; }
      var li = el("li", {}, [el("span", { text: s.text })]);
      if (s.cmds) { li.appendChild(guideCommand(s.cmds)); }
      if (s.examples) {
        var ex = el("div", { class: "guide-examples" });
        s.examples.forEach(function (e) { ex.appendChild(el("code", { class: "guide-example", text: e })); });
        li.appendChild(ex);
      }
      ol.appendChild(li);
    });
    var body = el("div", { class: "fold-body" }, [ol]);
    if (phase.note) { body.appendChild(el("p", { class: "card-note guide-phase-note", text: phase.note })); }
    return el("details", { class: "fold-card guide-phase" }, [
      el("summary", { class: "guide-phase-title", text: phase.title }),
      body
    ]);
  }

  function renderSetupGuide(host) {
    if (!host) { return; }
    clear(host);
    host.appendChild(el("p", { class: "settings-desc", text: "A short walkthrough to get Audtheia running, on the desktop alone or with a Raspberry Pi field station. Open one phase at a time; any command can be copied with a click." }));

    var phases = [
      {
        title: "Set up the visual detection model (desktop)",
        steps: [
          "Get your trained detector's weights. From Roboflow, use Download Weights (not the dataset); this is the checkpoint file, for example weights.pt.",
          { text: "Export it to ONNX. This is a one-time, offline build step, not something that runs during capture:", cmds: ["pip install \"rfdetr[onnxexport]\"", "python scripts/export_rfdetr_onnx.py"] },
          "The exporter writes an .onnx file under models/visual. The exact arguments and the training workflow are in the custom models guide (docs/custom-models.md).",
          "Point the app at it in two places, because the desktop uses two models. Edit the station and set its Desktop screening model to that .onnx (the detector that runs during capture), and under Settings, Model paths set the Verification model to the same file (the re-score). One file can serve both.",
          "The .onnx file must actually exist on disk. Setting a path or a citation does not create it; if a path points at no file, capture cannot start."
        ]
      },
      {
        title: "Run desktop capture, no hardware",
        steps: [
          { text: "Open Detections, use Set capture source, pick a station, and enter a source, one of:", examples: ["webcam:0", "stream:<web page url>", "url:<direct stream url>", "file:C:/clip.mp4"] },
          "Open Detections, Capture, and press Start for that station. Detections appear below, each with its captured frame; a station with no screening model in place cannot start."
        ]
      },
      {
        title: "Connect a Raspberry Pi field station",
        steps: [
          "Settings, Stations, Add station: give it a name and environment. A station identifier is generated for you.",
          "On the station card, choose Connect to Pi and copy the shown desktop key.",
          "Flash a Raspberry Pi 5 with the AI HAT+ 2 using Raspberry Pi Imager. In the advanced options, enable SSH with public-key authentication and paste the key.",
          { text: "Boot the Pi on the same network, return to Connect to Pi, enter its address and user, and connect. It then runs on its own and broadcasts its field hotspot. The address is an IP or a name ending in .local:", examples: ["<ip address>", "<name>.local"] }
        ]
      },
      {
        title: "Add environmental sensors",
        steps: [
          "Settings, Sensors: expand a station and choose Add channel.",
          "Pick a reference sensor (pH, temperature, salinity, and so on) to fill the driver and quality-control ranges, or choose Custom, then save. Readings appear on the Sensors panel once the station is capturing."
        ]
      },
      {
        title: "Species data and reports",
        steps: [
          "Settings, Species data credentials: add your free IUCN token to enrich records with Red List conservation status. GBIF taxonomy is anonymous and needs no login.",
          "Settings, Schedules: choose how often reports and the longitudinal pass run. Generate a report any time from the Reports panel."
        ]
      }
    ];

    phases.forEach(function (p) { host.appendChild(guidePhase(p)); });

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
