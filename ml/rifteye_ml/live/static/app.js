// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Federico Vietti and Wardeye contributors
//
// The live page's only script: an EventSource for state/feed, an <img> for the MJPEG stream, and
// an SVG overlay drawn from the latest state. Plain ES2019, no build step, no modules, nothing
// fetched from outside this page -- it has to work with the browser offline except for this server.
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var LABEL_MIN_PX = 60; // a track's on-screen box needs to be at least this big (frame px) for a label

  var titleEl = document.getElementById("title");
  var statusEl = document.getElementById("status");
  var statsEl = document.getElementById("stats-text");
  var bannerEl = document.getElementById("banner");
  var videoEl = document.getElementById("video");
  var overlay = document.getElementById("overlay");
  var playersEl = document.getElementById("players");
  var feedEl = document.getElementById("feed");
  var hoverEl = document.getElementById("hover");
  var showBoxesEl = document.getElementById("show-boxes");

  var trackEls = {};       // track id -> { poly, label, track }
  var hotTrack = null;     // the track id currently hovered
  var pointer = { x: 0, y: 0 };
  var frameW = 1920;
  var frameH = 1080;

  // ---------------------------------------------------------------------------------------
  // /stream.mjpg: the browser decodes the multipart stream on its own; we only need to retry
  // if it ever errors (the server restarting, a network blip).
  // ---------------------------------------------------------------------------------------
  videoEl.addEventListener("error", function () {
    setTimeout(function () {
      videoEl.src = "/stream.mjpg?r=" + Date.now();
    }, 1000);
  });

  // ---------------------------------------------------------------------------------------
  // /events (Server-Sent Events)
  // ---------------------------------------------------------------------------------------
  function connect() {
    setStatus("connecting");
    var es = new EventSource("/events");
    es.addEventListener("state", function (ev) {
      try {
        renderState(JSON.parse(ev.data));
      } catch (err) {
        /* a malformed message: ignore it and wait for the next one */
      }
    });
    es.addEventListener("feed", function (ev) {
      try {
        addFeedItem(JSON.parse(ev.data));
      } catch (err) {
        /* ignore */
      }
    });
    // EventSource reconnects itself on error; we just reflect that in the pill until the next
    // "state" message arrives.
    es.onerror = function () {
      setStatus("connecting");
    };
  }

  function setStatus(status) {
    var label = status === "live" ? "LIVE" : status === "starting" ? "connecting" : status;
    statusEl.textContent = label;
    statusEl.className = "pill pill-" + (status === "starting" ? "connecting" : status);
  }

  function renderState(state) {
    titleEl.textContent = state.title ? "– " + state.title : "";
    document.title = "Wardeye live" + (state.title ? " – " + state.title : "");

    if (state.frame && state.frame.width && state.frame.height) {
      frameW = state.frame.width;
      frameH = state.frame.height;
      overlay.setAttribute("viewBox", "0 0 " + frameW + " " + frameH);
    }

    setStatus(state.status || "starting");
    var fps = (state.fps && state.fps.processed) || 0;
    var latencyMs = Math.round((state.latency_s || 0) * 1000);
    statsEl.textContent = fps.toFixed(1) + " fps · " + latencyMs + " ms";

    var showBanner = (state.status === "error" || state.status === "ended" || state.status === "away") && !!state.message;
    bannerEl.hidden = !showBanner;
    bannerEl.textContent = showBanner ? state.message : "";
    bannerEl.className = state.status === "error" ? "error" : "";

    drawTracks(state.tracks || []);
    renderPlayers(state.players || [], state.tracks || []);
  }

  // ---------------------------------------------------------------------------------------
  // The SVG overlay: one <polygon>+<text> pair kept per track id, updated in place so a track
  // that stays on the table doesn't lose its hover listeners between updates.
  // ---------------------------------------------------------------------------------------
  function drawTracks(tracks) {
    var seen = {};
    var candidates = []; // named tracks big enough to be worth a label, biggest first
    for (var i = 0; i < tracks.length; i++) {
      var track = tracks[i];
      seen[track.id] = true;
      var entry = trackEls[track.id];
      if (!entry) {
        entry = trackEls[track.id] = makeTrackEls(track.id);
      }
      entry.track = track;
      // Out of sight (under a hand or another card) but still on the board: listed, not drawn.
      entry.poly.style.display = track.hidden ? "none" : "";
      if (track.hidden) {
        entry.label.style.display = "none";
        continue;
      }

      var quad = track.quad || [];
      var pts = [];
      var xs = [];
      var ys = [];
      for (var j = 0; j < quad.length; j++) {
        pts.push(quad[j][0].toFixed(1) + "," + quad[j][1].toFixed(1));
        xs.push(quad[j][0]);
        ys.push(quad[j][1]);
      }
      entry.poly.setAttribute("points", pts.join(" "));
      entry.poly.setAttribute("class", "box box-" + track.state + (track.kind === "rune" ? " box-rune" : "") +
        (hotTrack === track.id ? " hot" : ""));
      entry.label.style.display = "none"; // candidates below turn their own label back on

      if (track.state === "named" && track.name && track.kind !== "rune" && xs.length > 0) {
        var w = Math.max.apply(null, xs) - Math.min.apply(null, xs);
        var h = Math.max.apply(null, ys) - Math.min.apply(null, ys);
        if (Math.min(w, h) >= LABEL_MIN_PX) {
          candidates.push({
            entry: entry, area: w * h,
            cx: (Math.max.apply(null, xs) + Math.min.apply(null, xs)) / 2,
            top: Math.min.apply(null, ys),
          });
        }
      }
    }
    for (var id in trackEls) {
      if (Object.prototype.hasOwnProperty.call(trackEls, id) && !seen[id]) {
        overlay.removeChild(trackEls[id].poly);
        overlay.removeChild(trackEls[id].label);
        delete trackEls[id];
        if (hotTrack === id) hideHover();
      }
    }
    placeLabels(candidates);
  }

  // A label can be wider than its own (often narrow) card, so two cards close together can
  // propose overlapping labels. The bigger (closer to camera, more likely to matter) box wins;
  // the loser's box still gets drawn, just without a name floating over a neighbour's.
  function placeLabels(candidates) {
    candidates.sort(function (a, b) {
      return b.area - a.area;
    });
    var placed = [];
    for (var i = 0; i < candidates.length; i++) {
      var c = candidates[i];
      var label = c.entry.label;
      label.textContent = c.entry.track.name;
      label.setAttribute("x", c.cx.toFixed(1));
      label.setAttribute("y", Math.max(24, c.top - 8).toFixed(1));
      label.style.display = "";
      var box = label.getBBox();
      var rect = { x: box.x - 4, y: box.y - 4, width: box.width + 8, height: box.height + 8 };
      var collides = false;
      for (var k = 0; k < placed.length; k++) {
        if (rectsOverlap(rect, placed[k])) {
          collides = true;
          break;
        }
      }
      if (collides) {
        label.style.display = "none";
      } else {
        placed.push(rect);
      }
    }
  }

  function rectsOverlap(a, b) {
    return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
  }

  function makeTrackEls(id) {
    var poly = document.createElementNS(SVG_NS, "polygon");
    var label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("class", "box-label");
    label.setAttribute("text-anchor", "middle");
    poly.addEventListener("pointerenter", function () {
      onHover(id);
    });
    poly.addEventListener("pointerleave", function () {
      if (hotTrack === id) hideHover();
    });
    overlay.appendChild(poly);
    overlay.appendChild(label);
    return { poly: poly, label: label, track: null };
  }

  showBoxesEl.addEventListener("change", function (e) {
    overlay.classList.toggle("hide-boxes", !e.target.checked);
  });

  // ---------------------------------------------------------------------------------------
  // Hover card
  // ---------------------------------------------------------------------------------------
  function onHover(id) {
    hotTrack = id;
    var entry = trackEls[id];
    if (!entry || (entry.track && entry.track.kind === "rune")) return; // runes are tracked, not shown
    entry.poly.classList.add("hot");
    showHoverCard(entry.track);
  }

  function hideHover() {
    if (hotTrack && trackEls[hotTrack]) trackEls[hotTrack].poly.classList.remove("hot");
    hotTrack = null;
    hoverEl.hidden = true;
  }

  function showHoverCard(track) {
    hoverEl.textContent = "";
    if (track.state === "named") {
      var one = document.createElement("div");
      one.className = "hover-one";
      one.appendChild(artImg(track.printing_id, track.name));
      var name = document.createElement("div");
      name.className = "hover-name";
      name.textContent = track.name || "(unnamed)";
      one.appendChild(name);
      if (track.printing_id) {
        var meta = document.createElement("div");
        meta.className = "hover-meta"; // the printing id, in mono
        meta.textContent = track.printing_id;
        one.appendChild(meta);
      }
      var sure = document.createElement("div");
      sure.className = "hover-sure";
      sure.textContent = "Wardeye is " + Math.round((track.confidence || 0) * 100) + "% sure";
      one.appendChild(sure);
      hoverEl.appendChild(one);
    } else if (track.state === "unsure") {
      var note = document.createElement("div");
      note.className = "hover-note";
      note.textContent = "Not sure yet. Best guesses:";
      var row = document.createElement("div");
      row.className = "hover-guesses";
      var guesses = (track.guesses || []).slice(0, 3);
      for (var i = 0; i < guesses.length; i++) {
        var g = guesses[i];
        var fig = document.createElement("figure");
        fig.appendChild(artImg(g.printing_id, g.name));
        var cap = document.createElement("figcaption");
        var capName = document.createElement("span");
        capName.textContent = g.name || g.printing_id || "?";
        var pct = document.createElement("span"); // the guess's probability, small and primary, under its name
        pct.className = "p";
        pct.textContent = Math.round((g.p || 0) * 100) + "%";
        cap.appendChild(capName);
        cap.appendChild(pct);
        fig.appendChild(cap);
        row.appendChild(fig);
      }
      hoverEl.appendChild(note);
      hoverEl.appendChild(row);
    } else if (track.state === "facedown") {
      hoverEl.textContent = "Face-down card: never identified";
    } else {
      hoverEl.textContent = "New card, not identified yet";
    }
    if (track.under && track.under.length && track.state !== "facedown") {
      var stack = document.createElement("div");
      stack.className = "hover-under";
      stack.textContent = "Under it: " + track.under.map(function (u) { return u.name; }).join(", ");
      hoverEl.appendChild(stack);
    }
    hoverEl.hidden = false;
    positionHover();
  }

  function positionHover() {
    if (hoverEl.hidden) return;
    var r = hoverEl.getBoundingClientRect();
    var x = pointer.x + 18;
    var y = pointer.y - r.height / 2;
    if (x + r.width > window.innerWidth - 8) x = pointer.x - r.width - 18;
    y = Math.max(8, Math.min(window.innerHeight - r.height - 8, y));
    hoverEl.style.left = x + "px";
    hoverEl.style.top = y + "px";
  }

  document.addEventListener("pointermove", function (e) {
    pointer.x = e.clientX;
    pointer.y = e.clientY;
    if (hotTrack && !(e.target && e.target.classList && e.target.classList.contains("box"))) hideHover();
    positionHover();
  });

  function artImg(printingId, alt) {
    var img = document.createElement("img");
    img.alt = alt || "";
    if (printingId) {
      img.src = "/art/" + encodeURIComponent(printingId) + ".jpg";
      img.onerror = function () {
        img.style.visibility = "hidden";
      };
    } else {
      img.style.visibility = "hidden";
    }
    return img;
  }

  // ---------------------------------------------------------------------------------------
  // Right panel: per player, the legend and what is on their side of the table.
  // ---------------------------------------------------------------------------------------
  function renderPlayers(players, tracks) {
    playersEl.textContent = "";
    for (var p = 0; p < players.length; p++) {
      var player = players[p];
      playersEl.appendChild(playerSection(player, tracks));
    }
  }

  function playerSection(player, tracks) {
    var side = player.side;
    var named = [];
    var unsureN = 0;
    var facedownN = 0;
    var runesN = 0;
    var underIds = {};  // cards listed with the card they lie under
    tracks.forEach(function (t) { (t.under || []).forEach(function (u) { underIds[u.id] = true; }); });
    for (var i = 0; i < tracks.length; i++) {
      var t = tracks[i];
      if (t.side !== side || underIds[t.id]) continue;
      if (t.kind === "rune") {
        runesN++;
        continue;
      }
      if (t.kind === "legend") continue; // shown above as the player's legend
      if (t.state === "named") named.push(t);
      else if (t.state === "unsure") unsureN++;
      else if (t.state === "facedown") facedownN++;
    }
    var order = [];
    var groups = {};
    for (var n = 0; n < named.length; n++) {
      var track = named[n];
      var key = track.printing_id || track.name || track.id;
      if (!groups[key]) {
        groups[key] = { printing_id: track.printing_id, name: track.name || "(unnamed)", count: 0, under: [] };
        order.push(key);
      }
      groups[key].count++;
      (track.under || []).forEach(function (u) { if (groups[key].under.indexOf(u.name) < 0) groups[key].under.push(u.name); });
    }

    var section = document.createElement("section");
    section.className = "player";

    var h = document.createElement("h2");
    h.textContent = player.label || side;
    section.appendChild(h);

    var legend = document.createElement("div");
    legend.className = "legend";
    if (player.legend) {
      legend.appendChild(artImg(player.legend.printing_id, player.legend.name));
      var ln = document.createElement("span");
      ln.textContent = player.legend.name;
      legend.appendChild(ln);
    } else {
      var none = document.createElement("span");
      none.className = "muted";
      none.textContent = "legend not seen yet";
      legend.appendChild(none);
    }
    section.appendChild(legend);

    var tableH = document.createElement("h3");
    tableH.textContent = "On the table";
    section.appendChild(tableH);

    var list = document.createElement("ul");
    list.className = "card-list";
    if (order.length === 0 && unsureN === 0 && facedownN === 0) {
      var empty = document.createElement("li");
      empty.className = "muted";
      empty.textContent = "nothing yet";
      list.appendChild(empty);
    }
    for (var g = 0; g < order.length; g++) {
      var group = groups[order[g]];
      var li = document.createElement("li");
      li.appendChild(artImg(group.printing_id, group.name));
      var span = document.createElement("span");
      span.textContent = group.name + (group.count > 1 ? " ×" + group.count : "");
      li.appendChild(span);
      if (group.under.length) {
        var withEl = document.createElement("span");
        withEl.className = "with";
        withEl.textContent = " + " + group.under.join(", ");
        li.appendChild(withEl);
      }
      list.appendChild(li);
    }
    if (unsureN > 0 || facedownN > 0 || runesN > 0) {
      var bits = [];
      if (runesN > 0) bits.push(runesN + (runesN > 1 ? " runes" : " rune"));
      if (unsureN > 0) bits.push(unsureN + " unsure");
      if (facedownN > 0) bits.push(facedownN + " face-down");
      var muted = document.createElement("li");
      muted.className = "muted small";
      muted.textContent = bits.join(", ");
      list.appendChild(muted);
    }
    section.appendChild(list);
    return section;
  }

  // ---------------------------------------------------------------------------------------
  // Plays feed: newest first. Both the initial batch (sent oldest-first on connect) and later
  // live events are handled the same way -- prepend each as it arrives -- so newest ends up on
  // top either way.
  // ---------------------------------------------------------------------------------------
  function addFeedItem(ev) {
    var li = document.createElement("li");
    li.className = "feed-" + ev.kind;
    var time = document.createElement("span");
    time.className = "feed-time";
    time.textContent = formatTime(ev.t);
    var thumb = ev.printing_id ? artImg(ev.printing_id, "") : document.createElement("span");
    if (!ev.printing_id) thumb.className = "feed-thumb-empty";
    var text = document.createElement("span");
    text.className = "feed-text";
    text.textContent = ev.text || "";
    li.appendChild(time);
    li.appendChild(thumb);
    li.appendChild(text);
    feedEl.insertBefore(li, feedEl.firstChild);
  }

  function formatTime(t) {
    var s = Math.max(0, Math.floor(t || 0));
    var h = Math.floor(s / 3600);
    var m = Math.floor((s % 3600) / 60);
    var r = s % 60;
    var ms = (r < 10 ? "0" : "") + r;
    return h > 0 ? h + ":" + (m < 10 ? "0" : "") + m + ":" + ms : m + ":" + ms; // a VOD's clock: 14:59:55
  }

  connect();
})();
