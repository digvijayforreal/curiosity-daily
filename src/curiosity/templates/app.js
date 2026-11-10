/* Curiosity Daily — front-end for the live search / tracking pages.
 *
 * If window.CURIOSITY_API_BASE is set (a deployed backend), the search bar uses
 * the full pipeline: Wikipedia + image + pageview trend + web search + AI
 * summary, and the "Track" button works. If it is blank, the page falls back to
 * Wikipedia's public APIs directly (summary + image + pageview trend) so the
 * static site still does something useful without a backend.
 */
(function () {
  "use strict";
  var API = (window.CURIOSITY_API_BASE || "").replace(/\/$/, "");

  function el(id) { return document.getElementById(id); }
  function esc(s) {
    return (s || "").replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function sparkline(views) {
    if (!views || !views.length) return "";
    var w = 260, h = 48, max = Math.max.apply(null, views), min = Math.min.apply(null, views);
    var rng = max - min || 1;
    var pts = views.map(function (v, i) {
      var x = (i / (views.length - 1)) * (w - 4) + 2;
      var y = h - 2 - ((v - min) / rng) * (h - 6);
      return x.toFixed(1) + "," + y.toFixed(1);
    }).join(" ");
    return '<svg class="spark" viewBox="0 0 ' + w + ' ' + h + '" width="' + w + '" height="' + h +
      '"><polyline fill="none" stroke="#5b3df5" stroke-width="2" points="' + pts + '"/></svg>';
  }

  function statusBadge(status) {
    if (!status) return "";
    var cls = "badge";
    if (/spik|rising/i.test(status)) cls += " rising";
    else if (/cool|fad/i.test(status)) cls += " fading";
    else cls += " peaking";
    return '<span class="' + cls + '">' + esc(status) + "</span>";
  }

  function renderResult(r) {
    if (!r || !r.found) {
      return '<div class="empty">Nothing found. Try another name or spelling.</div>';
    }
    var pv = r.pageviews || {};
    var imgs = r.images || (r.image ? [r.image] : []);
    var gallery = imgs.slice(0, 4).map(function (u) {
      return '<img class="thumb" src="' + esc(u) + '" alt="" loading="lazy" referrerpolicy="no-referrer">';
    }).join("");
    var web = (r.web || []).map(function (w) {
      return '<li><a href="' + esc(w.url) + '" target="_blank" rel="noopener">' + esc(w.title) +
        "</a> <span class=\"src\">" + esc(w.source || "") + "</span></li>";
    }).join("");
    var trackBtn = API
      ? '<button id="trackBtn" class="track-btn" data-q="' + esc(r.query || r.title) + '">★ Track daily</button>'
      : '<span class="hint">Deploy the backend to enable daily tracking.</span>';

    return (
      '<article class="result">' +
      '<div class="result-head">' +
      (r.image ? '<img class="portrait" src="' + esc(r.image) + '" alt="" referrerpolicy="no-referrer">' : "") +
      '<div class="result-meta"><h2>' + esc(r.title) + "</h2>" +
      '<div class="muted">' + esc(r.description || "") + "</div>" +
      '<div class="badges">' + statusBadge(pv.status) +
      (pv.today ? '<span class="badge views">' + Number(pv.today).toLocaleString() + " views/day</span>" : "") +
      "</div>" +
      (pv.views ? sparkline(pv.views) : "") +
      "<div>" + trackBtn + "</div>" +
      "</div></div>" +
      (r.summary ? '<p class="summary">' + esc(r.summary) + "</p>" : "") +
      (r.extract && !r.summary ? '<p class="summary">' + esc(r.extract) + "</p>" : "") +
      (gallery ? '<div class="gallery">' + gallery + "</div>" : "") +
      (web ? '<div class="sources"><span class="label">On the web</span><ul>' + web + "</ul></div>" : "") +
      (r.wikipedia_url ? '<a class="wiki-link" href="' + esc(r.wikipedia_url) + '" target="_blank" rel="noopener">Read on Wikipedia →</a>' : "") +
      "</article>"
    );
  }

  // --- data sources ---------------------------------------------------- //
  function trackViaApi(q) {
    return fetch(API + "/api/track?q=" + encodeURIComponent(q)).then(function (r) { return r.json(); });
  }

  // Client-side fallback using Wikipedia's CORS-enabled APIs (no backend).
  function trackViaWikipedia(q) {
    var base = "https://en.wikipedia.org";
    return fetch(base + "/w/api.php?action=opensearch&limit=1&namespace=0&format=json&origin=*&search=" +
      encodeURIComponent(q))
      .then(function (r) { return r.json(); })
      .then(function (os) {
        var title = os[1] && os[1][0];
        if (!title) return { query: q, found: false };
        return fetch(base + "/api/rest_v1/page/summary/" + encodeURIComponent(title))
          .then(function (r) { return r.json(); })
          .then(function (s) {
            var result = {
              query: q, found: true, title: s.title,
              description: s.description || "",
              extract: s.extract || "",
              image: (s.thumbnail && s.thumbnail.source) || "",
              images: (s.thumbnail && [s.thumbnail.source]) || [],
              wikipedia_url: s.content_urls && s.content_urls.desktop && s.content_urls.desktop.page,
              web: [], pageviews: {}
            };
            // pageviews (last 30 days)
            var end = new Date(Date.now() - 2 * 86400000);
            var start = new Date(end.getTime() - 29 * 86400000);
            function ymd(d) { return "" + d.getUTCFullYear() + String(d.getUTCMonth() + 1).padStart(2, "0") + String(d.getUTCDate()).padStart(2, "0"); }
            var pv = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/" +
              encodeURIComponent(title.replace(/ /g, "_")) + "/daily/" + ymd(start) + "/" + ymd(end);
            return fetch(pv).then(function (r) { return r.json(); }).then(function (p) {
              if (p && p.items) {
                var views = p.items.map(function (i) { return i.views; });
                result.pageviews = { views: views, today: views[views.length - 1] || 0, status: "" };
              }
              return result;
            }).catch(function () { return result; });
          });
      });
  }

  function doSearch(q) {
    if (!q) return;
    var out = el("results");
    out.innerHTML = '<div class="loading">Searching…</div>';
    var p = API ? trackViaApi(q) : trackViaWikipedia(q);
    p.then(function (r) {
      r.query = q;
      out.innerHTML = renderResult(r);
      var btn = el("trackBtn");
      if (btn) btn.addEventListener("click", function () { addWatch(btn.getAttribute("data-q"), btn); });
    }).catch(function (e) {
      out.innerHTML = '<div class="empty">Something went wrong: ' + esc(String(e)) + "</div>";
    });
  }

  function addWatch(q, btn) {
    if (!API) return;
    btn.disabled = true; btn.textContent = "Adding…";
    fetch(API + "/api/watch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q: q })
    }).then(function (r) { return r.json(); }).then(function () {
      btn.textContent = "★ Tracking";
    }).catch(function () { btn.disabled = false; btn.textContent = "★ Track daily"; });
  }

  function initSearch() {
    var form = el("searchForm");
    if (!form) return;
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      doSearch(el("q").value.trim());
    });
    var params = new URLSearchParams(location.search);
    if (params.get("q")) { el("q").value = params.get("q"); doSearch(params.get("q")); }
  }

  function initWatchlist() {
    var out = el("watchlist");
    if (!out) return;
    if (!API) { out.innerHTML = '<div class="empty">Tracking needs the backend. Set it up to see your watchlist here.</div>'; return; }
    out.innerHTML = '<div class="loading">Loading…</div>';
    fetch(API + "/api/watchlist").then(function (r) { return r.json(); }).then(function (d) {
      var items = (d.watchlist || []);
      if (!items.length) { out.innerHTML = '<div class="empty">No tracked subjects yet. Search for someone and click “Track daily”.</div>'; return; }
      out.innerHTML = items.map(function (w) {
        var s = w.latest || {};
        return '<article class="story">' +
          '<h2>' + esc(w.display) + " " + statusBadge(s.status) + "</h2>" +
          (s.image ? '<img class="thumb" src="' + esc(s.image) + '" referrerpolicy="no-referrer" alt="">' : "") +
          (s.summary ? '<p class="explanation">' + esc(s.summary) + "</p>" : "") +
          "</article>";
      }).join("");
    }).catch(function () { out.innerHTML = '<div class="empty">Could not load the watchlist.</div>'; });
  }

  document.addEventListener("DOMContentLoaded", function () { initSearch(); initWatchlist(); });
})();
