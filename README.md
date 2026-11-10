# Curiosity Daily

**What the world suddenly wants to know — generated automatically every day.**

Curiosity Daily detects Wikipedia pageview spikes, groups related pages into
stories, finds the news behind each story, writes a short *grounded* "why it's
trending" explanation, forecasts whether each topic will keep rising, peak or
fade, and publishes a static website, an RSS feed and a shareable card image.
It also exposes an **MCP server** so other AI assistants can query it.

Everything runs on **free tiers only**: MongoDB Atlas M0, Jina embeddings,
Gemini/Groq free LLM tiers, GitHub Actions (cron) and GitHub Pages (hosting).

---

## How it works (daily pipeline)

```
ingest → detect spikes → cluster → embed+store news → hybrid retrieval
       → agent (grounded + verified) → forecast → publish (site + RSS + card)
```

1. **Ingest** the top ~1000 Wikipedia articles for the latest complete day, pull
   60 days of history for the ~100 most notable candidates, and fetch news from
   free RSS feeds. All Wikimedia calls send a descriptive `User-Agent` (required).
2. **Detect spikes** with a robust median/MAD z-score against each page's own
   baseline, plus a minimum absolute view count. Keep the strongest.
3. **Cluster** related pages into single stories using Jina embeddings
   (`separation` task) and cosine similarity (e.g. *Solar eclipse* + *Totality*).
4. **Embed + store** news items in MongoDB with a 14-day TTL index.
5. **Hybrid retrieval** per story: a keyword Atlas Search query and a vector
   Atlas Search query run **separately**, then fuse in Python with Reciprocal
   Rank Fusion (we do **not** rely on `$rankFusion`, which may be unavailable on
   the free tier).
6. **Agent** writes a 2–3 sentence explanation from the retrieved snippets only,
   retries once with a reworded search if evidence is weak, and a verification
   pass requires every claim to be supported — otherwise it says *"reason
   unclear"*. No article text is copied; sources are linked.
7. **Forecast** the next 1–3 days of views (AutoETS when available, light
   damped-trend baseline otherwise) and label each story **Rising / Peaking /
   Fading** with a confidence indicator. The next day each forecast is scored
   against the actual views (log MAE) for the "how accurate were we" footer.
8. **Publish** a clean, mobile-friendly static site (today + archive), a
   `feed.xml` RSS feed, and a 1200×630 PNG share card.

---

## Quick start (no keys, no database)

The dry run uses small bundled sample data with a mocked LLM and mocked
embeddings, so you can see the whole thing work offline:

```bash
uv sync
uv run curiosity run --dry-run
```

Then open `site/index.html` in a browser. Run the tests with:

```bash
uv run pytest -q
```

---

## Project layout

```
src/curiosity/
  ingest.py       Wikimedia pageviews + RSS news (descriptive User-Agent)
  filters.py      drop Main_Page, Special:, lists, year pages, bot traffic
  detect.py       robust median/MAD spike detection
  embeddings.py   Jina v3 client (512-dim, task types, batching, usage budget)
  cluster.py      greedy embedding clustering into stories
  retrieval.py    keyword + vector search fused with RRF
  agent.py        grounded explanation + claim verification
  forecast.py     AutoETS / baseline forecast, labels, log-MAE accuracy
  publish.py      static site + RSS + Pillow share card
  mcp_server.py   FastMCP tools
  db.py           MongoStore (Atlas) and MemoryStore (dry-run/tests)
  pipeline.py     orchestration
  cli.py          `curiosity run | setup-indexes | mcp`
tests/            detector, fusion, labels, filters
config.yaml       all non-secret settings (feeds, thresholds, budgets)
.env.example      secrets template (copy to .env)
.github/workflows/daily.yml   daily cron + Pages deploy + manual runs
```

---

## Full setup (production)

### 1. Create a MongoDB Atlas free cluster
1. Sign up at <https://www.mongodb.com/atlas> and create a free **M0** cluster.
2. Create a database user and add your IP (or `0.0.0.0/0` for CI) to the Network
   Access allowlist.
3. Copy the connection string, e.g.
   `mongodb+srv://USER:PASS@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority`.

### 2. Get API keys (all have free tiers)
- **Jina** embeddings: <https://jina.ai/embeddings> → copy your API key.
- **LLM**: either **Gemini** (<https://aistudio.google.com/app/apikey>) or
  **Groq** (<https://console.groq.com/keys>). Set `LLM_PROVIDER` accordingly.

### 3. Configure locally
```bash
cp .env.example .env
# edit .env and fill in MONGODB_URI, JINA_API_KEY, LLM_PROVIDER, and the key
# matching your provider (GEMINI_API_KEY or GROQ_API_KEY)
uv sync
```

### 4. Create the two Atlas indexes
Exactly two indexes are used: one **Vector Search** index on
`news_items.embedding` (512 dims, cosine) and one **Atlas Search** index on the
news text fields.

```bash
uv run curiosity setup-indexes
```
Atlas builds search indexes asynchronously — give it a minute before the first
query. (Free tier allows up to 3 such indexes; we use 2.)

### 5. Run it for real
```bash
uv run curiosity run
```
This writes the site to `site/`. Run it again the next day and the forecast
accuracy footer starts reporting real numbers.

### 6. Automate with GitHub Actions + Pages
1. Push this repo to GitHub.
2. In **Settings → Secrets and variables → Actions**, add:
   `MONGODB_URI`, `MONGODB_DB` (optional), `JINA_API_KEY`, `LLM_PROVIDER`,
   `GEMINI_API_KEY` **or** `GROQ_API_KEY`, and `CURIOSITY_CONTACT`
   (a contact string for the Wikimedia User-Agent).
3. In **Settings → Pages**, set **Source = GitHub Actions**.
4. Set `site.base_url` in `config.yaml` to your Pages URL
   (e.g. `https://your-user.github.io/your-repo`) so RSS links are absolute.
5. The workflow runs daily at 06:30 UTC and can be triggered manually from the
   **Actions** tab (*Run workflow*).

---

## MCP server

Start it with:
```bash
uv run curiosity mcp
```

Tools exposed:
- `get_trending_today()` — today's ranked stories.
- `why_trending(topic)` — the grounded explanation with sources.
- `forecast_topic(topic)` — Rising/Peaking/Fading forecast.
- `search_news(query)` — hybrid keyword+vector news search.

Example client config (e.g. for a FastMCP-compatible assistant):
```json
{
  "mcpServers": {
    "curiosity": {
      "command": "uv",
      "args": ["run", "curiosity", "mcp"],
      "env": { "MONGODB_URI": "...", "JINA_API_KEY": "...", "LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "..." }
    }
  }
}
```

---

## Free-tier safety & storage

- **Storage**: history is stored only for candidate pages, news and embeddings
  expire after 14 days via a TTL index, and each run logs approximate collection
  sizes so you can watch the 512 MB M0 cap.
- **Jina tokens**: usage is tracked per run and logged; a warning fires when a
  run pushes total usage past 80% of `embeddings.monthly_token_budget` in
  `config.yaml`.
- **Resilience**: a failing feed or a single failing story is logged and
  skipped. If the LLM is unavailable, pages are still published with pageview
  data only and marked *reason unclear*.
- **No unsupported claims**: explanations are verified against retrieved
  snippets; unsupported topics are labelled *reason unclear*. Only a short title
  and a link are kept from any news source — never the article body.

---

## Commands

| Command | What it does |
|---|---|
| `uv run curiosity run` | Full daily pipeline + publish `site/`. |
| `uv run curiosity run --dry-run` | Same, offline, with bundled sample data. |
| `uv run curiosity setup-indexes` | Create the two Atlas search/vector indexes. |
| `uv run curiosity mcp` | Start the FastMCP server. |
| `uv run pytest -q` | Run the test suite. |

## License

MIT.
