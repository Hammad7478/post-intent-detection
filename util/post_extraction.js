(async () => {
  const MAX_POSTS = 100;
  const MIN_WORDS = 40;
  const BAD_RE = /discord\.gg|faq|rules|survey|moderators|mod team|save3rdpartyapps|cscareerquestions protests/i;
  const CONCURRENCY = 6;

  // Build queue from listing page (old reddit)
  const queue = [];
  document.querySelectorAll(".thing.link").forEach(p => {
    const permalink = p.getAttribute("data-permalink");
    const author = (p.getAttribute("data-author") || "").toLowerCase();
    const stickied = p.classList.contains("stickied");
    const id = p.getAttribute("data-fullname") || "";

    if (!permalink) return;
    if (stickied) return;
    if (author === "automoderator") return;

    queue.push({ id_from_listing: id, url: "https://old.reddit.com" + permalink });
  });

  const targets = queue.slice(0, MAX_POSTS);
  console.log(`Processing ${targets.length} posts...`);

  // Fetch JSON for each post
  const results = new Array(targets.length);
  let idx = 0;

  async function fetchOne(item) {
    const res = await fetch(item.url + ".json", { credentials: "include" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const post = data?.[0]?.data?.children?.[0]?.data;

    const id = post?.name || item.id_from_listing || "";
    const subreddit = post?.subreddit || "";
    const text = (post?.selftext || "").trim();

    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const keep =
      text.length > 0 &&
      text !== "[removed]" &&
      text !== "[deleted]" &&
      wordCount >= MIN_WORDS &&
      !BAD_RE.test(text);

    return { id, subreddit, text, keep };
  }

  async function worker() {
    while (idx < targets.length) {
      const myIdx = idx++;
      try {
        results[myIdx] = await fetchOne(targets[myIdx]);
      } catch (e) {
        results[myIdx] = { id: targets[myIdx].id_from_listing || "", subreddit: "", text: "", keep: false };
      }
    }
  }

  await Promise.all(Array.from({ length: CONCURRENCY }, worker));

  // Group by subreddit, store only {id, text}, keep only keep:true
  const grouped = {};
  for (const r of results) {
    if (!r || !r.keep) continue;
    if (!grouped[r.subreddit]) grouped[r.subreddit] = [];
    grouped[r.subreddit].push({ id: r.id, text: r.text });
  }

  const payload = JSON.stringify(grouped, null, 2);

  // Clipboard copy (with fallback)
  try {
    await navigator.clipboard.writeText(payload);
    console.log("Copied grouped JSON ✅");
  } catch (e) {
    console.log("Clipboard blocked; printing below. Manually copy.");
  }

  console.log(payload);
})();