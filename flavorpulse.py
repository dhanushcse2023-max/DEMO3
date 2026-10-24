"""
🍽️  FlavorPulse – Live Intelligence System (single-file prototype)
=================================================================
Turns guest feedback into live operational intelligence within minutes – with
NO traditional database: events live in memory (st.session_state) and can be
exported as CSV.

Event-driven flow
  Review arrives (stream / form)  ->  AI enrichment (Claude, or built-in rule NLP)
  -> Rules engine (severity, SLA, cluster detection) -> Alerts -> Live dashboard

Run
  pip install streamlit pandas            # + anthropic (optional, for Generative AI)
  streamlit run flavorpulse.py

Generative AI (optional): set ANTHROPIC_API_KEY (env var / Streamlit secret) or paste
the key in the sidebar. Without a key the app works fully with the offline rule engine.
"""
import json
import os
import random
import re
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

LOCATIONS = ["Downtown Flagship", "Mall Bistro", "Airport Express"]
SOURCES = ["Google", "Yelp", "TripAdvisor", "App survey", "Instagram"]
MODEL = os.getenv("FLAVORPULSE_MODEL", "claude-haiku-5-5")
RANK = {"None": 0, "Low": 0, "Medium": 1, "High": 2, "Critical": 3}
ICON = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢", "None": "🟢"}
SLA_MIN = {"Critical": 15, "High": 60, "Medium": 240}

# ============================ 1. AI / NLP ENGINE ============================
SAFETY_RULES = [  # (severity, label, terms, action, owner)
    ("Critical", "Illness", ["food poisoning", "vomit", "vomiting", "threw up", "throwing up", "diarrhea",
        "stomach cramps", "got sick", "made me sick", "made us sick", "nauseous", "nausea", "stomach bug", "hospital"],
     "Contact the guest within 15 min, log the illness complaint, quarantine suspect batches and review the last 48 h of prep and temperature logs.",
     "General Manager + Food Safety Officer"),
    ("Critical", "Allergen", ["allergic reaction", "allergy", "anaphylaxis", "epipen", "cross-contamination"],
     "Call the guest immediately, verify allergen controls and ticket notes, brief all staff on the allergen protocol.",
     "General Manager + Food Safety Officer"),
    ("Critical", "Pest", ["cockroach", "roach", "mouse", "mice", "maggot", "insect", "bug in"],
     "Isolate the area, call pest control, photograph the evidence and inspect kitchen and dining zones.",
     "General Manager + Facilities"),
    ("Critical", "Foreign object", ["broken glass", "glass shard", "piece of glass", "piece of plastic", "metal piece", "band-aid", "bandage"],
     "Retain the item/photos, inspect the line equipment, notify the guest and document the incident.", "Kitchen Manager"),
    ("High", "Undercooked", ["undercooked", "raw chicken", "raw inside", "pink inside", "still pink", "not cooked through"],
     "Check core cooking temperatures and thermometer calibration, retrain the grill/fry station, review the shift's batches.",
     "Kitchen Manager"),
    ("High", "Spoiled", ["spoiled", "rotten", "expired", "moldy", "mold", "smelled off", "tasted off"],
     "Check cold-chain and expiry dates, discard the suspect batch, record the supplier lot.", "Kitchen Manager"),
    ("Medium", "Hygiene", ["hair in", "found a hair", "dirty plate", "dirty fork", "sticky table", "unclean", "filthy", "dirty bathroom"],
     "Inspect the area, deep-clean, reinforce hairnet / cleaning checklists and apologise to the guest.", "Shift Supervisor"),
]
CATEGORIES = {
    "Food Quality": ["cold", "bland", "salty", "greasy", "soggy", "tasteless", "overcooked", "dry", "burnt", "delicious", "tasty", "fresh", "flavor"],
    "Service": ["rude", "waiter", "waitress", "server", "staff", "manager", "ignored", "friendly", "attentive", "service"],
    "Wait Time": ["wait", "waited", "slow", "took forever", "delay", "delayed", "queue", "quick", "fast"],
    "Cleanliness": ["dirty", "clean", "messy", "bathroom", "sticky", "smell"],
    "Pricing/Value": ["expensive", "overpriced", "price", "value", "pricey", "worth"],
    "Order Accuracy": ["wrong order", "wrong item", "forgot", "missing", "incorrect", "mistake"],
}
CAT_OWNER = {"Food Quality": "Kitchen Manager", "Service": "Front-of-House Manager", "Wait Time": "Front-of-House Manager",
             "Cleanliness": "Shift Supervisor", "Pricing/Value": "General Manager", "Order Accuracy": "Expo Lead", "General": "Shift Supervisor"}
CAT_ACTION = {"Food Quality": "Taste-check the dish at the pass, review recipe/holding times, offer the guest a remake.",
              "Service": "Coach the team member, call the guest to apologise and invite them back.",
              "Wait Time": "Review staffing vs. forecast and ticket times; add a runner during peaks.",
              "Cleanliness": "Run an immediate cleaning round and update the cleaning checklist.",
              "Pricing/Value": "Review portion size and menu pricing for this item.",
              "Order Accuracy": "Add a read-back step at expo and double-check modifiers.",
              "General": "Review the feedback with the shift team and follow up with the guest."}
POS = {"amazing", "great", "excellent", "delicious", "tasty", "friendly", "fresh", "love", "loved", "perfect", "best", "good",
       "lovely", "efficient", "fast", "quick", "attentive", "cozy", "clean", "wonderful", "happy"}
NEG = {"terrible", "awful", "bad", "worst", "rude", "cold", "slow", "dirty", "disgusting", "bland", "soggy", "overpriced",
       "wrong", "sick", "never", "ignored", "horrible", "stale", "greasy", "expensive", "noisy", "forgot"}


def _has(text, term):
    """Whole-word match that ignores negated mentions ('no mold')."""
    for m in re.finditer(r"\b" + re.escape(term) + r"\b", text):
        if not re.search(r"(\bno |\bnot |n't |\bnever |\bwithout )\s*\w*\s*$", text[max(0, m.start() - 14):m.start()]):
            return True
    return False


def rule_analyze(text, rating):
    t = text.lower()
    words, lex = re.findall(r"[a-z']+", t), 0
    for i, w in enumerate(words):
        neg = any(p in ("not", "no", "never", "wasn't", "isn't", "didn't") for p in words[max(0, i - 2):i])
        if w in POS:
            lex += -1 if neg else 1
        elif w in NEG:
            lex += 0.5 if neg else -1
    lex = max(-1, min(1, lex / 3))
    score = round(0.6 * (rating - 3) / 2 + 0.4 * lex, 2)
    sentiment = "Positive" if score > 0.25 else "Negative" if score < -0.25 else "Neutral"

    sev, evidence, action, owner, labels = "None", [], None, None, []
    for level, label, terms, act, own in SAFETY_RULES:
        hits = [x for x in terms if _has(t, x)]
        if hits:
            labels.append(label + " concern"); evidence += hits
            if RANK[level] > RANK[sev]:
                sev, action, owner = level, act, own
    food_safety = bool(labels)
    cats = [c for c, kws in CATEGORIES.items() if any(_has(t, k) for k in kws)]
    if food_safety:
        cats = ["Food Safety"] + cats
    if not food_safety and sentiment == "Negative":
        sev = "High" if rating == 1 else "Medium"
    elif not food_safety and rating <= 2:
        sev = "Medium"
    if not food_safety and RANK[sev] > 0:
        main = next((c for c in cats if c in CAT_ACTION), "General")
        action, owner = CAT_ACTION[main], CAT_OWNER[main]
        labels = [main + " complaint"]
    return {"sentiment": sentiment, "score": score, "categories": cats or ["General"], "severity": sev,
            "food_safety": food_safety, "evidence": evidence, "action": action or "No action needed – share praise with the team.",
            "owner": owner or "—", "summary": (labels[0] if labels else sentiment + " feedback"), "engine": "Rules"}


def _secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def get_client():
    key = st.session_state.get("api_key") or os.getenv("ANTHROPIC_API_KEY") or _secret("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=key)
    except Exception:
        return None


def llm(prompt, max_tokens=500):
    """Generative AI call; returns None when no key / any error (callers fall back to rules)."""
    client = get_client()
    if client is None:
        return None
    try:
        r = client.messages.create(model=MODEL, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
    except Exception as e:
        st.session_state["llm_error"] = str(e)[:140]
        return None


def analyze(text, rating, location, use_ai=True):
    res = rule_analyze(text, rating)
    if not use_ai:
        return res
    out = llm(f"""You are a restaurant operations analyst. Analyse this guest review and reply with ONLY a JSON object.
Location: {location}. Star rating: {rating}/5. Review: "{text}"
Keys: "sentiment" (Positive|Neutral|Negative), "severity" (None|Low|Medium|High|Critical; Critical = illness, allergen, pests, foreign objects),
"food_safety" (true|false), "categories" (list from: Food Safety, Food Quality, Service, Wait Time, Cleanliness, Pricing/Value, Order Accuracy),
"summary" (max 12 words), "action" (one concrete sentence for the manager).""", 350)
    try:
        j = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
        if RANK.get(j.get("severity"), 0) > RANK[res["severity"]]:
            res["severity"] = j["severity"]
        res["food_safety"] = res["food_safety"] or bool(j.get("food_safety"))
        res["sentiment"] = j.get("sentiment", res["sentiment"])
        res["categories"] = list(dict.fromkeys(res["categories"] + [c for c in j.get("categories", []) if c in CATEGORIES or c == "Food Safety"]))
        res["summary"] = str(j.get("summary", res["summary"]))[:90]
        if not res["evidence"] and j.get("action"):
            res["action"] = str(j["action"])
        res["engine"] = "Claude + rules"
    except Exception:
        pass
    return res


# ============================ 2. EVENT STORE (in memory) ============================
def ingest(location, source, rating, text, posted=None, detected=None, use_ai=True):
    now = datetime.now()
    a = analyze(text, rating, location, use_ai)
    st.session_state.next_id += 1
    ev = {"id": st.session_state.next_id, "posted": posted or now, "detected": detected or now, "location": location,
          "source": source, "rating": rating, "text": text, "status": "Open", **a,
          "sla_min": SLA_MIN.get(a["severity"], 0)}
    st.session_state.events.append(ev)
    return ev


SEED = [  # (minutes ago, location, source, rating, text)
    (12, 0, "Google", 5, "Amazing pasta and the staff were super friendly. Will be back!"),
    (35, 0, "Yelp", 4, "Great flavor, a little noisy but the server was attentive."),
    (70, 0, "Google", 2, "Waited 45 minutes for our food and the server ignored us. The fries were cold."),
    (95, 0, "TripAdvisor", 1, "Found a hair in my salad. Disgusting, and the manager just shrugged."),
    (140, 0, "App survey", 5, "Fresh ingredients and fast service, best burger in town."),
    (200, 0, "Instagram", 3, "Decent food, a bit overpriced for the portion."),
    (20, 1, "Google", 1, "My wife and I both got sick after eating the grilled chicken last night. Stomach cramps and vomiting all night."),
    (55, 1, "Yelp", 1, "The chicken was still pink inside. We sent it back but my kid felt nauseous afterwards."),
    (90, 1, "Google", 4, "Lovely dessert and a cozy atmosphere."),
    (130, 1, "App survey", 3, "Okay burger but my order was wrong and they forgot the drinks."),
    (160, 1, "Google", 5, "Friendly staff, quick service, delicious fries!"),
    (30, 1, "Instagram", 2, "Table was sticky and the bathroom was dirty. Not great."),
    (8, 2, "Google", 2, "Wait time was terrible, 30 minutes for a sandwich. The staff were rude."),
    (40, 2, "TripAdvisor", 1, "Very overpriced and the wrap was cold and soggy. Never again."),
    (75, 2, "Google", 5, "Perfect quick bite before my flight, tasty and fresh."),
    (110, 2, "Yelp", 4, "Efficient service, tasty bowl and good value."),
    (175, 2, "App survey", 1, "Allergic reaction after eating the salad even though I told them about my nut allergy."),
    (50, 2, "Google", 3, "Average food, nothing special but clean."),
]
STREAM = [  # (location index, source, rating, text) for the simulated live stream
    (0, "Google", 5, "Loved the steak, perfectly cooked and the waiter was lovely."),
    (0, "Yelp", 1, "Found a piece of plastic in my soup. Unacceptable!"),
    (0, "Instagram", 2, "Slow service tonight, waited ages for the bill."),
    (1, "Google", 1, "I threw up after the shrimp tacos. Do not eat here."),
    (1, "TripAdvisor", 4, "Great value lunch menu, friendly team."),
    (1, "Yelp", 2, "Burger was dry and the fries were cold."),
    (2, "Google", 5, "Super quick and delicious, perfect for travellers!"),
    (2, "App survey", 2, "Wrong order again and nobody apologised."),
    (2, "Yelp", 1, "Saw a cockroach near the counter. Disgusting."),
    (2, "Google", 3, "Fine for an airport, a bit pricey."),
    (0, "Google", 4, "Fresh salad and super clean tables."),
    (1, "Instagram", 1, "Chicken tasted off and the smell was awful, I am worried about my stomach."),
]


def seed_demo():
    st.session_state.events, st.session_state.next_id = [], 0
    now = datetime.now()
    for mins, li, src, rating, text in SEED:
        posted = now - timedelta(minutes=mins)
        ingest(LOCATIONS[li], src, rating, text, posted, posted + timedelta(seconds=random.randint(25, 140)), use_ai=False)
    st.session_state.last_stream = datetime.now()


def stream_one():
    li, src, rating, text = random.choice(STREAM)
    now = datetime.now()
    ingest(LOCATIONS[li], src, rating, text, now - timedelta(seconds=random.randint(5, 60)), now, st.session_state.get("use_ai", True))
    st.session_state.last_stream = now


# ============================ 3. RULES ENGINE / METRICS ============================
def is_incident(e):
    return RANK[e["severity"]] >= 1


def due(e):
    return e["detected"] + timedelta(minutes=e["sla_min"])


def sla_text(e, now):
    if e["status"] == "Resolved":
        return "✅ Resolved"
    mins = int((due(e) - now).total_seconds() // 60)
    return f"⏰ Overdue by {-mins} min" if mins < 0 else f"⏳ {mins} min left (due {due(e):%H:%M})"


def clusters(events, now, hours=3):
    """>=2 food-safety events at one location within 3 h => possible outbreak."""
    out = []
    for loc in LOCATIONS:
        hits = [e for e in events if e["location"] == loc and e["food_safety"] and e["posted"] >= now - timedelta(hours=hours)]
        if len(hits) >= 2:
            out.append((loc, hits))
    return out


def health_score(evs):
    if not evs:
        return None
    avg = sum(e["rating"] for e in evs) / len(evs)
    neg = sum(e["sentiment"] == "Negative" for e in evs) / len(evs)
    pen = sum(12 for e in evs if e["severity"] == "Critical" and e["status"] != "Resolved") + \
          sum(5 for e in evs if e["severity"] == "High" and e["status"] != "Resolved")
    return int(max(0, min(100, 100 * (0.5 * (avg - 1) / 4 + 0.5 * (1 - neg)) - pen)))


def fallback_brief(evs, now):
    inc = [e for e in evs if is_incident(e) and e["status"] != "Resolved"]
    crit = [e for e in inc if e["severity"] == "Critical"]
    cl = clusters(evs, now)
    avg = sum(e["rating"] for e in evs) / max(1, len(evs))
    lines = [f"**Executive brief ({now:%H:%M})** – {len(evs)} reviews, average rating {avg:.1f}/5, {len(inc)} open incidents ({len(crit)} critical)."]
    for loc, hits in cl:
        lines.append(f"🚨 **Possible food-safety cluster at {loc}:** {len(hits)} illness/safety complaints in 3 h – escalate to the GM and Food Safety Officer now.")
    for e in sorted(crit, key=lambda x: x["posted"])[:3]:
        lines.append(f"- {e['location']}: {e['summary']} → {e['action']}")
    worst = min(LOCATIONS, key=lambda l: health_score([e for e in evs if e["location"] == l]) or 101)
    lines.append(f"Lowest health score: **{worst}**. Priority: close critical incidents inside their 15-minute SLA, then review wait times.")
    return "\n\n".join(lines)


def brief(evs, now):
    inc = [e for e in evs if is_incident(e) and e["status"] != "Resolved"][:12]
    data = json.dumps([{"location": e["location"], "rating": e["rating"], "severity": e["severity"], "summary": e["summary"], "text": e["text"][:140]} for e in inc])
    out = llm(f"Write a crisp executive brief (max 130 words) for a restaurant chain's leadership from these open incidents: {data}. "
              "Start with the most urgent food-safety risk, mention any pattern across locations, and end with the top 3 actions.", 450)
    return out or fallback_brief(evs, now)


def guest_reply(e):
    out = llm(f"Write a warm, professional public reply (max 80 words) from the restaurant manager to this review. Apologise sincerely, "
              f"do NOT admit legal liability for illness, offer to resolve it offline via manager@flavorpulse.example.\nReview ({e['rating']}/5): {e['text']}", 250)
    if out:
        return out
    if e["food_safety"]:
        return ("Thank you for letting us know, and we are very sorry. Your wellbeing matters most to us. Our manager would like to speak with you "
                "personally and we are reviewing our kitchen procedures right now. Please contact manager@flavorpulse.example so we can help.")
    return ("Thank you for your honest feedback – we're sorry we missed the mark. We've shared your comments with the team and would love "
            "the chance to make it right. Please reach out to manager@flavorpulse.example.")


# ============================ 4. UI ============================
CSS = """<style>
.stApp{background:linear-gradient(165deg,#fff6ee 0%,#f7f9ff 55%,#eefaf4 100%)}
.hero{background:linear-gradient(90deg,#ff6b35,#ff3d71);color:#fff;border-radius:20px;padding:18px 24px;margin-bottom:14px}
.hero h1{margin:0;color:#fff;font-size:1.9rem}.hero p{margin:4px 0 0;opacity:.95}
.loc{background:#fff;border-radius:16px;padding:14px;box-shadow:0 4px 14px rgba(255,107,53,.14);text-align:center}
.loc b{font-size:2rem}.loc span{color:#6b7280;font-size:.85rem}
div.stButton>button{border-radius:12px;font-weight:700}
</style>"""


def ack(i):
    next(e for e in st.session_state.events if e["id"] == i)["status"] = "Acknowledged"


def resolve(i):
    next(e for e in st.session_state.events if e["id"] == i)["status"] = "Resolved"


def sidebar():
    with st.sidebar:
        st.header("⚙️ Control panel")
        st.toggle("📡 Live stream (new review every ~8 s)", key="streaming")
        st.button("➕ Add one live event now", on_click=stream_one)
        st.divider()
        st.multiselect("Locations", LOCATIONS, default=LOCATIONS, key="locs")
        st.selectbox("Time window", ["Last 3 hours", "Last 6 hours", "Last 24 hours"], index=2, key="window")
        st.divider()
        st.checkbox("Use Generative AI for new events", value=True, key="use_ai")
        st.text_input("Anthropic API key (optional)", type="password", key="api_key")
        if get_client():
            st.success(f"🧠 Claude connected ({MODEL})")
        else:
            st.info("🧩 Offline rule-based NLP engine active. Add an API key for Generative AI.")
        if st.session_state.get("llm_error"):
            st.warning("AI call failed – fell back to rules: " + st.session_state.llm_error)
        st.button("🔄 Reset demo data", on_click=seed_demo)


def dashboard():
    now = datetime.now()
    if st.session_state.get("streaming") and (now - st.session_state.last_stream).total_seconds() >= 7:
        stream_one()
    hours = {"Last 3 hours": 3, "Last 6 hours": 6, "Last 24 hours": 24}[st.session_state.get("window", "Last 24 hours")]
    locs = st.session_state.get("locs") or LOCATIONS
    evs = [e for e in st.session_state.events if e["location"] in locs and e["posted"] >= now - timedelta(hours=hours)]
    inc = [e for e in evs if is_incident(e)]
    openinc = [e for e in inc if e["status"] != "Resolved"]
    tabs = st.tabs(["📡 Command Center", "🚨 Incident Board", "🧠 AI Briefing & Replies", "➕ Submit Feedback", "🏗️ Architecture"])

    # ---- Command center ----
    with tabs[0]:
        for loc, hits in clusters(evs, now):
            st.error(f"🚨 **POSSIBLE FOOD-SAFETY CLUSTER – {loc}:** {len(hits)} safety/illness complaints within 3 hours. Escalate to the General Manager and Food Safety Officer immediately.")
        for e in [x for x in openinc if x["severity"] == "Critical"][:3]:
            st.error(f"{ICON['Critical']} **CRITICAL – {e['location']}:** {e['summary']} · owner: {e['owner']} · {sla_text(e, now)}")
        ttd = sorted((e["detected"] - e["posted"]).total_seconds() / 60 for e in evs)
        k = st.columns(5)
        k[0].metric("Reviews", len(evs))
        k[1].metric("Avg rating", f"{sum(e['rating'] for e in evs) / max(1, len(evs)):.1f} ★")
        k[2].metric("Negative", f"{100 * sum(e['sentiment'] == 'Negative' for e in evs) / max(1, len(evs)):.0f}%")
        k[3].metric("Open incidents", len(openinc))
        k[4].metric("Median time-to-detect", f"{ttd[len(ttd) // 2]:.1f} min" if ttd else "–", help="Before FlavorPulse: 24–48 hours")
        cols = st.columns(len(locs))
        for c, loc in zip(cols, locs):
            le = [e for e in evs if e["location"] == loc]
            hs = health_score(le)
            colr = "#e03131" if hs is not None and hs < 60 else "#f08c00" if hs is not None and hs < 80 else "#2f9e44"
            c.markdown(f'<div class="loc"><span>{loc}</span><br><b style="color:{colr}">{"–" if hs is None else hs}</b><br><span>health score · {len(le)} reviews · '
                       f'{sum(is_incident(e) and e["status"] != "Resolved" for e in le)} open incidents</span></div>', unsafe_allow_html=True)
        if evs:
            df = pd.DataFrame(evs)
            df["bucket"] = df["posted"].dt.floor("30min")
            sent = df.pivot_table(index="bucket", columns="sentiment", values="id", aggfunc="count", fill_value=0)
            sent = sent.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
            a, b = st.columns(2)
            a.caption("Sentiment over time (30-min buckets)")
            a.bar_chart(sent, color=["#2fbf71", "#b0b7c3", "#e03131"])
            neg = df[df["sentiment"] == "Negative"].explode("categories")["categories"].value_counts()
            b.caption("What guests complain about")
            if len(neg):
                b.bar_chart(neg)
            else:
                b.info("No negative reviews in this window 🎉")
        st.subheader("Live feed")
        for e in sorted(evs, key=lambda x: x["detected"], reverse=True)[:8]:
            st.markdown(f"{ICON[e['severity']]} **{e['location']}** · {e['source']} · {'★' * e['rating']}{'☆' * (5 - e['rating'])} · "
                        f"_{e['sentiment']}_ · {e['detected']:%H:%M:%S}  \n&nbsp;&nbsp;&nbsp;“{e['text']}”")

    # ---- Incident board ----
    with tabs[1]:
        stt = st.radio("Show", ["Open", "Acknowledged", "Resolved", "All"], horizontal=True, key="inc_status")
        rows = [e for e in inc if stt == "All" or e["status"] == stt]
        rows.sort(key=lambda e: (-RANK[e["severity"]], e["detected"]))
        if not rows:
            st.success("No incidents in this view. 👏")
        for e in rows:
            with st.container(border=True):
                st.markdown(f"### {ICON[e['severity']]} {e['severity']} · {e['location']} · {e['source']} · {e['rating']}★")
                st.markdown(f"“{e['text']}”")
                st.markdown(f"**AI summary:** {e['summary']} · **Categories:** {', '.join(e['categories'])}"
                            + (f" · **Evidence:** {', '.join(e['evidence'])}" if e["evidence"] else "") + f"  \n_Engine: {e['engine']}_")
                st.markdown(f"**Recommended action:** {e['action']}  \n**Owner:** {e['owner']} · **Status:** {e['status']} · {sla_text(e, now)}")
                b1, b2, _ = st.columns([1, 1, 3])
                b1.button("✋ Acknowledge", key=f"ack{e['id']}", on_click=ack, args=(e["id"],), disabled=e["status"] != "Open")
                b2.button("✅ Resolve", key=f"res{e['id']}", on_click=resolve, args=(e["id"],), disabled=e["status"] == "Resolved")
        if inc:
            exp = pd.DataFrame(inc)[["id", "posted", "detected", "location", "source", "rating", "severity", "summary", "owner", "status", "text"]]
            st.download_button("⬇️ Export incidents (CSV)", exp.to_csv(index=False), "flavorpulse_incidents.csv", "text/csv")

    # ---- AI tab ----
    with tabs[2]:
        st.subheader("Executive briefing")
        if st.button("✨ Generate briefing"):
            st.session_state.brief = brief(evs, now)
        if st.session_state.get("brief"):
            st.markdown(st.session_state.brief)
        st.subheader("Draft a guest reply")
        neg = [e for e in sorted(evs, key=lambda x: x["posted"], reverse=True) if e["sentiment"] == "Negative"][:15]
        if neg:
            pick = st.selectbox("Choose a negative review", neg, format_func=lambda e: f"{e['location']} · {e['rating']}★ · {e['text'][:60]}…")
            if st.button("✍️ Draft reply"):
                st.session_state.reply = guest_reply(pick)
            if st.session_state.get("reply"):
                st.text_area("Reply (edit before sending)", st.session_state.reply, height=150)
        else:
            st.info("No negative reviews to reply to.")

    # ---- Submit ----
    with tabs[3]:
        st.subheader("Simulate a new review")
        with st.form("fb", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            loc = c1.selectbox("Location", LOCATIONS)
            src = c2.selectbox("Source", SOURCES)
            rating = c3.slider("Rating", 1, 5, 3)
            text = st.text_area("Guest review", placeholder="e.g. My food was cold and we waited 40 minutes…")
            go = st.form_submit_button("Analyse & publish")
        if go and text.strip():
            e = ingest(loc, src, rating, text.strip(), use_ai=st.session_state.get("use_ai", True))
            st.success(f"Analysed in real time → {ICON[e['severity']]} severity **{e['severity']}**, sentiment **{e['sentiment']}**, owner **{e['owner']}**")
            st.json({k: e[k] for k in ("sentiment", "score", "categories", "severity", "food_safety", "evidence", "summary", "action", "engine")})
        elif go:
            st.warning("Please type a review first.")

    # ---- Architecture ----
    with tabs[4]:
        st.markdown("""
### Event-driven design (no traditional database)
```
 Review sources            1. INGEST              2. ENRICH (AI)           3. DECIDE                 4. ACT
 Google / Yelp / App  -->  event dict in     -->  Claude LLM (or rule  --> severity + SLA clock  --> live alerts, incident
 survey / Instagram        memory (session)       NLP fallback):            cluster detection        board, owner routing,
 (stream or form)                                 sentiment, category,      (>=2 safety events       AI brief + guest reply
                                                  food-safety flag          / location / 3 h)        CSV export
```
**Why no database?** The prototype keeps the event list in application memory (`st.session_state`), and exports CSV for audit. That is enough for a live-ops screen; production would swap the in-memory list for a message queue/webhook endpoint.

**Low-code stack:** Streamlit (UI + app server) · Python rules engine · Claude (Generative AI, optional) · pandas.

**Business impact:** detection time drops from 24–48 hours to ~1–2 minutes; critical food-safety issues get a 15-minute SLA with a named owner; clusters across a location are flagged automatically.

**Known limits (be honest in the demo):** data disappears on refresh/restart; the stream is simulated; keyword rules can miss unusual phrasing (the LLM layer helps); real review platforms need their API/webhook access.
""")


def main():
    st.set_page_config(page_title="FlavorPulse", page_icon="🍽️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    if "events" not in st.session_state:
        seed_demo()
    st.markdown('<div class="hero"><h1>🍽️ FlavorPulse Live Intelligence</h1><p>Guest feedback → AI analysis → live operational alerts, in minutes instead of days.</p></div>', unsafe_allow_html=True)
    sidebar()
    if hasattr(st, "fragment"):
        st.fragment(run_every=8 if st.session_state.get("streaming") else None)(dashboard)()
    else:
        dashboard()


if __name__ == "__main__":
    main()
