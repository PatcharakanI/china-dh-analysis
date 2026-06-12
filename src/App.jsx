import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

const DATA_FILES = {
  framingYear: "framing_scores_by_year.json",
  framingDecade: "framing_scores_by_decade.json",
  cooccurrenceYear: "economic_security_cooccurrence_by_year.json",
  cooccurrenceDecade: "economic_security_cooccurrence_by_decade.json",
  engagementShiftDecade: "engagement_competition_shift_by_decade.json",
  technologyBridge: "technology_bridge_by_decade.json",
  semanticPairs: "semantic_similarity_pairs_by_decade.json",
  keywords: "top_keywords_by_decade.json",
  articles: "representative_articles_for_close_reading.json",
  events: "historical_events.json",
  sectionFraming: "section_framing_by_decade.json",
  newsDeskFraming: "news_desk_framing_by_decade.json",
  balanceSummary: "balanced_sampling_summary.json",
  topWords: "top_40_words_by_decade.json",
  themeFraming: "theme_framing_by_decade.json",
  network: "cooccurrence_network.json",
  wordNetwork: "word_cooccurrence_network_by_decade.json",
  references: "references.json",
};

const DECADES = ["1990s", "2000s", "2010s", "2020s"];
const FRAME_OPTIONS = [
  { key: "opportunity_score", label: "Opportunity", color: "#2b8a67" },
  { key: "threat_score", label: "Threat", color: "#b84949" },
  { key: "competition_score", label: "Competition", color: "#8a5b23" },
  { key: "cooperation_score", label: "Engagement", color: "#3b6ea8" },
];
const PAIR_OPTIONS = [
  "trade/security",
  "technology/security",
  "china/competition",
  "market/security",
  "china/trade",
];
const DICTIONARY_TERMS = {
  opportunity: {
    title: "Opportunity / Positive Economic Framing",
    definition: "Words that describe China through market growth, investment, reform, economic opening, and possible mutual benefit.",
    terms: [
      "opportunity", "opportunities", "growth", "growing", "market", "markets", "investment",
      "investors", "business", "cooperation", "partnership", "partner", "opening", "reform",
      "development", "prosperity", "integration", "globalization", "engagement", "agreement",
      "trade", "exports", "consumer", "innovation", "modernization", "benefit", "benefits", "potential",
    ],
  },
  threat: {
    title: "Threat / Negative Security Framing",
    definition: "Words that connect China to danger, national security, repression, military conflict, cyber risk, sanctions, or state coercion.",
    terms: [
      "threat", "threats", "risk", "risks", "danger", "dangerous", "security", "national_security",
      "military", "spy", "espionage", "surveillance", "coercion", "crackdown", "conflict", "war",
      "missile", "cyber", "cybersecurity", "sanctions", "blacklist", "repression", "censorship",
      "authoritarian", "aggression", "aggressive", "warning", "fear", "concern", "concerns",
    ],
  },
  competition: {
    title: "Competition / Rivalry Framing",
    definition: "Words that describe China through rivalry, strategic competition, tariffs, decoupling, confrontation, or pressure.",
    terms: [
      "competition", "competitor", "competitors", "rival", "rivalry", "strategic_competition",
      "race", "trade_war", "tariff", "tariffs", "decoupling", "containment", "dispute",
      "disputes", "confrontation", "pressure", "retaliation", "dominance", "challenge", "challenging",
    ],
  },
  cooperation: {
    title: "Cooperation / Engagement Framing",
    definition: "Words that frame China through diplomacy, partnership, talks, integration, reform, WTO-era engagement, and institutional relationship-building.",
    terms: [
      "cooperation", "cooperative", "partner", "partnership", "engagement", "dialogue", "talks",
      "summit", "agreement", "diplomacy", "diplomatic", "relations", "ties", "integration",
      "wto", "world_trade_organization", "reform", "opening",
    ],
  },
};
const HEATMAP_COLORS = ["#ffffcc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#2c7fb8"];

function heatmapColor(value, maxValue) {
  const numericValue = Number(value || 0);
  const safeMax = Math.max(Number(maxValue || 0), 1);
  const index = Math.min(HEATMAP_COLORS.length - 1, Math.floor((numericValue / safeMax) * HEATMAP_COLORS.length));
  return HEATMAP_COLORS[index];
}

function heatmapCellStyle(value, maxValue) {
  const ratio = Number(value || 0) / Math.max(Number(maxValue || 0), 1);
  return {
    background: heatmapColor(value, maxValue),
    color: ratio > 0.72 ? "#fff" : "#102b3c",
  };
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(digits);
}

function readableLabel(value) {
  const labels = {
    economy_trade: "Economic / Trade",
    military_security: "Military / Security",
    technology_industry: "Technology / Industry",
    public_health_covid: "Public Health / COVID",
    human_rights_values: "Human Rights / Values",
    diplomacy_engagement: "Diplomacy / Engagement",
    competition_rivalry: "Competition / Rivalry",
    territory_sovereignty: "Territory / Sovereignty",
    domestic_politics: "Domestic Politics",
  };
  return labels[value] || String(value).replaceAll("_", " ");
}

function dataUrl(file) {
  return `${import.meta.env.BASE_URL}data/${file}`;
}

async function loadJson(file) {
  const response = await fetch(dataUrl(file));
  if (!response.ok) throw new Error(`Could not load ${file}`);
  return response.json();
}

function useProjectData() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all(
      Object.entries(DATA_FILES).map(async ([key, file]) => [key, await loadJson(file)])
    )
      .then((entries) => setData(Object.fromEntries(entries)))
      .catch((err) => setError(err.message));
  }, []);

  return { data, error };
}

function Card({ title, children }) {
  return (
    <aside className="card">
      <h3>{title}</h3>
      <div>{children}</div>
    </aside>
  );
}

function Explainer({ measures, read, caution }) {
  return (
    <div className="explainer">
      <h4>What this graph measures</h4>
      <p>{measures}</p>
      <h4>How to read it</h4>
      <p>{read}</p>
      <h4>What it does not prove</h4>
      <p>{caution}</p>
    </div>
  );
}

function EventMarkers({ events, y = "top" }) {
  return events.map((event) => (
    <div className={`event-marker ${y}`} key={`${event.year}-${event.event}`}>
      <span>{event.year}</span>
      <small>{event.event}</small>
    </div>
  ));
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip">
      <strong>{label}</strong>
      {payload.map((item) => (
        <div key={item.dataKey} style={{ color: item.color }}>
          {item.name}: {formatNumber(item.value)}
        </div>
      ))}
    </div>
  );
}

function Landing({ balanceSummary, onResult, onAbout }) {
  const target = balanceSummary?.[0]?.target_articles_per_decade;
  return (
    <section className="hero" id="about">
      <div className="hero-copy">
        <p className="eyebrow">Digital History Project</p>
        <h1>From Engagement to Competition</h1>
        <p className="hero-subtitle">How has the language surrounding China in New York Times reporting shifted over the past three decades?</p>
        <p className="lede">
          This project asks when China stopped appearing mainly as an opportunity of globalization
          and began appearing through the language of security, technology, rivalry, and strategic competition.
        </p>
        <div className="hero-actions">
          <button onClick={onResult}>Explore Results</button>
          <button onClick={onAbout} className="secondary">Project Argument</button>
        </div>
      </div>
      <div className="hero-panel">
        <img src={`${import.meta.env.BASE_URL}assets/hero-discourse.png`} alt="Editorial collage of U.S.-China discourse, trade routes, newsprint, and semiconductor motifs" />
        <span>Balanced Corpus</span>
        <strong>{target ? `${target} clean articles per decade` : "Balanced sample"}</strong>
        <p>
          After extracting data, I used quality-filtered NYT API records to build a balanced corpus with same number of articles for each decade.
        </p>
      </div>
    </section>
  );
}

function AboutPage({ framingDecade = [], cooccurrenceDecade = [], themeFraming = [], balanceSummary = [] }) {
  const target = balanceSummary?.[0]?.target_articles_per_decade;
  const net1990s = metricFor(framingDecade, "1990s", "net_framing_score");
  const net2020s = metricFor(framingDecade, "2020s", "net_framing_score");
  const overlap1990s = metricFor(cooccurrenceDecade, "1990s", "percent_with_both_economic_and_security");
  const overlap2020s = metricFor(cooccurrenceDecade, "2020s", "percent_with_both_economic_and_security");
  const top2020Themes = themeFraming
    .filter((row) => row.decade === "2020s")
    .sort((a, b) => Number(b.percent_articles || 0) - Number(a.percent_articles || 0))
    .slice(0, 3)
    .map((row) => `${readableLabel(row.theme)} (${formatNumber(row.percent_articles, 0)}%)`)
    .join(", ");

  return (
    <main className="page">
      <section className="section about-page">
        <div className="section-heading">
          <p className="eyebrow">About The Project</p>
          <h2>From Economic Integration To Economic Security</h2>
          <p className="section-note">
            This is a digital history project about how a major American newspaper, New York Times, made sense of China
            during the transition from globalization to strategic competition.
          </p>
        </div>
        <div className="about-grid">
          <Card title="Motivation">
            <p>
              When I was younger, I was never someone who paid much attention to international politics or economic news. As a result, I often could not understand why my everyday life changed the way it did—why imported goods suddenly became more expensive, or why China's influence in the global economy seemed to expand so rapidly over the past two decades. As I became more aware of international events, I realized that many of these changes were connected to the evolving relationship between the United States and China. Through what I learned in this Digital History class, I became interested in exploring the economic and political relationship between these two powerful countries and how it has been represented in public discourse.  
            </p>
          </Card>
          <Card title="Background">
            <p>
              Today, the United States and China are widely viewed as strategic competitors. Yet during the era of globalization in the 1990s and early 2000s, economic integration was often associated with cooperation, reform, and mutual benefit. I became interested in investigating when this narrative began to change and what events appeared alongside the growing tensions between the two countries.
            </p>
            <p>
              This project asks whether the changing relationship can be explained solely by economic competition and political differences, or whether other factors also contributed to the shift. More specifically, it examines whether economic issues increasingly became linked to ideas of security, competition, and rivalry. Rather than focusing only on political events, the project investigates how these transformations were reflected in media language and public discourse.
            </p>
          </Card>
          <Card title="Main Question">
            <p>
              This project asks whether the changing relationship can be explained solely by economic competition and political differences, or whether other factors also contributed to the shift. More specifically, it examines whether economic issues increasingly became linked to ideas of security, competition, and rivalry. Rather than focusing only on political events, the project investigates how these transformations were reflected in media language and public discourse.
            </p>
          </Card>
          <Card title="Short Answer From The Project Result">
            <p>
              The language surrounding China appears to shift from an engagement-centered vocabulary of markets, reform, trade, and integration toward a vocabulary in which economic issues are more frequently connected to security, technology, rivalry, and competition. The shift is gradual, but it becomes increasingly visible during the 2010s and is most apparent in the 2020s.
            </p>
          </Card>
          
          
          <Card title="Historical Argument">
            <p>
              The project does not argue simply that China came to be described more negatively. Instead, it tests a more specific historical transformation: the securitization of economic discourse. China shifted from a story centered on engagement, reform, and market opportunity toward one in which economic issues became increasingly tied to national security, technological competition, and strategic rivalry.
            </p>
          </Card>
          <Card title="Digital Method">
            <p>
              Using the New York Times Article Search API, I collected article metadata and textual summaries from 1990 to 2025. After filtering low-information records and balancing the corpus by decade, I analyzed the data using framing dictionaries, semantic similarity measures, metadata categories, co-occurrence analysis, and representative articles selected for close reading.
            </p>
          </Card>
          <Card title="How To Read The Evidence">
            <p>
              The visualizations are not a substitute for interpretation. Instead, they provide clues about historical change. They show where economic, security, technology, opportunity, and rivalry vocabularies rise, overlap, or move closer together over time. Historical interpretation comes from connecting these patterns to major events and examining representative articles in their broader context.
            </p>
          </Card>
          <Card title="Summary Of Measured Patterns">
            <p>
              In the balanced corpus of 389 clean articles per decade, the net framing score declines from 11.9 in the 1990s to 2.7 in the 2020s. Economic-security overlap increases from 8.7% to 10.0%. By the 2020s, some of the most prominent thematic categories include Domestic Politics (53%), Economy and Trade (40%), and Military and Security (29%). Taken together, these patterns suggest a growing connection between economic and security-oriented language.
            </p>
          </Card>
          <Card title="Conclusion">
            <p>
              The results suggest a transformation in framing rather than a simple increase or decrease in attention to China. Economic vocabulary remains important throughout the entire period, but by the 2020s it is more likely to appear alongside terms associated with national security, technological rivalry, public health, and strategic competition.
            </p>
            <p>
              The data does not identify a single catalyst for changes in U.S.-China relations. Instead, it points to a gradual and layered shift. Trade and market language remain visible, but the meaning attached to economic connection appears to change. The 2010s emerge as a transition period, while the 2020s provide the clearest evidence that economic topics are increasingly discussed through the language of security, competition, technology, and vulnerability. This suggests that economic connections with China were increasingly discussed as matters of security and strategic competition rather than simply opportunities for trade and cooperation.
            </p>
          </Card>
          <Card title="Limitations And Next Steps">
            <p>
              This corpus reflects the perspective of a single newspaper, is constrained by API search and sampling limitations, and relies primarily on article summaries rather than full-text articles. Future research could compare multiple newspapers, test alternative dictionaries, manually validate samples, and investigate whether similar patterns appear in media outside the United States.
            </p>
          </Card>
        </div>
      </section>
    </main>
  );
}


function metricFor(data, decade, key) {
  const row = data?.find((item) => item.decade === decade);
  return row ? row[key] : null;
}

function HistoricalNarrative({ framingDecade, cooccurrenceDecade }) {
  const net1990s = metricFor(framingDecade, "1990s", "net_framing_score");
  const net2020s = metricFor(framingDecade, "2020s", "net_framing_score");
  const overlap1990s = metricFor(cooccurrenceDecade, "1990s", "percent_with_both_economic_and_security");
  const overlap2020s = metricFor(cooccurrenceDecade, "2020s", "percent_with_both_economic_and_security");

  return (
    <section className="section narrative-section" id="narrative">
      <div className="section-heading">
        <p className="eyebrow">Historical Narrative</p>
        <h2>Not Just Negative Coverage, But A Change In What “Economy” Meant</h2>
        <p className="section-note">
          The project’s central claim is not that New York Times coverage simply became anti-China.
          The more interesting transformation is that economic topics increasingly appeared beside security,
          technology, and strategic rivalry language.
        </p>
      </div>
      <div className="narrative-grid">
        <article className="narrative-card primary">
          <span>Core argument</span>
          <h3>From engagement to economic security</h3>
          <p>
            In the 1990s and 2000s, China often appeared as part of globalization: trade, markets,
            investment, development, and reform. By the 2020s, those economic vocabularies had not disappeared,
            but they were more often entangled with terms such as security, competition, threat, sanctions,
            chips, and strategic rivalry.
          </p>
        </article>
        <article className="narrative-card">
          <span>Framing balance</span>
          <h3>{formatNumber(net1990s, 1)} → {formatNumber(net2020s, 1)}</h3>
          <p>
            The net framing score moves from clearly positive/opportunity-oriented in the 1990s toward
            a nearly balanced or negative frame in the 2020s. This suggests a shift in vocabulary, not a
            direct measurement of journalistic intent.
          </p>
        </article>
        <article className="narrative-card">
          <span>Economy-security overlap</span>
          <h3>{formatNumber(overlap1990s, 1)}% → {formatNumber(overlap2020s, 1)}%</h3>
          <p>
            The share of articles containing both economic and security language is one of the clearest
            indicators of securitization. It asks whether economic discussion and security discussion begin
            to occupy the same article space.
          </p>
        </article>
      </div>
    </section>
  );
}

function DecadeSummaries({ framingDecade, cooccurrenceDecade, technologyBridge, engagementShiftDecade }) {
  const decadeText = {
    "1990s": {
      title: "Engagement and uncertainty",
      body: "Coverage still carries Cold War afterlives, human-rights concerns, and diplomatic tension, but economic opportunity and engagement remain central frames. China is not yet normalized as the strategic competitor familiar today.",
    },
    "2000s": {
      title: "Globalization and WTO-era integration",
      body: "The 2000s show China more firmly embedded in global trade, markets, manufacturing, and development. The narrative is not free of concern, but economic integration remains a major way of understanding China.",
    },
    "2010s": {
      title: "Competition becomes harder to separate from economics",
      body: "This decade is a transition zone. Economic vocabulary remains strong, but technology, cyber, territorial disputes, and strategic competition become more visible. The data should be read as gradual change rather than a single sudden break.",
    },
    "2020s": {
      title: "Economic security and technological rivalry",
      body: "The 2020s show the clearest signs of securitization. Economic discussion increasingly overlaps with technology and security: chips, supply chains, export controls, Taiwan, and strategic competition become part of the economic story.",
    },
  };

  return (
    <section className="section decade-stories" id="decade-stories">
      <div className="section-heading">
        <p className="eyebrow">Decade Summaries</p>
        <h2>Four Periods In The Corpus</h2>
        <p className="section-note">
          These summaries translate the charts into a historical reading. The numbers are clues for interpretation,
          not automatic conclusions.
        </p>
      </div>
      <div className="decade-card-grid">
        {DECADES.map((decade) => {
          const framing = framingDecade.find((row) => row.decade === decade) || {};
          const overlap = cooccurrenceDecade.find((row) => row.decade === decade) || {};
          const tech = technologyBridge.find((row) => row.decade === decade) || {};
          const shift = engagementShiftDecade.find((row) => row.decade === decade) || {};
          return (
            <article className="decade-story-card" key={decade}>
              <span>{decade}</span>
              <h3>{decadeText[decade].title}</h3>
              <p>{decadeText[decade].body}</p>
              <dl>
                <div><dt>Net framing</dt><dd>{formatNumber(framing.net_framing_score, 1)}</dd></div>
                <div><dt>Economy + security overlap</dt><dd>{formatNumber(overlap.percent_with_both_economic_and_security, 1)}%</dd></div>
                <div><dt>Conflict prevalence</dt><dd>{formatNumber(shift.conflict_prevalence, 1)}%</dd></div>
                <div><dt>Tech + economy + security</dt><dd>{formatNumber(tech.percent_technology_economy_security, 1)}%</dd></div>
              </dl>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function TurningPointExplanations({ events, framingYear, cooccurrenceYear }) {
  const selected = [2001, 2008, 2012, 2018, 2020, 2022, 2023];
  const explanations = {
    2001: "China’s WTO accession is a key marker for the engagement/globalization framework. On the website, it should be read as a historical reference point for comparing later security-oriented vocabulary.",
    2008: "The financial crisis helps frame a period when confidence in globalization weakened and China’s economic power became more visible in global discussions.",
    2012: "Xi Jinping’s rise marks a useful political reference point for tracking later changes in governance, technology, and strategic competition language.",
    2018: "The U.S.-China trade war is a major test case for securitization because trade policy became openly connected to national strategy, tariffs, and rivalry.",
    2020: "COVID-19 intensified discussion of supply chains, public health, vulnerability, and geopolitical distrust. It can help explain why economic and security vocabularies overlap more strongly in the 2020s.",
    2022: "The CHIPS Act and semiconductor export controls make technology one of the clearest bridges between economic policy and security policy.",
    2023: "Outbound investment and chip-control debates show that economic connection itself could be framed as a security risk.",
  };

  const rows = selected.map((year) => {
    const event = events.find((item) => Number(item.year) === year);
    const frame = framingYear.find((item) => Number(item.year) === year) || {};
    const overlap = cooccurrenceYear.find((item) => Number(item.year) === year) || {};
    return { year, event: event?.event || "Historical reference point", frame, overlap };
  });

  return (
    <section className="section turning-points" id="turning-points">
      <div className="section-heading">
        <p className="eyebrow">Turning Points</p>
        <h2>Events That Help Interpret The Curves</h2>
        <p className="section-note">
          These events are not treated as automatic causes. They are guideposts for asking when changes in vocabulary
          become visible in the corpus.
        </p>
      </div>
      <div className="turning-point-list">
        {rows.map(({ year, event, frame, overlap }) => (
          <article className="turning-point-card" key={year}>
            <div className="turning-year">{year}</div>
            <div>
              <h3>{event}</h3>
              <p>{explanations[year]}</p>
              <ul>
                <li>Threat score: <strong>{formatNumber(frame.threat_score, 1)}</strong></li>
                <li>Opportunity score: <strong>{formatNumber(frame.opportunity_score, 1)}</strong></li>
                <li>Economy-security overlap: <strong>{formatNumber(overlap.percent_with_both_economic_and_security, 1)}%</strong></li>
              </ul>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function CloseReadingGuide({ articles }) {
  const desiredProfiles = [
    "high_economic_low_security",
    "high_security_low_economic",
    "high_economy_security_overlap",
    "high_technology_security_overlap",
  ];
  const selected = desiredProfiles
    .map((profile) => articles.find((article) => article.profile === profile && article.abstract))
    .filter(Boolean);

  return (
    <section className="section close-reading-guide" id="close-reading-guide">
      <div className="section-heading">
        <p className="eyebrow">Close Reading Examples</p>
        <h2>From Quantitative Pattern To Historical Interpretation</h2>
      </div>
      <div className="close-reading-grid">
        {selected.map((article) => (
          <article className="close-reading-feature" key={`${article.profile}-${article.url}`}>
            <span>{article.profile.replaceAll("_", " ")}</span>
            <h3>{article.headline}</h3>
            <p>{article.abstract}</p>
            <dl>
              <div><dt>Year</dt><dd>{article.year}</dd></div>
              <div><dt>Section</dt><dd>{article.section || "Unknown"}</dd></div>
              <div><dt>Economic score</dt><dd>{formatNumber(article.economic_score, 1)}</dd></div>
              <div><dt>Security score</dt><dd>{formatNumber(article.security_score, 1)}</dd></div>
            </dl>
            {article.url && <a href={article.url} target="_blank" rel="noreferrer">Read source article</a>}
          </article>
        ))}
      </div>
    </section>
  );
}

function TimelineDashboard({ framingYear, events }) {
  const [activeFrames, setActiveFrames] = useState(["opportunity_score", "threat_score"]);
  const [showEventLabels, setShowEventLabels] = useState(false);
  const visibleFrames = FRAME_OPTIONS.filter((frame) => activeFrames.includes(frame.key));
  const highlightEvents = events.filter((event) =>
    [1999, 2001, 2008, 2012, 2018, 2019, 2020, 2022, 2023].includes(Number(event.year))
  );

  function toggleFrame(key) {
    setActiveFrames((current) =>
      current.includes(key) ? current.filter((item) => item !== key) : [...current, key]
    );
  }

  return (
    <section id="timeline" className="section">
      <div className="section-heading">
        <p className="eyebrow">Timeline Dashboard</p>
        <h2>Framing Scores Over Time</h2>
      </div>
      <div className="dashboard-grid">
        <div className="chart-shell">
          <div className="toggle-row">
            {FRAME_OPTIONS.map((frame) => (
              <button
                key={frame.key}
                className={activeFrames.includes(frame.key) ? "active" : ""}
                onClick={() => toggleFrame(frame.key)}
                style={{ "--accent": frame.color }}
              >
                {frame.label}
              </button>
            ))}
            <button className={showEventLabels ? "active" : ""} onClick={() => setShowEventLabels(!showEventLabels)}>
              {showEventLabels ? "Hide event labels" : "Show event labels"}
            </button>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={420}>
              <LineChart data={framingYear} margin={{ top: 16, right: 26, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dbe1e7" />
                <XAxis dataKey="year" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip content={<ChartTooltip />} />
                <Legend />
                {showEventLabels && highlightEvents.map((event) => (
                  <ReferenceLine
                    key={`${event.year}-${event.event}`}
                    x={Number(event.year)}
                    stroke="#d05a77"
                    strokeDasharray="4 4"
                    label={{ value: `${event.year}`, angle: -90, position: "top", fill: "#8a2944", fontSize: 11 }}
                  />
                ))}
                {visibleFrames.map((frame) => (
                  <Line
                    key={frame.key}
                    type="monotone"
                    dataKey={frame.key}
                    name={frame.label}
                    stroke={frame.color}
                    strokeWidth={1.7}
                    dot={false}
                    activeDot={{ r: 5 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
            <div className="event-strip">
              <EventMarkers events={highlightEvents} />
            </div>
          </div>
        </div>
        <Card title="How To Read This Quickly">
          <Explainer
            measures="The yearly average number of frame-dictionary words per 1,000 article tokens."
            read="A rising threat line means that, in that year, the sampled articles used more words from the threat dictionary. Start with opportunity vs. threat, then turn on competition or engagement to compare specific frames."
            caution="It does not prove what journalists felt or intended. It only shows changes in vocabulary frequency."
          />
        </Card>
      </div>
    </section>
  );
}

function DecadeFramingComparison({ framingDecade }) {
  const data = framingDecade.map((row) => ({
    decade: row.decade,
    opportunity: row.opportunity_score,
    threat: row.threat_score,
    positive: row.positive_framing_score,
    negative: row.negative_framing_score,
  }));

  return (
    <section id="decade-framing" className="section">
      <div className="section-heading">
        <p className="eyebrow">Decade Comparison</p>
        <h2>Positive / Opportunity vs Negative / Threat Framing</h2>
      </div>
      <div className="dashboard-grid">
        <div className="chart-shell">
          <ResponsiveContainer width="100%" height={430}>
            <BarChart data={data} margin={{ top: 18, right: 24, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ead6df" />
              <XAxis dataKey="decade" />
              <YAxis />
              <Tooltip content={<ChartTooltip />} />
              <Legend />
              <Bar dataKey="opportunity" name="Opportunity score" fill="#2b8a67" />
              <Bar dataKey="threat" name="Threat score" fill="#d05a77" />
              <Bar dataKey="positive" name="Positive framing" fill="#77b255" />
              <Bar dataKey="negative" name="Negative framing" fill="#b2224c" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <Card title="Why This Chart Is Here">
          <Explainer
            measures="Average dictionary-term mentions per 1,000 article tokens, grouped by decade."
            read="Bars make decade comparison easier than the yearly line chart. Positive framing combines opportunity and cooperation terms. Negative framing combines threat and competition terms."
            caution="The bars are dictionary indicators, not sentiment labels assigned by a human reader."
          />
        </Card>
      </div>
    </section>
  );
}

function Securitization({ cooccurrenceYear, cooccurrenceDecade }) {
  return (
    <section id="securitization" className="section">
      <div className="section-heading">
        <p className="eyebrow">Economic Securitization</p>
        <h2>When Economic Coverage Contains Security Language</h2>
      </div>
      <div className="dashboard-grid">
        <div className="chart-shell">
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={cooccurrenceYear} margin={{ top: 16, right: 24, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbe1e7" />
              <XAxis dataKey="year" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip content={<ChartTooltip />} />
              <Legend />
              <Line type="monotone" dataKey="percent_with_economic_terms" name="Economic articles" stroke="#2b8a67" strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="percent_with_security_terms" name="Security articles" stroke="#b84949" strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="percent_with_both_economic_and_security" name="Economic + security overlap" stroke="#2f5f7f" strokeWidth={2.5} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <Card title="Metric Definition">
          <Explainer
            measures="The share of articles containing economic language, security language, or both in the same article."
            read="The decade cards show the overlap rate: the percent of articles in that decade that contain at least one economic term and at least one security term."
            caution="A co-occurrence does not prove the article argues that economics is security. It shows that both vocabularies appear together."
          />
          <dl className="metric-list">
            <dt>Overlap</dt>
            <dd>Percent of articles with both economic and security terms.</dd>
            <dt>Security-per-economic ratio</dt>
            <dd>Security mentions divided by economic mentions plus one.</dd>
          </dl>
        </Card>
      </div>
      <div className="mini-bars">
        {cooccurrenceDecade.map((row) => (
          <div key={row.decade} className="mini-bar">
            <span>{row.decade}</span>
            <strong>{formatNumber(row.percent_with_both_economic_and_security)}%</strong>
            <em>articles with both economic and security language</em>
            <div>
              <i style={{ width: `${Math.min(row.percent_with_both_economic_and_security * 4, 100)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function SemanticShift({ semanticPairs }) {
  const pairNames = useMemo(() => {
    const names = semanticPairs.map((row) => `${row.term_a}/${row.term_b}`);
    return Array.from(new Set(names)).filter((name) => PAIR_OPTIONS.includes(name));
  }, [semanticPairs]);
  const [selectedPairs, setSelectedPairs] = useState(["technology/security"]);
  const chartData = DECADES.map((decade) => {
    const item = { decade };
    selectedPairs.forEach((pair) => {
      const [termA, termB] = pair.split("/");
      const found = semanticPairs.find(
        (row) => row.decade === decade && row.term_a === termA && row.term_b === termB
      );
      item[pair] = found?.similarity ?? null;
    });
    return item;
  });

  function togglePair(pair) {
    setSelectedPairs((current) =>
      current.includes(pair) ? current.filter((item) => item !== pair) : [...current, pair]
    );
  }

  return (
    <section id="semantic" className="section">
      <div className="section-heading">
        <p className="eyebrow">Semantic Shift</p>
        <h2>Embedding Similarity Across Decades</h2>
      </div>
      <div className="dashboard-grid">
        <div className="chart-shell">
          <div className="toggle-row wrap">
            {pairNames.map((pair) => (
              <button key={pair} className={selectedPairs.includes(pair) ? "active" : ""} onClick={() => togglePair(pair)}>
                {pair}
              </button>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData} margin={{ top: 16, right: 24, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbe1e7" />
              <XAxis dataKey="decade" />
              <YAxis domain={[0, 1]} />
              <Tooltip content={<ChartTooltip />} />
              <Legend />
              {selectedPairs.map((pair, index) => (
                <Line key={pair} type="monotone" dataKey={pair} stroke={["#2f5f7f", "#b84949", "#2b8a67", "#8a5b23"][index % 4]} strokeWidth={2.5} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <Card title="Cosine Similarity">
          <Explainer
            measures="Cosine similarity between two word vectors trained separately for each decade."
            read="A higher value means the two words appeared in more similar surrounding contexts in that decade. For example, technology/security asks whether technology words were used in contexts closer to security words."
            caution="It does not prove the words mean the same thing, and missing vocabulary is left blank rather than estimated."
          />
        </Card>
      </div>
    </section>
  );
}

function FrequencyAndThemes({ topWords, themeFraming }) {
  const [decade, setDecade] = useState("2020s");
  const wordRows = topWords.filter((row) => row.decade === decade).slice(0, 24);
  const themes = Array.from(new Set(themeFraming.map((row) => row.theme)));
  const maxPercent = Math.max(...themeFraming.map((row) => Number(row.percent_articles || 0)), 1);

  return (
    <section id="frequency" className="section">
      <div className="section-heading">
        <p className="eyebrow">Vocabulary</p>
        <h2>What Words And Frames Are Most Visible?</h2>
      </div>
      <div className="control-bar">
        <select value={decade} onChange={(event) => setDecade(event.target.value)}>
          {DECADES.map((item) => <option key={item}>{item}</option>)}
        </select>
      </div>
      <div className="vocabulary-grid">
        <div className="chart-shell tall-chart word-frequency-panel">
          <ResponsiveContainer width="100%" height={620}>
            <BarChart data={wordRows} layout="vertical" margin={{ top: 10, right: 22, bottom: 8, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ead6df" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="word" tickFormatter={readableLabel} width={118} interval={0} tick={{ fontSize: 13 }} />
              <Tooltip content={<ChartTooltip />} />
              <Bar dataKey="per_1000_words" name="Mentions per 1,000 words" fill="#00a6ca" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-shell theme-heatmap-shell coverage-frame-panel">
          <h3>Coverage Frames by Decade</h3>
          <div className="theme-heatmap">
            <div className="theme-heatmap-header">
              <span>Theme</span>
              {DECADES.map((item) => <strong key={item}>{item}</strong>)}
            </div>
            {themes.map((theme) => (
              <div className="theme-heatmap-row" key={theme}>
                <strong>{readableLabel(theme)}</strong>
                {DECADES.map((item) => {
                  const found = themeFraming.find((row) => row.theme === theme && row.decade === item);
                  const value = Number(found?.percent_articles || 0);
                  return (
                    <span key={item} style={heatmapCellStyle(value, maxPercent)}>
                      {formatNumber(value, 0)}%
                    </span>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
      <Card title="What This Presents">
        <Explainer
          measures="The left chart shows frequent individual words. The heatmap shows the percent of articles in each decade that contain at least one term from each theme dictionary."
          read="Darker heatmap cells mean a larger share of articles in that decade touched that theme. The word bars show normalized frequency, not article share."
          caution="Theme dictionaries simplify complex language. They show where to look, not a complete interpretation."
        />
      </Card>
    </section>
  );
}


function FramingByMetadata({ sectionFraming, newsDeskFraming }) {
  const [mode, setMode] = useState("section");
  const [metric, setMetric] = useState("rivalry_score");
  const source = mode === "section" ? sectionFraming : newsDeskFraming;
  const labelKey = mode === "section" ? "section" : "news_desk";
  const topLabels = useMemo(() => {
    const totals = new Map();
    source.forEach((row) => totals.set(row[labelKey], (totals.get(row[labelKey]) || 0) + Number(row.articles || 0)));
    return Array.from(totals.entries()).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([label]) => label);
  }, [source, labelKey]);
  const heatRows = source.filter((row) => topLabels.includes(row[labelKey]));
  const maxMetric = Math.max(...heatRows.map((row) => Number(row[metric] || 0)), 1);

  return (
    <section id="metadata" className="section">
      <div className="section-heading">
        <p className="eyebrow">Metadata View</p>
        <h2>Where China Coverage Appears</h2>
      </div>
      <p className="section-note">
        This view groups articles by NYT section or news desk and shows the selected
        average framing score. It helps separate changes in language from changes in
        where China stories were published.
      </p>
      <div className="control-bar">
        <button className={mode === "section" ? "active" : ""} onClick={() => setMode("section")}>Section</button>
        <button className={mode === "news_desk" ? "active" : ""} onClick={() => setMode("news_desk")}>News Desk</button>
        <select value={metric} onChange={(event) => setMetric(event.target.value)}>
          <option value="rivalry_score">Rivalry score</option>
          <option value="economic_score">Economic score</option>
          <option value="security_score">Security score</option>
          <option value="technology_score">Technology score</option>
          <option value="net_framing_score">Net framing score</option>
        </select>
      </div>
      <div className="heatmap">
        <div className="heatmap-header">
          <span />
          {DECADES.map((decade) => <strong key={decade}>{decade}</strong>)}
        </div>
        {topLabels.map((label) => (
          <div className="heatmap-row" key={label}>
            <strong>{label}</strong>
            {DECADES.map((decade) => {
              const found = heatRows.find((row) => row[labelKey] === label && row.decade === decade);
              const value = Number(found?.[metric] || 0);
              return <span key={decade} style={heatmapCellStyle(value, maxMetric)}>{formatNumber(value, 1)}</span>;
            })}
          </div>
        ))}
      </div>
      <Card title="Reading The Heatmap">
        <p>
          The table shows the top metadata categories by article count. Darker cells
          indicate higher average scores for the selected metric within a decade.
        </p>
      </Card>
    </section>
  );
}

function KeywordsAndCloseReading({ keywords, articles }) {
  const [decade, setDecade] = useState("2020s");
  const [profile, setProfile] = useState("all");
  const [section, setSection] = useState("all");
  const keywordRows = keywords.filter((row) => row.decade === decade).slice(0, 20);
  const sections = Array.from(new Set(articles.map((row) => row.section).filter(Boolean))).sort();
  const articleRows = articles.filter((row) => {
    const articleDecade = `${Math.floor(Number(row.year) / 10) * 10}s`;
    return (
      articleDecade === decade &&
      (profile === "all" || row.profile === profile) &&
      (section === "all" || row.section === section)
    );
  });

  return (
    <section id="close-reading" className="section">
      <div className="section-heading">
        <p className="eyebrow">Keywords And Close Reading</p>
        <h2>From Pattern To Article</h2>
      </div>
      <div className="control-bar">
        <select value={decade} onChange={(event) => setDecade(event.target.value)}>
          {DECADES.map((item) => <option key={item}>{item}</option>)}
        </select>
        <select value={profile} onChange={(event) => setProfile(event.target.value)}>
          <option value="all">All frame types</option>
          <option value="high_economic_low_security">High economic, low security</option>
          <option value="high_security_low_economic">High security, low economic</option>
          <option value="high_economy_security_overlap">High economy + security overlap</option>
          <option value="high_technology_security_overlap">High technology + security overlap</option>
        </select>
        <select value={section} onChange={(event) => setSection(event.target.value)}>
          <option value="all">All sections</option>
          {sections.map((item) => <option key={item}>{item}</option>)}
        </select>
      </div>
      <div className="split">
        <div className="table-card">
          <h3>Top TF-IDF Keywords</h3>
          <table>
            <thead><tr><th>Rank</th><th>Keyword</th><th>Score</th></tr></thead>
            <tbody>
              {keywordRows.map((row) => (
                <tr key={`${row.decade}-${row.rank}-${row.keyword}`}>
                  <td>{row.rank}</td>
                  <td>{String(row.keyword).replaceAll("_", " ")}</td>
                  <td>{formatNumber(row.tfidf, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="article-list">
          {articleRows.length === 0 && <p className="empty">No representative articles match these filters.</p>}
          {articleRows.map((article) => (
            <article key={`${article.profile}-${article.url}`} className="article-card">
              <span>{article.profile.replaceAll("_", " ")}</span>
              <h3>{article.headline}</h3>
              <p>{article.abstract || "No abstract available."}</p>
              <dl>
                <dt>Date</dt><dd>{String(article.date).slice(0, 10)}</dd>
                <dt>Section</dt><dd>{article.section || "Unknown"}</dd>
                <dt>Keywords</dt><dd>{article.keywords || "None listed"}</dd>
              </dl>
              {article.url && <a href={article.url} target="_blank" rel="noreferrer">Open NYT article</a>}
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function MethodLimitations() {
  return (
    <section id="method" className="section method">
      <div className="section-heading">
        <p className="eyebrow">Method And Limitations</p>
        <h2>How The Evidence Was Made</h2>
      </div>
      <div className="method-grid">
        <Card title="Data Source">
          <p>
            The corpus comes from the New York Times Article Search API. The analysis uses
            headline, abstract, snippet, lead paragraph when available, and NYT keywords.
          </p>
        </Card>
        <Card title="Balanced Corpus">
          <p>
            The site uses the balanced quality-filtered corpus. The same number of clean
            articles is sampled from each decade to reduce article-count imbalance.
          </p>
        </Card>
        <Card title="Dictionary Scores">
          <p>
            Framing scores count transparent word dictionaries. They indicate textual
            framing patterns and should be paired with close reading.
          </p>
        </Card>
        <Card title="Embeddings">
          <p>
            Word vectors are built separately by decade from co-occurrence patterns.
            Similarity scores compare terms within each decade's vocabulary.
          </p>
        </Card>
        <Card title="Limits">
          <p>
            The project reflects NYT coverage, API retrieval limits, metadata availability,
            search bias, and a headline/abstract-centered corpus rather than full article text.
          </p>
        </Card>
      </div>
    </section>
  );
}

function ReferencesPage({ references }) {
  return (
    <main className="page">
      <ReferencesSection references={references} />
    </main>
  );
}

function ReferencesSection({ references }) {
  return (
    <section className="section" id="reference-documentation">
      <div className="section-heading">
        <p className="eyebrow">Reference / Documentation</p>
        <h2>Sources And Documentation</h2>
      </div>
      <div className="reference-list">
        {references.map((ref) => (
          <article className="article-card" key={ref.url}>
            <h3>{ref.title}</h3>
            <p>{ref.note}</p>
            <a href={ref.url} target="_blank" rel="noreferrer">{ref.url}</a>
          </article>
        ))}
      </div>
    </section>
  );
}

function TermsPage() {
  const termKeys = Object.keys(DICTIONARY_TERMS);
  const [active, setActive] = useState(termKeys[0]);
  const activeDictionary = DICTIONARY_TERMS[active];

  return (
    <main className="page">
      <section className="section">
        <div className="section-heading">
          <p className="eyebrow">Project Terms</p>
          <h2>Definitions And Dictionary Words</h2>
          <p className="section-note">
            This page explains the key terms used in this project and lists the words counted in each framing dictionary.
            These dictionaries are interpretive tools: they make the method transparent, but they are not objective proof of sentiment.
          </p>
        </div>

        <div className="terms-layout">
          <aside className="terms-menu" aria-label="Dictionary categories">
            {termKeys.map((key) => (
              <button
                key={key}
                className={active === key ? "active" : ""}
                onClick={() => setActive(key)}
              >
                {DICTIONARY_TERMS[key].title}
              </button>
            ))}
          </aside>

          <article className="terms-panel">
            <p className="eyebrow">Selected Dictionary</p>
            <h3>{activeDictionary.title}</h3>
            <p>{activeDictionary.definition}</p>
            <div className="term-chip-grid">
              {activeDictionary.terms.map((term) => (
                <span className="term-chip" key={term}>{term}</span>
              ))}
            </div>
          </article>
        </div>

        <div className="method-grid term-definitions">
          <Card title="Engagement">
            <p>
              In this project, engagement means language that treats China as a country to be integrated through trade,
              diplomacy, talks, institutions, reform, or partnership.
            </p>
          </Card>
          <Card title="Security">
            <p>
              Security means language that presents China-related issues as risks involving national security, military power,
              surveillance, cyber conflict, sanctions, repression, or strategic vulnerability.
            </p>
          </Card>
          <Card title="Securitization">
            <p>
              Securitization means economic topics appearing together with security language. The project measures this by
              tracking overlap between economic words and security words in the same article record.
            </p>
          </Card>
          <Card title="Competition">
            <p>
              Competition means language of rivalry, trade conflict, tariffs, decoupling, strategic pressure, or dominance.
              It is counted separately from threat because competition can be economic, political, or technological.
            </p>
          </Card>
        </div>
      </section>
    </main>
  );
}

function ExhibitPage({ data, setPage }) {
  return (
    <main>
      <Landing balanceSummary={data.balanceSummary} onResult={() => setPage("result")} onAbout={() => setPage("about")} />
      <ReferencesSection references={data.references} />
    </main>
  );
}

function ResultPage({ data }) {
  return (
    <main className="page">
      <HistoricalNarrative framingDecade={data.framingDecade} cooccurrenceDecade={data.cooccurrenceDecade} />
      <TimelineDashboard framingYear={data.framingYear} events={data.events} />
      <DecadeSummaries
        framingDecade={data.framingDecade}
        cooccurrenceDecade={data.cooccurrenceDecade}
        technologyBridge={data.technologyBridge}
        engagementShiftDecade={data.engagementShiftDecade}
      />
      <DecadeFramingComparison framingDecade={data.framingDecade} />
      <Securitization cooccurrenceYear={data.cooccurrenceYear} cooccurrenceDecade={data.cooccurrenceDecade} />
      <TurningPointExplanations events={data.events} framingYear={data.framingYear} cooccurrenceYear={data.cooccurrenceYear} />
      <SemanticShift semanticPairs={data.semanticPairs} />
      <FrequencyAndThemes topWords={data.topWords} themeFraming={data.themeFraming} />
      <FramingByMetadata sectionFraming={data.sectionFraming} newsDeskFraming={data.newsDeskFraming} />
      <MethodLimitations />
    </main>
  );
}

function CloseReadingPage({ data }) {
  return (
    <main className="page">
      <CloseReadingGuide articles={data.articles} />
      <KeywordsAndCloseReading keywords={data.keywords} articles={data.articles} />
    </main>
  );
}

export default function App() {
  const { data, error } = useProjectData();
  const [page, setPage] = useState("exhibit");

  function goToReferences() {
    setPage("exhibit");
    window.setTimeout(() => {
      document.getElementById("reference-documentation")?.scrollIntoView({ behavior: "smooth" });
    }, 0);
  }

  if (error) {
    return <main className="status">Data loading error: {error}</main>;
  }

  if (!data) {
    return <main className="status">Loading project data...</main>;
  }

  return (
    <>
      <nav className="top-nav">
        <button className={page === "exhibit" ? "active" : ""} onClick={() => setPage("exhibit")}>Home</button>
        <button className={page === "about" ? "active" : ""} onClick={() => setPage("about")}>About</button>
        <button className={page === "terminology" ? "active" : ""} onClick={() => setPage("terminology")}>Terminology</button>
        <button className={page === "result" ? "active" : ""} onClick={() => setPage("result")}>Result</button>
        <button className={page === "close-reading" ? "active" : ""} onClick={() => setPage("close-reading")}>Close Reading</button>
      </nav>
      {page === "exhibit" && <ExhibitPage data={data} setPage={setPage} />}
      {page === "about" && (
        <AboutPage
          framingDecade={data.framingDecade}
          cooccurrenceDecade={data.cooccurrenceDecade}
          themeFraming={data.themeFraming}
          balanceSummary={data.balanceSummary}
        />
      )}
      {page === "terminology" && <TermsPage />}
      {page === "result" && <ResultPage data={data} />}
      {page === "close-reading" && <CloseReadingPage data={data} />}
      {page === "references" && <ReferencesPage references={data.references} />}
      <footer>
        <strong>NYT China Discourse Project</strong>
        <span>Digital history exhibit using local JSON data from a balanced, quality-filtered New York Times corpus.</span>
        <button className="footer-link" onClick={goToReferences}>Reference / documentation</button>
      </footer>
    </>
  );
}
