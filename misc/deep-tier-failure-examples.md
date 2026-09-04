# DeepSearch QA — Concrete Failure Examples

Stakeholder-facing companion to
[`deep-tier-inaccessible-source-loop-analysis.md`](./deep-tier-inaccessible-source-loop-analysis.md).
Every example below is verbatim from the eval run — questions, reference answers, the agent's
output, and the exact URLs it attempted to open.

- **Eval run:** `results/aiq/deepsearchqa-adaptive-all-post-token-optim-full`
- **Config:** `configs/config_adaptive_frag.yml` (`workflow_timeout_seconds: 1200`)
- **Scale:** 198 questions — 93 `single_shot`, 60 `standard`, 45 `deep`

---

## The one-sentence version

> In all three categories the agent **finds the exact document holding the answer** and then
> cannot open it, because the deployment has **no tool that can fetch a URL** — only a web
> *search* tool. It searches for the file, gets results *about* the file, rewords the query, and
> repeats until the clock runs out.

---

## Category A — `standard` tier: the answer is inside a PDF

These runs **did not time out**. They finished quickly (280–800 s) and returned a confident,
well-formatted, wrong answer. The agent located the correct source document in every case and
then guessed at its contents.

### A1 — `dsqa_id_102` · accuracy **0.0** · 626 s · 69 searches

> **Question:** According to Orange County Transportation Authority's (OCTA) February 2024 Bus
> Book, identify the city that is the starting point for the most local fixed routes. Of those
> local fixed routes with the identified city as the starting point, look at the route with the
> largest number. How many stops have scheduled departures for that route?

| | |
| :-- | :-- |
| **Correct answer** | `5` |
| **Agent answered** | `28` stops on Route 99 |

**PDFs the agent tried to open (7 attempts):**
```
https://www.octa.net/ebusbook/Route_99_Feb2024.pdf
https://www.octa.net/ebusbook/CompleteBusBook_February2024.pdf
https://www.octa.net/ebusbook/RoutePDF/route099.pdf?n=202402
site:octa.net ebusbook CompleteBusBook.pdf 202402 February 2024
site:octa.net ebusbook Route_99_Feb2024.pdf
```

**What went wrong:** The agent identified the exact PDF and the exact route. The timetable inside
that PDF has 5 stops with scheduled departures. Unable to open it, the agent invented "28
individual stops." The answer is off by 5.6×, presented with full confidence and a citation.

---

### A2 — `dsqa_id_81` · accuracy **0.0** · 280 s · 49 searches

> **Question:** I have recently graduated high school and am searching for a college or university
> to attend. Of the colleges and universities in Florida, which is closest to Tallahassee that
> satisfies the following criteria: was a National Merit college sponsor in the 2021–2022 academic
> year, and offers an undergraduate degree in aerospace engineering.

| | |
| :-- | :-- |
| **Correct answer** | University of Florida |
| **Agent answered** | FAMU-FSU College of Engineering |

**PDF the agent tried to open:**
```
https://www.nationalmerit.org/s/1758/images/gid2/editor_documents/merit_sponsor_leaflet_2022.pdf
```

**What went wrong:** The sponsor list is a PDF. The agent could not read it, assumed FSU was on it
because FSU is a large university in Tallahassee, and answered with the geographically closest
option. The actual sponsor list does not include FSU — it includes University of Florida.
The failure is a fabricated membership claim, not a distance-calculation error.

---

### A3 — `dsqa_id_96` · accuracy **0.25** · 378 s · 41 searches

> **Question:** Which states had alcohol-impaired-driving fatality rates per 100 million vehicle
> miles traveled that exceeded 0.43 in 2022, according to NHTSA data, and also had an election for
> governor during the 2022 midterm elections that resulted in a Republican winner?

| | |
| :-- | :-- |
| **Correct answer** | Nevada, Texas, Tennessee, South Carolina |
| **Agent answered** | Alaska, Florida, Georgia, Ohio, South Carolina, Texas |

**PDF the agent tried to open:**
```
https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/813579
```

**What went wrong:** Only 2 of 6 states overlap with the correct set; Nevada and Tennessee were
missed entirely. The per-state rate table lives in the NHTSA publication PDF. The agent
substituted plausible-looking rates (0.44–0.47) that do not match the source, then filtered on
them. **Note the second-order risk:** the election half of the question was answered correctly, so
the output looks internally consistent and is hard to spot as wrong without the source.

---

### A4 — `dsqa_id_122` · accuracy **0.25** · 595 s · 191 searches

> **Question:** Using information from the Council on Tall Buildings and Urban Habitat rankings
> from 49 to 53 on the list of tallest buildings in the world in 1970, give me the buildings in
> order of the date of incorporation of their respective cities, from earliest to latest.

| | |
| :-- | :-- |
| **Correct answer** | BNY Mellon Center at One Boston Place, La Tour CIBC, Chicago Board of Trade, 800 Bell Street, Gables Republic Tower |
| **Agent answered** | Partial list — ranks 50, 51, 53 returned as *"Conflicting/unreliable data"* |

**PDFs the agent tried to open:**
```
https://www.ctbuh.org/knowledge/ctbuh-1970-100-tallest-buildings.pdf
https://cloud.ctbuh.org/CTBUH-100-Tallest-Buildings-1970.pdf
```

**What went wrong:** 191 searches — the most of any `standard` run — and the two buildings it did
name are both wrong. To its credit the agent **correctly flagged its own uncertainty** rather than
fabricating. This is the honest-failure version of the same root cause, and it shows the wasted
cost plainly: 191 searches to produce an explicit "I don't know."

---

### A5 — `dsqa_id_93` · accuracy **0.0** · 799 s · 97 searches

> **Question:** Of the top 5 species groups in global aquaculture from the years 2020, 2021, and
> 2022, according to the Food and Agriculture Organization of the United Nations' "Top 10 species
> groups in global aquaculture" reports from those years, which 2 species groups did not
> consistently remain at the exact same ranking over all 3 years?

| | |
| :-- | :-- |
| **Correct answer** | Brown seaweeds, Red seaweeds |
| **Agent answered** | Built a ranking table from carps/molluscs; never surfaced either seaweed group |

**FAO documents the agent tried to open:**
```
https://openknowledge.fao.org/handle/20.500.14283/cd1194en
https://openknowledge.fao.org/bitstreams/6c485171-8a5a-4379-92f6-b0e9fbc7e95d/download
```

**What went wrong:** The answer requires comparing three FAO factsheets year over year. The agent
reconstructed the rankings from search snippets and produced a table dominated by the largest
categories. The two groups that actually moved — brown and red seaweeds — never appeared.

---

### Category A summary

| ID | Accuracy | Latency | Searches | Document type | Failure |
| :-- | --: | --: | --: | :-- | :-- |
| `dsqa_id_102` | 0.0 | 626 s | 69 | Transit timetable PDF | Answered 28, correct is 5 |
| `dsqa_id_81` | 0.0 | 280 s | 49 | Sponsor-list PDF | Fabricated list membership |
| `dsqa_id_96` | 0.25 | 378 s | 41 | NHTSA statistics PDF | 2 of 6 states correct |
| `dsqa_id_122` | 0.25 | 595 s | 191 | CTBUH ranking PDF | Self-declared "unreliable" |
| `dsqa_id_93` | 0.0 | 799 s | 97 | FAO report PDFs | Missed both answer items |

**Pattern:** Fast, cheap, confident, and wrong. Because these runs never hit a timeout there is no
error and no warning — the failure is silent. From a stakeholder standpoint this is the most
dangerous category in the eval: the output is fluent, cited, and undetectably incorrect without
checking the source by hand.

---

## Category B — `deep` tier: timed out at 1200 s

23 of 45 `deep` runs (51%) ended this way, returning the boilerplate
*"Research stopped before completion because the 1200s workflow time limit was reached."*
Mean accuracy for this group: **0.09**. Mean accuracy for `deep` runs that finished: **0.68**.

### B1 — `dsqa_id_82` · accuracy **0.0** · 1200 s · **281 searches** (most in the eval)

> **Question:** Using data from the IEA, determine which countries had government policies enacted
> in 2021 or prior that are currently in effect for efficient or cleaner technologies. Then,
> identify the number of EV car sales for battery electric vehicles in 2021 for each country
> (IEA). Discard any countries that had more than 100,000 sales or fewer than 2,750 sales.
> Finally, using the IEA's Renewable Energy Progress Tracker, identify the cumulative capacity
> total in gigawatts for the amount of bioenergy for each country. Determine the country that had
> the lowest amount of BEV sales in 2021 while having the highest amount of gigawatts used for
> bioenergy, and provide the country's population count for 2021 according to database.earth.

| | |
| :-- | :-- |
| **Correct answer** | `209,550,294` |
| **Agent answered** | Timed out — 23 sources consulted, no synthesis |

**What went wrong:** Four chained filters across three separate IEA datasets plus a fourth site
for the final population figure. Each stage requires a full dataset, not a snippet. 281 searches
in, the agent had still not cleared stage one.

---

### B2 — `dsqa_id_0` · accuracy **0.25** · 1207 s · 270 searches

> **Question:** Consider the OECD countries whose total population was composed of at least 20% of
> foreign-born populations as of 2023 (according to the Observatory of Migration at the university
> of Oxford). Amongst them, which country saw their overall criminality score increase by at least
> +0.2 point between 2021 and 2023 and their resilience score decrease by more than 0.3 between
> these same dates (according to the Organised Crime Index)?

| | |
| :-- | :-- |
| **Correct answer** | New Zealand |
| **Agent answered** | Timed out — 40 sources consulted, no synthesis |

**PDFs chased — one per country, per year:**
```
ocindex.net/assets/downloads/2023/english/ocindex_profile_new_zealand_2023.pdf
ocindex.net/assets/downloads/2021/english/ocindex_profile_austria_2021.pdf
ocindex.net/assets/downloads/2023/english/ocindex_profile_australia_2023.pdf
ocindex.net/assets/downloads/2021/english/ocindex_profile_switzerland_2021.pdf
ocindex.net/assets/downloads/2025/english/ocindex_profile_new_zealand_2025.pdf
```

**What went wrong:** The agent's method was correct — it even reached the New Zealand 2023 profile,
the document containing the answer. But each country needs two PDFs (2021 and 2023) across ~10
candidate countries: 20 PDFs, none openable. It burned 270 searches re-requesting them.

---

### B3 — `dsqa_id_31` · accuracy **0.0** · 1200 s · 248 searches

> **Question:** According to the Our World in Data migration chart and the IMF's World Economic
> Outlook 2023 GDP per capita data, which countries had an absolute change in the total number of
> emigrants between 3 million and 4 million from 1990 to 2024, a relative change in emigrants
> above 100% from 1990 to 2024, and a GDP per capita above $2,000 in 2023?

| | |
| :-- | :-- |
| **Correct answer** | Romania, Egypt |
| **Agent answered** | Timed out — 11 sources consulted, no synthesis |

**Datasets and APIs chased:**
```
https://ourworldindata.org/grapher/migrant-stock-emigrants.csv
https://ourworldindata.org/grapher/migrant-stock-total.csv
https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/WEO/NGDPDPC.?startPeriod=2023&endPeriod=2023&format=CSV
https://api.imf.org/v2/data/NGDPDPC?countries=all&startPeriod=2023&endPeriod=2023&format=csv
```

**What went wrong:** This question is trivial *with the two CSVs* — a filter and a join. Without a
fetch tool the agent tried to reconstruct a global emigration dataset country-by-country via
search. It also fell back to searching per-country (`"migrant-stock-emigrants.csv India 1990 2024
emigrants"`), which is why the search count is so high.

---

### B4 — `dsqa_id_21` · accuracy **0.0** · 1200 s · 200 searches

> **Question:** There is an article that begins on page 433 of Volume 47 of Polar Biology (Springer
> Nature Link). In the first-listed source from 2017 (under "references") that was written by the
> first two listed authors of the aforementioned article, what are the last names (in alphabetical
> order) of the first ten authors/scholars cited in the introduction? Don't count any last names
> more than once. Separate all last names with a comma.

| | |
| :-- | :-- |
| **Correct answer** | Camhi, Cortes, Goldman, Gruber, Hoenig, Jensen, King, McFarlane, Rose, Winemiller |
| **Agent answered** | Timed out — 7 sources consulted, no synthesis |

**Publisher pages chased:**
```
https://www.researchgate.net/publication/319156390_Age_and_Growth_of_Elasmobranchs...
https://repository.library.noaa.gov/view/noaa/65736/noaa_65736_DS1.pdf
https://www.sciencedirect.com/science/chapter/bookseries/pii/S0065288117300020
https://sdgresources.relx.com/book-chapters/advances-marine-biology-chapter-6...?page=179
```

**What went wrong:** A three-hop citation chain (locate article → resolve its 2017 reference →
read that paper's introduction), each hop strictly dependent on the previous. The agent correctly
identified the target paper (Matta & Tribuzio 2017) but needed to read the body text of an
academic PDF. It reworded the same query **8 times** — the single worst reformulation loop in the
run.

---

### B5 — `dsqa_id_15` · accuracy **0.0** · 1200 s · 172 searches · **zero repeated queries**

> **Question:** List all of the countries that meet all of the following conditions: Is an EU
> member state as of 2024; Doesn't have a monarchy; Had over 100,000 people immigrate in 2022; As
> of 2024, the voting age is 18 […]

| | |
| :-- | :-- |
| **Correct answer** | France, Italy, Romania, Portugal |
| **Agent answered** | Timed out — no synthesis |

**What went wrong:** Included deliberately as a **contrast case**. There is no loop here — every
one of the 172 searches was distinct and productive. Four independent constraints across 27
member states is simply more retrieval than fits in 1200 s at ~33 s per search round. For
questions like this, more time genuinely is the fix; for B1–B4 it is not.

---

### Category B summary

| ID | Accuracy | Searches | Blocker | More time alone fixes it? |
| :-- | --: | --: | :-- | :-- |
| `dsqa_id_82` | 0.0 | 281 | 3 IEA datasets, 4 chained filters | Unlikely |
| `dsqa_id_0` | 0.25 | 270 | ~20 country-profile PDFs | No — needs fetch |
| `dsqa_id_31` | 0.0 | 248 | 2 CSVs + IMF API | No — needs fetch |
| `dsqa_id_21` | 0.0 | 200 | Academic PDF body text | No — needs fetch |
| `dsqa_id_15` | 0.0 | 172 | Breadth only, no loop | **Yes** |

---

## Category C — `deep` tier: blocked on a specific file format

Same root cause as Category B, isolated by artifact type to scope the fetch tool. Note that
**C2 did not time out** — it finished in 938 s and still failed, proving the gap is a missing
capability rather than a missing deadline.

### C1 — `dsqa_id_68` · **`.ods` spreadsheets** · accuracy 0.25 · 1200 s · 253 searches · 110 file reads

> **Question:** I am doing a study on the performance of rail operators in the UK. Identify the
> rail operators that had between 5 and 10 million passenger journeys in the UK between October
> and December 2021 (according to the ORR Passenger rail usage report), and also had a punctuality
> score of less than 70% of trains on time in the same period. Of these operators, did any have a
> delay compensation claim approval rate of less than 80% and a volume of delay compensation
> claims closed of less than 100,000, between October 2021 and January 2022? If so, please list them.

| | |
| :-- | :-- |
| **Correct answer** | East Midlands Railway, TransPennine Express |
| **Agent answered** | Timed out — 17 sources consulted |

**Spreadsheets chased:**
```
https://dataportal.orr.gov.uk/media/2050/table-3133-2021-22-q3.ods
https://dataportal.orr.gov.uk/media/1496/4410-delay-compensation-claims.ods
https://dataportal.orr.gov.uk/media/2048/table-1223.ods
```

**Why it matters:** All three UK regulator statistics tables are OpenDocument spreadsheets. The
agent named the correct file numbers (`table-3133`, `4410`) — it knew precisely what it needed.
**Requires: ODS/XLSX parsing.**

---

### C2 — `dsqa_id_56` · **`.xlsx` spreadsheets** · accuracy 0.0 · 938 s · 117 searches · **no timeout**

> **Question:** Of the school districts in Maricopa County that had a 4-day a week schedule for the
> 2023-2024 school year (according to the Maricopa County School Superintendent), which ones had a
> minimum of 1200 students for the same school year, an inexperienced core teachers, principals,
> and school leaders percentage for their Title I schools of 25% or less, and teachers with
> emergency credentials of 5% or under for their Title I schools, according to AZ School Report
> Cards for 2024? Include only the final answer.

| | |
| :-- | :-- |
| **Correct answer** | Wickenburg Unified District |
| **Agent answered** | *"No districts can be confirmed as meeting all criteria"* |

**Spreadsheets chased:**
```
https://www.azed.gov/data/public-data-sets
azed.gov Teacher_Qualification_2024.xlsx
azed.gov Teacher_Qualification_2024.csv
```

**Why it matters:** The single clearest example in the eval. The agent finished **262 s inside the
deadline** and still failed, explicitly reporting that it could not obtain the data. Raising the
timeout would change nothing here. **Requires: XLSX parsing.**

---

### C3 — `dsqa_id_100` · **`.csv` downloads** · accuracy 0.0 · 1200 s · 95 searches · **190 file reads**

> **Question:** Identify which lower tier/unitary authorities in England with a business birth rate
> (percentage of business newly registered for VAT and/or PAYE) in excess of over 16% also had an
> average gross disposable income of over the English average gross disposable income in 2021. Use
> the Office of National Statistics.

| | |
| :-- | :-- |
| **Correct answer** | Islington, Haringey, Hackney, Enfield, Wychavon |
| **Agent answered** | Timed out — 34 sources consulted |

**CSVs chased:**
```
https://www.ons.gov.uk/file?uri=/businessindustryandtrade/business/activitysizeandlocation/datasets/businessdemography2021localauthority.csv
https://www.ons.gov.uk/file?uri=/.../businessbirthsbylocalauthority2021.csv
https://www.ons.gov.uk/economy/regionalaccounts/.../RegionalGDHI_ITL3
```

**Why it matters:** **190 `read_file` calls** — the highest in the eval — suggests the agent
obtained *something* and re-read it repeatedly without extracting the answer. Worth investigating
whether a partial download path already exists and is misbehaving, rather than assuming a pure
capability gap. **Requires: CSV parsing with row/column slicing** (this file has ~300 local
authorities × many columns).

---

### C4 — `dsqa_id_8` · **JSON REST API** · accuracy 0.0 · 1200 s · 83 searches · 125 file reads

> **Question:** According to Data NYC's MTA Subway Trains Delayed: 2020-2024 and Subway and Bus
> ridership on the MTA website, which stations in the top 10 busiest stations of 2023 are serviced
> by the subway line with the most delays caused by Fire, Smoke, Debris in the same year?

| | |
| :-- | :-- |
| **Correct answer** | Times Sq-42 St/Port Authority Bus Terminal, 34 St-Herald Sq, 14 St-Union Sq |
| **Agent answered** | Timed out — 7 sources consulted |

**API endpoints chased — note the correctly-formed SoQL:**
```
https://data.ny.gov/resource/g937-7k7c.json?$select=line,delay_cause,sum(total_delays)
https://data.ny.gov/resource/g937-7k7c.json?$where=month between '2023-01-01' and '2023-12-31'
https://data.cityofnewyork.us/resource/kku6-nxdu.json?$select=line,count(*)&$where=delay_cause='Fire/Smoke/Debris'
https://data.ny.gov/api/views/g937-7k7c/rows.csv?accessType=DOWNLOAD
```

**Why it matters:** The strongest evidence that the model's *reasoning* is sound. It wrote valid
Socrata SoQL queries with the right dataset ID, the right aggregation, and the right filter —
queries that would have returned the answer directly. It had no way to execute an HTTP GET.
**Requires: JSON/REST fetch.**

---

### C5 — `dsqa_id_40` · **`.csv` from a report site** · accuracy 0.0 · 1200 s · 84 searches

> **Question:** Using the 2024 World Happiness Report and the US Census Bureau's International
> Database, which countries with a population of over 1,000,000 in 2023 were ranked within the top
> 20 by life evaluation in the 2021-2023 period, had a positive change in happiness from the
> 2006-2010 period to the 2021-2023 period and whose "happiest" age group was young people (under
> 30)?

| | |
| :-- | :-- |
| **Correct answer** | Israel, Lithuania, Czechia |
| **Agent answered** | Timed out — 7 sources consulted |

**CSVs chased:**
```
https://files.worldhappiness.report/WHR24/Figure-2-5.csv
https://worldhappiness.report/ed/2024/data/WHR24-Figure2-5-Changes.csv
```

**Why it matters:** The "happiest age group" data exists **only** as the Figure 2.5 data file — it
is not in the report's prose, so no amount of searching can surface it. **Requires: CSV fetch.**

---

### Category C summary — fetch-tool requirements, prioritized by observed need

| Format | Examples | Sites involved |
| :-- | :-- | :-- |
| **PDF** (text extraction) | A1, A2, A3, A4, A5, B2, B4 | OCTA, NationalMerit, NHTSA, CTBUH, FAO, ocindex, NOAA |
| **CSV** (parse + slice) | B3, C3, C5 | ONS, Our World in Data, World Happiness Report |
| **JSON / REST API** | B3, C4 | data.ny.gov, IMF, Socrata |
| **XLSX / ODS** | C1, C2 | ORR data portal, azed.gov |
| **HTML `<table>`** | B1 | IEA, NCES |

PDF is the clear priority — it blocks all five Category A failures plus two Category B runs.

---

## What stakeholders should take away

1. **This is a capability gap, not a model-quality problem.** When `deep` runs complete, they are
   the most accurate tier in the eval (**0.68**). The agent's plans, dataset identification, and
   even its hand-written API queries (C4) are correct.

2. **The gap is narrow and concrete.** There is no tool that can open a URL. Adding one addresses
   all 15 examples here and the majority of the 51 non-timeout failures across `standard` and
   `single_shot`.

3. **Raising the timeout is necessary but not sufficient.** It fixes breadth-bound questions like
   B5. It does nothing for C2, which finished 262 s early and still failed.

4. **The silent failures are the business risk.** Category A produced no error, no timeout, and no
   warning — just fluent, cited, wrong answers (A1 off by 5.6×). Category B at least announces
   that it failed.

5. **Estimated impact.** Eliminating deep-tier timeouts alone moves overall E2E accuracy from
   **0.33 → ~0.46**, before counting any newly-answerable questions from fetch support.

---

## Caveats on this analysis

- Tier labels come from the `declare_effort_tier` event in each record; failure categories were
  assigned by inspecting the generated answers and the attempted URLs, not by a labelled ground
  truth. The "what went wrong" narrative per example is an inference from that evidence.
- The 5 examples per category were selected as the clearest illustrations, not sampled randomly —
  they are representative of the patterns, but the per-category counts in the companion analysis
  doc are the quantitative claim.
- `dsqa_id_100`'s 190 `read_file` calls are unexplained and may indicate an existing partial
  download path; this should be confirmed before scoping fetch work.
