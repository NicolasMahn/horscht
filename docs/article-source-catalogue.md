# Article and document source catalogue

Updated: 2026-10-02

Status: pilot source access review completed for the sources below on 2026-10-02. Bounded feed/API and download-header checks only; no corpus import or production questions. Full harvesting connectors remain unimplemented. Game-generator infrastructure is separately authorized. Existing training remains frozen and bulk generation paused.

Access route, permission to fetch automatically, and permission to reuse full text are separate checks. Public availability, an API response or permissive robots.txt alone does not establish all three. Prefer published APIs, versioned downloads and bulk services over website crawling. Check item-level terms before admission; no blanket legal clearance is claimed here.

## Multilingual acquisition rule

User requirement, 2026-10-02: for every source offering multilingual material, use its available language editions or language-specific collections. This explicitly includes Wikipedia, Wikinews, Global Voices, MOT, government sources and research repositories. Do not silently default to English or only German and English. Inventory all available languages, plan sampling across eligible editions, and record unavailable, unlicensed or technically blocked editions with reasons. Check access and rights per host/edition; a cleared English host does not clear every other host. Actual release metadata and language identification take precedence over advertised language counts.

Use source-native text and human translations where available; machine-translating English is not a substitute for collecting multilingual sources. Track language, script, region, topic and original/translation status. Balance sampling explicitly rather than letting the largest editions dominate; do not pad small editions with repeats. Related translations remain in one split. Language coverage and editorial/source diversity are separate measures.

## Pilot access decisions — 2026-10-02

These are operational admission decisions based on the cited published policies, not individual publisher agreements. Eligible means a bounded text pilot may proceed with the stated item filters and attribution. Evidence and live probe results: [audit summary](../runs/source-access-audit-2026-10-02/summary.json). No bulk downloads were performed; RSS responses saved for access verification remain outside training discovery.

| Source | Decision and verified route | Conditions |
|---|---|---|
| MOT | Eligible for a version-pinned pilot via [official release v1.11](https://github.com/bltlab/mot/releases/tag/v1.11); release API and one asset HEAD returned 200 | Collection CC BY 4.0 per authors. Preserve corpus citation and source attribution; quarantine identifiable third-party/unclear-rights items. Article text only; separate editorial and language-learning material. No assumption that all current VOA content is reusable. |
| Global Voices | Eligible for original text in checked English, Spanish, Arabic and French hosts; RSS endpoints and robots returned 200 | Attribution license, original URL/author, license and changes retained; exceptions/third-party media excluded. Robots specifies 10-second crawl delay and excluded administrative paths. One request at a time with at least 10 seconds between requests per host; cache feeds. Other editions require the same checks before fetching. |
| Wikipedia | Eligible via official API or dumps; English rightsinfo API returned 200 and CC BY-SA 4.0 | Collect eligible language editions under the multilingual rule; preserve revision, attribution and share-alike obligations for reused/adapted text. English API success is not verification of every edition. Exclude separately restricted material. |
| Wikinews | Eligible for checked English text via Wikimedia API/dumps; rightsinfo returned CC BY 4.0 | Per-edition/revision license records required. Spanish policy text and footer reference different CC BY versions; resolve exact version before admitting Spanish records. Other editions remain pending their own checks. |
| PLOS | Eligible under explicit mining/reuse policy; official allofplos.zip HEAD returned 200 | Archive is currently 9,723,288,318 bytes, not the old page's approximate 5 GB. Prefer bounded eligible article retrieval through PMC for the pilot; no full archive download now. Keep article attribution/rights. The bare allof.plos.org root returned 404; the linked ZIP works. |
| PMC OA subset | Eligible only after item-level license filtering through approved services; current OAI Identify endpoint returned 200 | Pilot allowlist CC0/public-domain or CC BY text; other licenses need explicit handling. OAI: serial requests, at most 3/second; more than 100 requests outside documented peak hours. Legacy FTP recipes are obsolete. |
| GOV.UK | Eligible OGL-covered text via official Content API; two bounded API probes returned 200 | Preserve attribution and OGL reference, exclude third-party/other exceptions. Include available non-English versions; do not represent UK material as global coverage. |

MOT v1.11 manifest contains 54 archives, 43 distinct filename language prefixes and 8,342,562,863 compressed bytes. Release notes describe collection through 2025-03-16. The original paper's 44-language count is not the current manifest count. The Cantonese-named archive has a `cmn` prefix, so filenames alone are not sufficient language validation. Examples beyond English include Amharic, Bambara, Bangla, Burmese, Hausa, Khmer, Lao, Oromo, Somali, Swahili, Tibetan, Urdu and Vietnamese. Use the full manifest inventory, not just these examples.

Global Voices' [AI policy](https://globalvoices.org/about/global-voices-policy-on-ai/) addresses contributor writing and translation; it is not a separate external training license. Reuse relies on its [attribution policy](https://globalvoices.org/about/global-voices-attribution-policy/). Official [feeds](https://globalvoices.org/feeds/) and per-host robots were checked. Wikimedia [API rules](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines) require truthful identification and compliance with throttling; use a real contact before scheduled harvesting. Start API pilots serially at one request/second or slower, honoring any stricter host rule and Retry-After. Set bounded document/byte limits before a pilot.

Other publishers and the broader catalogue below retain their existing conditional or unreviewed status. Guardian/Reuters licensing has not been obtained; no outreach or paid access was initiated. PLOS [mining policy](https://plos.org/text-and-data-mining/), PMC [OAI documentation](https://pmc.ncbi.nlm.nih.gov/tools/oai/), GOV.UK [API documentation](https://content-api.publishing.service.gov.uk/) and [reuse policy](https://www.gov.uk/help/reuse-govuk-content) support the routes above.

## Sources with official documentation reviewed

| Source | Intended contribution | Preferred access | Admission decision |
|---|---|---|---|
| Wikipedia, many language editions | Broad factual topics, mixed-language comparisons, cross-section reasoning | Wikimedia API or official dumps; pin page/revision | Strong initial candidate; preserve attribution and applicable share-alike terms; check content exceptions and current bot policy |
| Wikibooks | Explanatory chapters and procedures | Wikimedia API/dumps | Candidate using same infrastructure; inspect project/item license before enabling |
| Wikisource | Historical essays, speeches and primary documents | Wikimedia API/dumps | Candidate; work, translation and edition rights need item-level checks |
| PLOS journals | Research methods, evidence, limitations, scientific argument | Published JATS XML bulk download, API or permitted PMC route | Strong initial candidate: explicit mining/reuse policy; retain article attribution and rights metadata |
| PMC Open Access Subset | Structured life-science research papers | Current PMC-designated cloud/API services | Strong initial candidate with item-license filtering; not all PMC content is reusable |
| Europe PMC OA content | Research discovery and accessible full text | REST API; OA full-text XML where available | Candidate; deduplicate against PMC/PLOS by identifiers; landing-page search verified, direct documentation open errored in this session |
| arXiv | Mathematics, physics, computing and other research | OAI-PMH/API metadata; official bulk full text or designated export host | Conditional: restrict to suitable per-paper licenses; default arXiv distribution license is not blanket redistribution permission; bulk transfer costs need checking |
| GOV.UK | Procedures, eligibility rules, reports and public-service guidance | Official Content API | Strong initial candidate under applicable Open Government Licence and exceptions |
| Our World in Data | Quantitative explanations, evidence and comparisons | Article acquisition subject to current access policy; chart API for data/metadata | Conditional: inspect writing/republication terms and third-party rights separately; chart API is not a full article API |
| OpenStax | Long explanatory textbooks across subjects | Publisher-provided downloads | Conditional: current help pages describe CC BY-NC-SA; inspect exact edition license and downstream compatibility rather than assuming older CC BY terms |
| Project Gutenberg | Long-form literature, essays and historical nonfiction | Designated harvest/mirror/catalog routes | Conditional: ordinary website crawling is restricted; check work rights in relevant jurisdictions and edition terms |
| NASA science material | Science explanations, reports and later media | Official downloadable resources or approved source-specific route | Candidate pending text-route verification; attribution/third-party exceptions; media guidelines do not blanket-clear all text |

Official evidence (checked 2026-10-01):

- [Wikimedia API policies](https://www.mediawiki.org/wiki/API:REST_API/Policies), [API usage guidelines](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines), [User-Agent policy](https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy/en), [content reuse overview](https://developer.wikimedia.org/use-content/content/).
- [PLOS text and data mining](https://plos.org/text-and-data-mining/).
- [PMC dataset access and licensing](https://pmc.ncbi.nlm.nih.gov/tools/textmining/). This page reports the 2026-08 transition away from legacy dataset distribution; do not implement from old FTP tutorials.
- [Europe PMC REST API](https://europepmc.org/RestfulWebService), [help](https://europepmc.org/Help).
- [arXiv bulk access](https://info.arxiv.org/help/bulk_data.html).
- [GOV.UK reuse](https://www.gov.uk/help/reuse-govuk-content), [Content API](https://content-api.publishing.service.gov.uk/).
- [Our World in Data terms and FAQ](https://ourworldindata.org/faqs).
- [OpenStax license help](https://help.openstax.org/s/article/What-is-Creative-Commons), [edition-specific example](https://openstax.org/books/introduction-computer-science/pages/preface). Help page body did not render in the browser fetch; search evidence and edition page must be reconfirmed during connector preflight.
- [Project Gutenberg robot access](https://www.gutenberg.org/policy/robot_access.html).
- [NASA science reuse FAQ](https://science.nasa.gov/earth/faq/), [media guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/).

## Broader backlog, not yet access-cleared

Keep these for geographic, topical and stylistic breadth. None is approved for harvesting on the basis of this list.

| Candidate family | What to investigate |
|---|---|
| SciELO | Multilingual research; item licenses and documented harvesting |
| DOAJ | Discovery and license metadata; not a blanket full-text corpus |
| OpenAlex and Crossref | Discovery/identifiers; metadata rights do not imply paper rights |
| HAL and institutional repositories | Humanities and regional research; repository route and item licenses |
| Zenodo | Reports/manuals; records may have different licenses and attachments |
| OpenEdition | Humanities/social sciences; access and item reuse terms |
| Global Voices | Regional journalism and translations; automated access and republication terms |
| The Conversation | Research communication/opinion; explicit AI/automation and republication review required |
| National statistical offices | Statistical bulletins, methodologies and explanations; country-specific terms |
| EU institutions and parliamentary records | Policy, debate and multilingual aligned documents; exact source terms |
| Municipal/regional government portals | Local procedures and public notices; varied jurisdictions/licenses |
| Museums, libraries and archives | Curatorial essays and historic documents; distinguish metadata from object rights |
| Open-source project manuals | Technical instructions and troubleshooting; documentation license distinct from code license |
| Standards/RFC-style documents | Definitions, constraints and interoperability; exact document rights |
| University open courseware | Educational explanation; download policies, NC/SA restrictions and third-party material |
| Consumer/how-to sources | Everyday procedures; explicit reusable licenses preferred |
| News publishers, blogs, forums | Valuable style diversity but no default scraping/reuse permission; source-specific review before inclusion |

## News organizations and archives

Approved for the planned article dataset on 2026-10-02. Reporting is a first-class source family, including the first acquisition pilot once source checks pass. No news harvest has started; bulk generation remains paused. Prefer multilingual, regional and local reporting as well as international outlets; include science, culture, sport, business and everyday civic reporting rather than predominantly politics.

| Source | Proposed role | Current decision |
|---|---|---|
| Global Voices | Regional perspectives and multilingual reporting/translations | Strong candidate: original content generally CC BY, exceptions and third-party media separate; automated acquisition route still to verify |
| Wikinews | Multilingual event reporting | Strong candidate using Wikimedia infrastructure; verify language-edition and revision-specific license |
| Multilingual Open Text (MOT), historical Voice of America corpus | Ready-made multilingual news archive | Authors report 44 languages and a CC BY 4.0 collection of public-domain source material; verify release manifest, provenance and exclusions before use; no blanket claim about present VOA pages or third-party wire stories |
| The Guardian | Reporting and opinion | Standard terms restrict AI training; separate licensing needed |
| Reuters | Wire reporting across events and regions | Offers AI-related content access/licensing; obtain applicable terms and price rather than scrape by default |
| ProPublica | Investigative reporting | Republication terms require review for this use; not cleared |
| AP, AFP, BBC, Deutsche Welle, France 24, Al Jazeera, regional/local publishers | Broader geographic/editorial coverage | Named backlog only: individual access, AI-use and content rights review outstanding |

Evidence: [Global Voices license](https://globalvoices.org/about/global-voices-attribution-policy/), [Global Voices AI policy](https://globalvoices.org/about/global-voices-policy-on-ai/), [Wikinews copyright](https://en.wikinews.org/wiki/Wikinews:Copyright), [MOT release paper](https://arxiv.org/abs/2201.05609), [Guardian terms](https://www.theguardian.com/open-platform/terms-and-conditions), [Guardian access packages](https://open-platform.theguardian.com/access/), [Reuters AI offerings](https://reutersagency.com/technology-ai/), [ProPublica republication terms](https://www.propublica.org/steal-our-stories). Source policies are not live connector verification. RSS and aggregator access do not themselves grant full-text training rights.

Suggested tasks: distinguish an allegation from an established finding; identify who made a claim; compare reports without assuming either is correct; determine what changed after a correction; assess whether a headline is supported; distinguish reporting, analysis, opinion and sponsored material; identify uncertainty and insufficient evidence. Targets should describe the provided evidence, not publisher reputation or assumed author affiliation.

Include ordinary comprehension and classification alongside reasoning across distant paragraphs and multiple reports. State whether a question asks what the article reports or what the supplied evidence establishes; attribution is not verification. Include an explicit insufficient-evidence option in both answerable and unanswerable cases, with its target determined by the evidence. Distinguish missing evidence from an inherently stochastic outcome.

Probability targets must record their meaning and derivation: a distribution derived from supplied counts or a stated mechanism, versus an explicitly identified teacher estimate. Do not invent numerical confidence for disputed causal claims or treat teacher estimates as calibrated truth. Apply the shared 265-output choice contract and permute choices together with their complete target distribution; reordered questions are augmentations, not new originals.

Preserve publication and update times plus the exact article revision. Group syndicated copies, translations and related event coverage for splitting; use temporal and event-held-out development cohorts. Do not let later reports leak future facts into questions framed at an earlier date. A quoted source's view is not automatically the journalist's view. Editorial/source diversity does not mean treating unsupported and well-supported factual claims as equally credible.

## Proposed first acquisition pilot

Begin with Wikipedia across multiple languages, PLOS or a filtered PMC subset, GOV.UK, and reporting. For reporting, investigate Global Voices, Wikinews and MOT first; select sources only after their acquisition and reuse checks pass. Use a small bounded sample first. This is a proposal, not a harvesting job. Add other sources as access/license checks pass rather than filling quotas with English biomedical papers. Source volume must not determine topic or language weighting. Track reporting explicitly in the source mix so it is not lost inside a generic article category; no numerical quota is fixed yet.

Before activating each connector, record the exact endpoint, robots/automation terms and check timestamp, license filter, per-host request budget, descriptive User-Agent/contact, retry/backoff rules, maximum bytes/documents, expected transfer cost and extraction format. Test a handful of permitted documents. Stop on access denial; do not bypass login, paywall, anti-bot controls or rate limits. No paid downloads without approval.

Preserve source URL, work ID (DOI/PMCID/arXiv ID/page ID), revision, language(s), authors/attribution, license text/URL, retrieval time and content hash. Keep raw documents separate from generated questions. Prefer structured XML/HTML to PDF when both exist; preserve section boundaries/tables and flag extraction/OCR defects. Article titles, figures and supplements may have different rights or supply missing evidence.

Group related revisions, translations, duplicate publications and all questions derived from a document before dataset splitting. A PLOS paper appearing in PMC and Europe PMC is one underlying work, not three independent sources. Never source questions from external benchmark collections or inspect their items for authoring.

Generate a limited variety per document: topic/language or mixed-language classification, supported claims, section integration, methodological limitations, contradictions, rule application and genuinely unanswerable questions. Evidence locations and author rationales remain outside model inputs unless explicitly part of the task. Known distributions require supported numerical mechanisms; ambiguous judgments must be labeled as estimates, not calibrated truth. Avoid inferring an author's private affiliation from the text.

Measure accepted original questions per document, extraction failure, source/language/topic balance, annotation cost and dependency across context. Long inputs alone do not establish long-context reasoning. Do not truncate decisive context to satisfy a token budget.

Related: [game and document roadmap](game-and-document-data-roadmap.md).
