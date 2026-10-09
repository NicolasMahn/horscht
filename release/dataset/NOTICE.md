This is a collection with several source licenses. Preserve this notice, the
per-row provenance, `public-source-manifest.json`, and
`document-attribution.jsonl` when redistributing applicable portions.

Nicolas Mahn created the original synthetic reasoning and game questions with
AI-assisted authoring. Older generated reasoning data passed structure and
probability arithmetic checks; its manifest explicitly says its prose semantics
were not independently reviewed. New document questions were independently
reviewed; game supervision was computed by deterministic solvers/enumeration.

Typed Decisions is attributed to LocalLLaMA/typed-decisions, pinned at
`e039ebffcc280174dd354227424fb2b249f191de`, Apache-2.0. The exporter selects from
the source-native training partition, expands typed decisions into individual
questions and retains source probabilities. Source acquisition evidence is in
`public-source-manifest.json`. The original publisher's attribution is retained;
the selected transformed data is not a claim of publisher endorsement.

SciQ is attributed to allenai/sciq and Johannes Welbl, Nelson F. Liu and Matt
Gardner, “Crowdsourcing Multiple Choice Science Questions” (2017), pinned at
`2c94ad3e1aafab77146f384e23536f97a4849815`, CC-BY-NC-3.0. Source-native train
questions only were acquired; 33 invalid duplicate-option conversions were
rejected before mixture selection. Its noncommercial license remains in force.
The selected rows were converted to context/question/options with hard targets.

Banking77 is attributed to PolyAI and Iñigo Casanueva, Tadas Temčinas, Daniela
Gerz, Matthew Henderson and Ivan Vulić, “Efficient Intent Detection with Dual
Sentence Encoders” (2020), CC-BY-4.0. The source-native training CSV is pinned
at PolyAI-LDN/task-specific-datasets revision
`57ec275d8078af65b7731c2a98be812d844a6d6b`; license metadata is pinned at
PolyAI/banking77 revision `90d4e2ee5521c04fc1488f065b8b083658768c57`. Inputs
are classified among the 77 original intent categories.

Public-source option permutations were applied after group assignment with
choice identities and the entire target distribution remapped together.
Export preserves those frozen orders and raw probabilities; it does not apply
the additional deterministic training-time permutations.

Wikipedia text is attributed to Wikipedia contributors. Each document notice
includes the article, pinned revision and contributor-history link, together
with the recorded extraction changes. Use the recorded CC-BY-SA-4.0 option;
this release does not elect GFDL as a substitute. Extracted article text and
original document-question adaptations are CC-BY-SA-4.0. Attribution by links
to the original article/history follows Wikimedia's licensing terms:
https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content

Global Voices article text, author/translator credits and source URLs are
recorded per item under CC-BY-3.0. Text extraction omits media assets; it does
not license photos or externally credited material. The source policy is
https://globalvoices.org/about/global-voices-attribution-policy/ .

GOV.UK text is attributed to © Crown copyright; GOV.UK and any named department
in its document notice. Contains public sector information licensed under the
Open Government Licence v3.0:
https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ .
No Crown endorsement is implied. Images, logos and recorded copyright
exceptions were excluded during original intake.

Scientific article text from PMC and PLOS carries the author credits, exact
article identifiers, license and rights evidence recorded in each notice.
The applicable item license can be CC-BY-2.0, CC-BY-3.0, CC-BY-4.0 or CC0-1.0;
PMC/PLOS inclusion alone is not a rights grant. Acquisition checked governing
item rights and omitted figure assets. Two-document packets preserve every
component's attribution, pinned version and license.

The common export schema removes authoring assignments, runtime paths,
unrelated review logs and seed metadata. Full question/context/option text and
the frozen target values are unchanged. Source rows are identified by SHA-256;
the authoritative split sidecar overrides historical source `split` fields.

wordfreq 3.1.1 provided generator-only word seeds in earlier authoring. The
frequency lists, ranks and assignment metadata are excluded from this release.
For that historical provenance, credit Robyn Speer, Marc Brysbaert et al.
(SUBTLEX, freely available), and OpenSubtitles. See the recorded upstream terms:
https://github.com/rspeer/wordfreq/blob/v3.1.1/NOTICE.md .
An individual seed word's incidental appearance in an original question is not
publication of a frequency list. No blanket grant is made for the seed pool.
