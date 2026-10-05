import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "runs/document-breadth-2026-10-03/acquisition/acquire_science_public.py"
_SPEC = importlib.util.spec_from_file_location("acquire_science_public", _SCRIPT)
assert _SPEC and _SPEC.loader
acquisition = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = acquisition
_SPEC.loader.exec_module(acquisition)


def test_candidate_rejects_combined_noncommercial_noderivatives_dc_rights():
    record = ET.fromstring(
        """<record xmlns:oai="http://www.openarchives.org/OAI/2.0/"
                  xmlns:dc="http://purl.org/dc/elements/1.1/"
                  xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/">
          <header><identifier>oai:pubmedcentral.nih.gov:123456</identifier></header>
          <metadata><oai_dc:dc>
            <dc:title>Rainfall changes in agricultural regions</dc:title>
            <dc:language>en</dc:language>
            <dc:rights>Creative Commons Attribution-NonCommercial-NoDerivs License; https://creativecommons.org/licenses/by/4.0/</dc:rights>
            <dc:rights>https://creativecommons.org/licenses/by/4.0/</dc:rights>
          </oai_dc:dc></metadata>
        </record>"""
    )

    candidate, pmcid, reason = acquisition.candidate_from_oai_dc(record, "test", "page-sha")

    assert candidate is None
    assert pmcid == "PMC123456"
    assert reason == "missing_or_unapproved_explicit_item_license"


def test_jats_rejects_bare_ccby_href_when_governing_license_is_restricted():
    jats = b"""<article xmlns:xlink="http://www.w3.org/1999/xlink">
      <front><article-meta>
        <article-title>Rainfall changes in agricultural regions</article-title>
        <license>https://creativecommons.org/licenses/by/4.0/ This article is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs License.</license>
        <license xlink:href="https://creativecommons.org/licenses/by/4.0/">https://creativecommons.org/licenses/by/4.0/</license>
      </article-meta></front>
    </article>"""

    document, reason = acquisition.parse_science_document(jats, "PMC123456", "test")

    assert document is None
    assert reason == "missing_or_unapproved_explicit_item_license"


def test_combined_jats_rights_reject_bare_ccby_href_next_to_separate_restriction():
    terms = [
        ("https://creativecommons.org/licenses/by/4.0/", ""),
        ("", "This work is NonCommercial and may not be redistributed as derivatives."),
    ]

    assert acquisition.canonical_combined_cc_license(terms) is None


def test_jats_permissions_restrictions_govern_alongside_license_href():
    meta = ET.fromstring(
        """<article-meta xmlns:xlink="http://www.w3.org/1999/xlink">
          <permissions><copyright-statement>This work is licensed for non-commercial use only.</copyright-statement></permissions>
          <license xlink:href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International</license>
        </article-meta>"""
    )
    terms, evidence = acquisition.jats_governing_license_evidence(meta)

    assert acquisition.canonical_combined_cc_license(terms) is None
    assert {row["node"] for row in evidence} == {"permissions", "copyright-statement", "license"}


def test_combined_license_evidence_holds_conflicting_ccby_versions():
    terms = [
        ("https://creativecommons.org/licenses/by/4.0/", ""),
        ("", "Creative Commons Attribution 3.0 International"),
    ]

    assert acquisition.canonical_combined_cc_license(terms) is None


def test_science_topic_record_hash_is_stable_and_canonical():
    row = {"title": "Münster ☆", "id": "science-fixture", "reason_codes": []}
    expected = acquisition.sha(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )

    assert acquisition.canonical_science_review_row_sha256(row) == expected


def test_science_identity_scope_hash_ignores_audit_check_timestamps():
    row = {
        "decision": "verified",
        "canonical_pmcid": "PMC123456",
        "canonical_doi": "10.1371/journal.pone.1234567",
        "canonical_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/",
        "raw_source_json_sha256": "a" * 64,
        "fulltext_raw_file": "runs/raw/article.xml",
        "fulltext_raw_sha256": "B" * 64,
        "dc_getrecord_request": {
            "url": "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/?verb=GetRecord",
            "http_status": 200,
            "raw_file": "runs/raw/dc.xml",
            "sha256": "c" * 64,
            "retrieved_at": "2026-10-03T10:00:00+00:00",
            "verification_checked_at": "2026-10-03T11:00:00+00:00",
            "cached": True,
        },
    }
    checked_later = json.loads(json.dumps(row))
    checked_later["dc_getrecord_request"]["retrieved_at"] = "2026-10-03T12:00:00+00:00"
    checked_later["dc_getrecord_request"]["verification_checked_at"] = "2026-10-03T12:00:00+00:00"
    checked_later["dc_getrecord_request"]["cached"] = False

    assert acquisition.science_identity_scope_sha256(row) == acquisition.science_identity_scope_sha256(checked_later)
    assert len(acquisition.science_identity_scope_payload(row)) == 32


def test_oai_title_identity_normalizes_publisher_symbol_elision_only():
    assert acquisition.normalized_title_identity("On modeling context-aware social collaboration processes☆") == acquisition.normalized_title_identity(
        "On modeling context-aware social collaboration processes()"
    )


def test_basic_cell_biology_is_admitted_without_patient_care_framing():
    assert acquisition.classify_science_topic(
        "TNF-α promotes breast cancer cell migration", "", "", "en"
    ) == ("biological_science", None)


def test_patient_biomarker_papers_remain_outside_basic_biology_scope():
    assert acquisition.classify_science_topic(
        "MicroRNA as a biomarker for follow-up in cancer patients", "", "", "en"
    ) == (None, "narrow_clinical_or_mechanistic_biomedical_title")


def test_german_native_title_signals_are_classified_without_translation():
    assert acquisition.classify_science_topic(
        "Wie die Lichtwahrnehmung unsere innere Uhr beeinflusst", "", "", "de"
    ) == ("biological_science", None)


def test_energy_restriction_title_is_not_misclassified_as_physical_science():
    assert acquisition.classify_science_topic(
        "The impact of intermittent energy restriction on women’s health", "", "", "en"
    ) == (None, "narrow_clinical_or_mechanistic_biomedical_title")


def test_wind_energy_wildlife_study_is_classified_as_environmental_ecology():
    assert acquisition.classify_science_topic(
        "Ultrasonic deterrents and bat fatalities at a wind energy facility", "", "", "en"
    ) == ("environment_ecology", None)


def test_plos_article_routes_use_publisher_journal_paths():
    assert acquisition.plos_article_slug("PLOS Climate", "10.1371/journal.pclm.0000133") == "climate"
    assert acquisition.plos_article_slug("PLOS Global Public Health", "10.1371/journal.pgph.0006211") == "globalpublichealth"
    assert acquisition.plos_article_slug("PLOS Water", "10.1371/journal.pwat.0000549") == "water"
    assert acquisition.plos_article_slug("PLOS Digital Health", "10.1371/journal.pdig.0001325") == "digitalhealth"


def test_science_topic_cap_is_applied_before_selecting_another_batch():
    assert acquisition.science_topic_capacity_available("biological_science", {"biological_science": 19})
    assert not acquisition.science_topic_capacity_available("biological_science", {"biological_science": 20})
    assert not acquisition.science_topic_capacity_available(
        "biological_science", {"biological_science": 19}, {"biological_science": 1}
    )


def test_wetland_and_solar_climate_research_are_scoped_as_environment():
    assert acquisition.classify_science_topic(
        "Export of macroinvertebrate prey from tidal freshwater wetlands provides an energy subsidy", "", "", "en"
    ) == ("environment_ecology", None)
    assert acquisition.classify_science_topic(
        "Dust as a solar shield", "A possible climate-change mitigation measure", "", "en"
    ) == ("environment_ecology", None)


def test_water_wildlife_research_uses_the_topic_context():
    assert acquisition.classify_science_topic(
        "Wildfire mobilization of arsenic into water resources", "", "", "en"
    ) == ("environment_ecology", None)
    assert acquisition.classify_science_topic(
        "Behavioural analysis of satellite telemetry data provides insight into narwhal winter prey selection", "", "", "en"
    ) == ("environment_ecology", None)
    assert acquisition.classify_science_topic(
        "The impact of racially-targeted food marketing on consumption in Black adolescent females with obesity", "", "", "en"
    ) == ("public_health", None)
    assert acquisition.classify_science_topic(
        "Advancing the design of the kissing bug kill trap for surveillance of triatomines", "", "", "en"
    ) == ("public_health", None)
    assert acquisition.classify_science_topic(
        "Water, sanitation, and hygiene in schistosomiasis control: A scoping review of evidence gaps across transmission pathways",
        "The review includes studies of agricultural water systems.", "", "en"
    ) == ("public_health", None)


def test_synthetic_biology_function_study_respects_biological_topic_limit():
    assert acquisition.classify_science_topic(
        "Precision engineering of biological function with large-scale measurements and machine learning", "", "", "en"
    ) == ("biological_science", None)


def test_jats_intro_paragraph_does_not_hide_following_sections():
    root = ET.fromstring(
        """<article><body>
          <p>Introductory context.</p>
          <sec><title>First section</title><p>First evidence block.</p></sec>
          <sec><title>Second section</title><p>Second evidence block.</p></sec>
        </body></article>"""
    )

    sections, _, _ = acquisition.extract_body(root)

    assert [section["section_path"] for section in sections] == [
        ["Body"], ["First section"], ["Second section"]
    ]
    assert "Introductory context." in sections[0]["text"]
    assert "Second evidence block." in sections[2]["text"]


def test_plos_solr_metadata_hint_does_not_substitute_for_governing_article_rights():
    candidate = {
        "id": "science-plos-journal-pone-1234567",
        "doi": "10.1371/journal.pone.1234567",
        "title": "Forest ecosystem restoration evidence",
        "publisher_article_url": "https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.1234567",
        "search_group": "plos_environment_ecology",
        "metadata_license_hint": {
            "id": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "statement": "Creative Commons Attribution 4.0 International",
        },
    }
    jats = b"""<article><front><article-meta>
      <article-title>Forest ecosystem restoration evidence</article-title>
      <article-id pub-id-type="doi">10.1371/journal.pone.1234567</article-id>
      <contrib-group><contrib contrib-type="author"><name><given-names>Ada</given-names><surname>Example</surname></name></contrib></contrib-group>
    </article-meta></front><body>
      <sec><title>Methods</title><p>Methods are documented here.</p></sec>
      <sec><title>Results</title><p>Results are documented here.</p></sec>
    </body></article>"""

    record, identity, reason = acquisition.parse_plos_science_document(
        jats, candidate, {"retrieved_at": "2026-10-03T12:00:00+00:00", "sha256": "fixture"}
    )

    assert record is None
    assert not identity["same_item_rights_verified"]
    assert reason == "solr_jats_item_license_mismatch_or_conflict"


def test_plos_record_uses_only_front_matter_pmcid_and_never_synthetic_id():
    paragraphs = " ".join(["A bounded field study records soil moisture, vegetation cover, and restoration outcomes across comparable forest plots."] * 35)
    jats = f"""<article xml:lang="en" xmlns:xlink="http://www.w3.org/1999/xlink"><front><journal-meta><journal-title>PLOS ONE</journal-title></journal-meta><article-meta>
      <article-title>Forest ecosystem restoration evidence</article-title>
      <article-id pub-id-type="doi">10.1371/journal.pone.1234567</article-id>
      <article-id pub-id-type="pmc">PMC1234567</article-id>
      <contrib-group><contrib contrib-type="author"><name><given-names>Ada</given-names><surname>Example</surname></name></contrib></contrib-group>
      <permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/"><license-p>Creative Commons Attribution 4.0 International</license-p></license></permissions>
    </article-meta></front><body>
      <sec><title>Methods</title><p>{paragraphs}</p></sec>
      <sec><title>Results</title><p>{paragraphs}</p></sec>
    </body><back><ref-list><ref id="r1"><mixed-citation>One cited study.</mixed-citation></ref></ref-list></back></article>""".encode()
    candidate = {
        "id": "science-plos-journal-pone-1234567", "doi": "10.1371/journal.pone.1234567",
        "title": "Forest ecosystem restoration evidence", "journal_slug": "plosone",
        "publisher_article_url": "https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.1234567",
        "search_group": "plos_environment_ecology", "published_at": "2026-01",
        "metadata_license_hint": {"id": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
    }

    record, identity, reason = acquisition.parse_plos_science_document(
        jats, candidate, {"retrieved_at": "2026-10-03T12:00:00+00:00", "raw_file": "article.xml", "sha256": "fixture", "response_bytes_decoded": len(jats), "http_status": 200, "url": "https://journals.plos.org/plosone/article/file"}
    )

    assert reason == "eligible_proof_pending_adapter"
    assert record is not None
    assert record["id"] == candidate["id"]
    assert record["pmcid"] == "PMC1234567"
    assert record["relation_keys"] == ["doi:10.1371/journal.pone.1234567", "pmcid:PMC1234567"]
    assert "PMCPLOS" not in record["revision"]
    assert identity["same_item_rights_verified"]
