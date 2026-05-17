"""Unit tests for metadata_frontend.

Run with:  cd paper2md && python -m pytest tests/test_metadata_frontend.py -v

Network-touching tests (the API queries) are skipped by default; pass
`--run-net` or set $PAPER2MD_RUN_NET=1 to exercise them.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Allow `import metadata_frontend` when running pytest from the package root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import metadata_frontend as mf  # noqa: E402


# ---------------------------------------------------------------------------
# License normalisation table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("https://creativecommons.org/licenses/by/4.0/", "cc-by-4.0"),
    ("https://creativecommons.org/licenses/by-nc/4.0", "cc-by-nc-4.0"),
    ("https://creativecommons.org/licenses/by-nc-sa/3.0/", "cc-by-nc-sa-3.0"),
    ("https://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
    ("https://creativecommons.org/publicdomain/mark/1.0/", "public-domain"),
    ("https://arxiv.org/licenses/nonexclusive-distrib/1.0/", "arxiv-default"),
    ("cc-by-4.0", "cc-by-4.0"),
    ("CC0", "cc0"),
    ("publicdomain", "public-domain"),
    ("mit", "mit"),
    ("", None),
    (None, None),
    ("garbage-license", None),
])
def test_normalise_license(raw, expected):
    assert mf._normalise_license(raw) == expected


# ---------------------------------------------------------------------------
# Safety classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("license_slug, expected", [
    ("cc0", "true"),
    ("public-domain", "true"),
    ("cc-by", "true"),
    ("cc-by-4.0", "true"),
    ("cc-by-sa-3.0", "true"),
    ("mit", "true"),
    ("cc-by-nc", "restricted"),
    ("cc-by-nc-4.0", "restricted"),
    ("cc-by-nd", "restricted"),
    ("cc-by-nc-sa-4.0", "restricted"),
    ("cc-by-nc-nd-4.0", "restricted"),
    # readable: publicly readable but not redistributable
    ("arxiv-default", "readable"),
    ("arxiv-pre-2004", "readable"),
    ("pmc-author-manuscript", "readable"),
    ("green-oa-no-license", "readable"),
    # public-domain-us: synthetic slug for the US 95-year rolling cutoff
    ("public-domain-us", "true"),
    # false: paywalled / unknown / publisher-TDM-only
    ("elsevier-tdm", "false"),
    (None, "false"),
    ("", "false"),
])
def test_classify_safety(license_slug, expected):
    assert mf._classify_safety(license_slug) == expected


@pytest.mark.parametrize("raw, expected", [
    ("pmc-author-manuscript", "pmc-author-manuscript"),
    ("green-oa-no-license", "green-oa-no-license"),
    ("public-domain-us", "public-domain-us"),
])
def test_synthetic_slugs_round_trip(raw, expected):
    """Synthetic slugs the resolver assigns must survive normalize() so
    they aren't silently dropped when round-tripped through the cache."""
    assert mf._normalise_license(raw) == expected


# ---------------------------------------------------------------------------
# DOI cleaning / blocklist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("10.1038/s41586-023-06924-6", "10.1038/s41586-023-06924-6"),
    ("10.1038/s41586-023-06924-6.", "10.1038/s41586-023-06924-6"),
    ("10.1038/s41586-023-06924-6).", "10.1038/s41586-023-06924-6"),
    ("10.1038/S41586-023-06924-6", "10.1038/s41586-023-06924-6"),
    ("10.5555/example.placeholder", None),  # blocklisted
    ("10.0/some-test", None),
])
def test_clean_doi(raw, expected):
    assert mf._clean_doi(raw) == expected


# ---------------------------------------------------------------------------
# DOI regex covers expected forms in PDF text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("doi: 10.1038/nature12345 published 2024", "10.1038/nature12345"),
    ("https://doi.org/10.1093/nar/gkad123", "10.1093/nar/gkad123"),
    ("DOI: 10.1109/TPAMI.2024.001234.", "10.1109/TPAMI.2024.001234"),
    ("no DOI here", None),
])
def test_doi_regex(text, expected):
    m = mf.DOI_RE.search(text)
    if expected is None:
        assert m is None
    else:
        assert m is not None
        assert mf._clean_doi(m.group(0)) == expected.lower()


# ---------------------------------------------------------------------------
# Mocked OpenAlex / Unpaywall / arXiv flows
# ---------------------------------------------------------------------------

class _StubResp:
    def __init__(self, status_code=200, json_payload=None, text=""):
        self.status_code = status_code
        self._json = json_payload or {}
        self.text = text
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return self._json

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def _mock_session(*responses):
    """Build a Session-like object whose .get returns the queue of responses
    in order. Each entry can be a _StubResp or a callable that takes the URL
    and returns a _StubResp."""
    queue = list(responses)

    class _S:
        def get(self, url, **kw):
            if not queue:
                return _StubResp(status_code=404)
            item = queue.pop(0)
            if callable(item):
                return item(url)
            return item

        def close(self):
            pass

    return _S()


def test_resolve_openalex_cc_by(tmp_path):
    """OpenAlex returns license + OA URL → confidence=high, safe=true."""
    # Build a minimal PDF that contains the DOI in a text page.
    pdf = _make_pdf_with_text(tmp_path / "paper.pdf",
                              "Title\n\nDOI: 10.1371/journal.pone.0000001\n")
    openalex_payload = {
        "title": "A test paper",
        "authorships": [{"author": {"display_name": "Jane Doe"}}],
        "open_access": {"oa_status": "gold",
                        "oa_url": "https://example.org/paper.pdf"},
        "primary_location": {
            "license": "https://creativecommons.org/licenses/by/4.0/",
        },
    }
    session = _mock_session(_StubResp(json_payload=openalex_payload))
    with patch.object(mf.requests, "Session", return_value=session):
        m = mf.resolve(pdf, allow_network=True, prefer_oa=False,
                       cache_path=tmp_path / "cache.json", timeout_s=2.0)
    assert m.doi == "10.1371/journal.pone.0000001"
    assert m.license == "cc-by-4.0"
    assert m.safe_to_distribute == "true"
    assert m.confidence == "high"
    assert m.resolved_via == "openalex"
    assert m.oa_status == "gold"


def test_resolve_unpaywall_fallback(tmp_path):
    """OpenAlex misses license → Unpaywall fills it."""
    pdf = _make_pdf_with_text(tmp_path / "paper.pdf",
                              "DOI: 10.1234/example\n")
    openalex_payload = {  # no license field
        "title": "X",
        "authorships": [],
        "open_access": {"oa_status": "bronze", "oa_url": None},
    }
    unpaywall_payload = {
        "is_oa": True,
        "best_oa_location": {
            "url_for_pdf": "https://repo.example/paper.pdf",
            "license": "cc-by-nc-4.0",
        },
    }
    session = _mock_session(
        _StubResp(json_payload=openalex_payload),
        _StubResp(json_payload=unpaywall_payload),
    )
    with patch.object(mf.requests, "Session", return_value=session), \
         patch.dict(os.environ, {"UNPAYWALL_EMAIL": "t@example.org"}):
        m = mf.resolve(pdf, allow_network=True, prefer_oa=False,
                       cache_path=tmp_path / "cache.json", timeout_s=2.0)
    assert m.license == "cc-by-nc-4.0"
    assert m.safe_to_distribute == "restricted"
    assert m.resolved_via == "unpaywall"


def test_resolve_offline_no_doi(tmp_path):
    """No network, no DOI → low-confidence, unsafe, doesn't crash."""
    pdf = _make_pdf_with_text(tmp_path / "paper.pdf", "Untitled paper.\n")
    m = mf.resolve(pdf, allow_network=False,
                   cache_path=tmp_path / "cache.json")
    assert m.doi is None
    assert m.license is None
    assert m.safe_to_distribute == "false"
    assert m.confidence == "low"
    assert "network-disabled" in m.provenance


def test_resolve_pmc_author_manuscript(tmp_path):
    """OpenAlex says green/no-license; Europe PMC has PDF but isOpenAccess=N
    (NIH/NASA-deposited author manuscript). Should classify as 'readable'
    with the synthetic pmc-author-manuscript license."""
    pdf = _make_pdf_with_text(tmp_path / "paper.pdf",
                              "DOI: 10.1038/example.test\n")
    openalex_payload = {
        "title": "X",
        "authorships": [],
        "open_access": {
            "oa_status": "green",
            "oa_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/",
        },
    }
    europepmc_payload = {
        "resultList": {"result": [{
            "isOpenAccess": "N",
            "hasPDF": "Y",
            "pmcid": "PMC123",
            "license": None,
        }]},
    }
    # green oa_status with a PMC URL routes through OpenAlex -> Unpaywall
    # (skipped without email) -> Europe PMC.
    session = _mock_session(
        _StubResp(json_payload=openalex_payload),
        _StubResp(json_payload=europepmc_payload),
    )
    with patch.object(mf.requests, "Session", return_value=session), \
         patch.dict(os.environ, {}, clear=True):  # no UNPAYWALL_EMAIL
        m = mf.resolve(pdf, allow_network=True, prefer_oa=False,
                       cache_path=tmp_path / "cache.json", timeout_s=2.0)
    assert m.license == "pmc-author-manuscript"
    assert m.safe_to_distribute == "readable"
    assert m.resolved_via == "europepmc"
    assert m.oa_pdf_url == "https://europepmc.org/articles/PMC123?pdf=render"


def test_resolve_green_oa_no_license_fallback(tmp_path):
    """OpenAlex returns oa_status=bronze with a PDF URL but no license; no
    other API has anything. Should default to green-oa-no-license / readable."""
    pdf = _make_pdf_with_text(tmp_path / "paper.pdf",
                              "DOI: 10.1234/example.bronze\n")
    openalex_payload = {
        "title": "Y",
        "authorships": [],
        "open_access": {"oa_status": "bronze",
                        "oa_url": "https://repo.example.org/paper.pdf"},
    }
    session = _mock_session(_StubResp(json_payload=openalex_payload))
    with patch.object(mf.requests, "Session", return_value=session), \
         patch.dict(os.environ, {}, clear=True):  # no UNPAYWALL_EMAIL
        m = mf.resolve(pdf, allow_network=True, prefer_oa=False,
                       cache_path=tmp_path / "cache.json", timeout_s=2.0)
    assert m.license == "green-oa-no-license"
    assert m.safe_to_distribute == "readable"
    assert m.oa_pdf_url == "https://repo.example.org/paper.pdf"


def test_resolve_osti_pages_hit(tmp_path):
    """A DOE-funded paper: OpenAlex says closed and gives no license;
    Europe PMC misses; OSTI returns a record matching the DOI with a
    fulltext link. Should set license=osti-public-access (readable
    tier) and use OSTI's URL as oa_pdf_url."""
    pdf = _make_pdf_with_text(tmp_path / "paper.pdf",
                              "DOI: 10.2172/example.doe\n")
    openalex_payload = {  # closed access, no license
        "title": "DOE-funded study",
        "authorships": [],
        "open_access": {"oa_status": "closed", "oa_url": None},
    }
    osti_payload = [
        {"osti_id": "12345",
         "doi": "10.2172/example.doe",
         "title": "DOE-funded study",
         "article_type": "Accepted Manuscript",
         "links": [{"rel": "fulltext",
                    "url": "https://www.osti.gov/pages/biblio/12345"}]},
    ]
    # OpenAlex (call 1) -> Unpaywall skipped (no email) -> Europe PMC
    # (call 2 — empty result) -> OSTI q-search (call 3, hit).
    session = _mock_session(
        _StubResp(json_payload=openalex_payload),
        _StubResp(json_payload={"resultList": {"result": []}}),
        _StubResp(json_payload=osti_payload),
    )
    with patch.object(mf.requests, "Session", return_value=session), \
         patch.dict(os.environ, {}, clear=True):
        m = mf.resolve(pdf, allow_network=True, prefer_oa=False,
                       cache_path=tmp_path / "cache.json", timeout_s=2.0)
    assert m.license == "osti-public-access"
    assert m.safe_to_distribute == "readable"
    assert m.oa_pdf_url == "https://www.osti.gov/pages/biblio/12345"
    assert m.oa_status == "green"
    assert m.resolved_via == "osti"
    assert "osti-ok" in m.provenance


def test_resolve_osti_pages_doi_mismatch_skips(tmp_path):
    """OSTI returns records but none has a matching DOI -> miss; no
    metadata change. Defends against picking up similar-titled
    unrelated PAGES records."""
    pdf = _make_pdf_with_text(tmp_path / "paper.pdf",
                              "DOI: 10.2172/example.real\n")
    openalex_payload = {
        "title": "X",
        "authorships": [],
        "open_access": {"oa_status": "closed", "oa_url": None},
    }
    # OSTI returns records, but the DOI doesn't match.
    osti_q_payload = [
        {"osti_id": "999",
         "doi": "10.2172/some.OTHER.paper",
         "title": "Different paper",
         "links": [{"rel": "fulltext", "url": "https://example.bad/"}]},
    ]
    osti_title_payload = []  # empty title-search fallback
    session = _mock_session(
        _StubResp(json_payload=openalex_payload),
        _StubResp(json_payload={"resultList": {"result": []}}),
        _StubResp(json_payload=osti_q_payload),
        _StubResp(json_payload=osti_title_payload),
    )
    with patch.object(mf.requests, "Session", return_value=session), \
         patch.dict(os.environ, {}, clear=True):
        m = mf.resolve(pdf, allow_network=True, prefer_oa=False,
                       cache_path=tmp_path / "cache.json", timeout_s=2.0)
    assert m.license is None or m.license == "green-oa-no-license"
    # license shouldn't be osti-public-access; the wrong-DOI record
    # was correctly skipped
    assert m.license != "osti-public-access"
    assert "osti-miss" in m.provenance


def test_classify_osti_public_access_is_readable():
    """The new synthetic slug must classify into the readable tier
    (publicly readable but not redistributable; same legal posture as
    pmc-author-manuscript)."""
    assert mf._classify_safety("osti-public-access") == "readable"
    # Round-trips through normalize:
    assert mf._normalise_license("osti-public-access") == "osti-public-access"


def test_article_metadata_to_dict_round_trip():
    """ArticleMetadata.to_dict carries every field that to_yaml_lines emits,
    plus provenance. Verifies the .meta.json sidecar is a faithful mirror
    of the YAML 'copyright:' block."""
    m = mf.ArticleMetadata(
        doi="10.1073/pnas.6.8.449",
        title="An ancient paper",
        authors=["George P. Merrill"],
        year=1920,
        license="public-domain-us",
        license_url="https://www.copyright.gov/help/faq/faq-duration.html",
        safe_to_distribute="true",
        confidence="medium",
        oa_status="closed",
        resolved_via="us-95-year-rule",
    )
    m.provenance.append("public-domain-us:year=1920")
    d = m.to_dict()
    assert d["doi"] == "10.1073/pnas.6.8.449"
    assert d["year"] == 1920
    assert d["license"] == "public-domain-us"
    assert d["safe_to_distribute"] == "true"
    assert d["resolved_via"] == "us-95-year-rule"
    assert d["authors"] == ["George P. Merrill"]
    assert d["oa_pdf_local"] is None  # path not set
    assert "public-domain-us:year=1920" in d["provenance"]
    # JSON-serialisable
    import json as _json
    assert _json.loads(_json.dumps(d))["doi"] == "10.1073/pnas.6.8.449"


def test_first_author_surname():
    """Surname extraction handles 'Given Family' and 'Family, Given'
    forms; empty list returns None."""
    assert mf._first_author_surname([]) is None
    assert mf._first_author_surname([""]) is None
    assert mf._first_author_surname(["Jerry Wackerle"]) == "Wackerle"
    assert mf._first_author_surname(["Wackerle, Jerry"]) == "Wackerle"
    assert mf._first_author_surname(["Madonna"]) == "Madonna"
    assert (mf._first_author_surname(["John P. Morgan"])
            == "Morgan")


def test_crossref_year_helper():
    """_crossref_year prefers `published`, falls back through `issued` /
    `published-print` / `published-online`. Returns None when no year is
    present or the data is malformed."""
    assert mf._crossref_year({"published": {"date-parts": [[1925, 6, 1]]}}) == 1925
    assert mf._crossref_year({"issued": {"date-parts": [[1962]]}}) == 1962
    assert mf._crossref_year({"published-print": {"date-parts": [[2020, 1]]}}) == 2020
    assert mf._crossref_year({}) is None
    assert mf._crossref_year({"published": {"date-parts": []}}) is None
    assert mf._crossref_year({"published": {"date-parts": [["junk"]]}}) is None


def test_resolve_public_domain_us_pre1930(tmp_path):
    """OpenAlex returns publication_year=1925 with no license. The 95-year
    rule fires after the API chain: license=public-domain-us, safe=true,
    confidence=high, year emitted in YAML."""
    pdf = _make_pdf_with_text(tmp_path / "paper.pdf",
                              "DOI: 10.1234/old.paper\n")
    openalex_payload = {
        "title": "An ancient paper",
        "authorships": [],
        "publication_year": 1925,
        "open_access": {"oa_status": "closed", "oa_url": None},
    }
    # OpenAlex hit; Unpaywall skipped (no email); Europe PMC miss; OSTI miss.
    session = _mock_session(
        _StubResp(json_payload=openalex_payload),
        _StubResp(json_payload={"resultList": {"result": []}}),
        _StubResp(json_payload=[]),  # OSTI q-search empty
    )
    with patch.object(mf.requests, "Session", return_value=session), \
         patch.dict(os.environ, {}, clear=True):
        m = mf.resolve(pdf, allow_network=True, prefer_oa=False,
                       cache_path=tmp_path / "cache.json", timeout_s=2.0)
    assert m.year == 1925
    assert m.license == "public-domain-us"
    assert m.safe_to_distribute == "true"
    assert m.resolved_via == "us-95-year-rule"
    assert any("public-domain-us:year=1925" in p for p in m.provenance)
    yaml_lines = m.to_yaml_lines()
    assert any("year: 1925" in line for line in yaml_lines)


def test_resolve_public_domain_us_recent_year_no_op(tmp_path):
    """OpenAlex returns publication_year=2020 with no license. The 95-year
    rule MUST NOT fire; license stays unresolved. Defends against falsely
    flagging modern papers as PD."""
    pdf = _make_pdf_with_text(tmp_path / "paper.pdf",
                              "DOI: 10.1234/recent.paper\n")
    openalex_payload = {
        "title": "Recent paper",
        "authorships": [],
        "publication_year": 2020,
        "open_access": {"oa_status": "closed", "oa_url": None},
    }
    session = _mock_session(
        _StubResp(json_payload=openalex_payload),
        _StubResp(json_payload={"resultList": {"result": []}}),
        _StubResp(json_payload=[]),
    )
    with patch.object(mf.requests, "Session", return_value=session), \
         patch.dict(os.environ, {}, clear=True):
        m = mf.resolve(pdf, allow_network=True, prefer_oa=False,
                       cache_path=tmp_path / "cache.json", timeout_s=2.0)
    assert m.year == 2020
    assert m.license is None
    assert m.safe_to_distribute == "false"
    assert m.resolved_via != "us-95-year-rule"


def test_resolve_public_domain_us_overrides_pmc_author_manuscript(tmp_path):
    """A 1920 paper that ended up with the synthetic 'pmc-author-manuscript'
    fallback (because Europe PMC hosts it) should be promoted to
    'public-domain-us' / true: the PMC manuscript posture is a 'no license
    declared' fallback, but copyright on the underlying 1920 work has
    expired. Real CC licenses are still respected by the next test."""
    pdf = _make_pdf_with_text(tmp_path / "paper.pdf",
                              "DOI: 10.1073/pnas.6.8.449\n")
    openalex_payload = {  # OpenAlex returns publication_year, no license
        "title": "An ancient paper hosted by PMC",
        "authorships": [],
        "publication_year": 1920,
        "open_access": {"oa_status": "green",
                        "oa_url": "https://www.ncbi.nlm.nih.gov/pmc/x/"},
    }
    europepmc_payload = {
        "resultList": {"result": [{
            "isOpenAccess": "N",
            "hasPDF": "Y",
            "pmcid": "PMC1084590",
            "license": None,
            "pubYear": "1920",
        }]},
    }
    session = _mock_session(
        _StubResp(json_payload=openalex_payload),
        _StubResp(json_payload=europepmc_payload),
    )
    with patch.object(mf.requests, "Session", return_value=session), \
         patch.dict(os.environ, {}, clear=True):
        m = mf.resolve(pdf, allow_network=True, prefer_oa=False,
                       cache_path=tmp_path / "cache.json", timeout_s=2.0)
    assert m.year == 1920
    assert m.license == "public-domain-us"
    assert m.safe_to_distribute == "true"
    assert m.resolved_via == "us-95-year-rule"
    assert any("override pmc-author-manuscript" in p for p in m.provenance)


def test_resolve_public_domain_us_does_not_override_real_license(tmp_path):
    """A pre-1930 paper that *does* have a CC license declared keeps the
    real license; the 95-year rule only fires when license is unresolved."""
    pdf = _make_pdf_with_text(tmp_path / "paper.pdf",
                              "DOI: 10.1234/cc.old\n")
    openalex_payload = {
        "title": "Old paper with explicit license",
        "authorships": [],
        "publication_year": 1900,
        "open_access": {"oa_status": "gold",
                        "oa_url": "https://example.org/x.pdf"},
        "primary_location": {
            "license": "https://creativecommons.org/licenses/by/4.0/",
        },
    }
    session = _mock_session(_StubResp(json_payload=openalex_payload))
    with patch.object(mf.requests, "Session", return_value=session):
        m = mf.resolve(pdf, allow_network=True, prefer_oa=False,
                       cache_path=tmp_path / "cache.json", timeout_s=2.0)
    assert m.year == 1900
    assert m.license == "cc-by-4.0"
    assert m.resolved_via == "openalex"
    assert m.safe_to_distribute == "true"


def test_cache_round_trip(tmp_path):
    """Cached DOI -> result is reused without making API calls."""
    pdf = _make_pdf_with_text(tmp_path / "paper.pdf",
                              "DOI: 10.9999/cached\n")
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({
        "10.9999/cached": {
            "doi": "10.9999/cached",
            "license": "cc-by-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "safe_to_distribute": "true",
            "confidence": "high",
            "oa_status": "gold",
            "resolved_via": "openalex",
        }
    }))
    # Empty session -- if anything tries an API call, the test will fail with
    # a 404 but the cache should short-circuit before that.
    session = _mock_session()  # all 404
    with patch.object(mf.requests, "Session", return_value=session):
        m = mf.resolve(pdf, allow_network=True,
                       cache_path=cache, timeout_s=2.0)
    assert m.license == "cc-by-4.0"
    assert m.safe_to_distribute == "true"
    assert "cache-hit" in m.provenance


# ---------------------------------------------------------------------------
# Journal slug resolution (DOI prefix -> publisher / sub-journal)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("doi, expected", [
    # APS — sub-journal carried in the DOI suffix.
    ("10.1103/PhysRevLett.103.225501", "aps-prl"),
    ("10.1103/PhysRevB.88.184107", "aps-prb"),
    ("10.1103/PhysRevX.10.011017", "aps-prx"),
    ("10.1103/PhysRevA.97.012523", "aps-pra"),
    ("10.1103/RevModPhys.84.45", "aps-rmp"),
    ("10.1103/SomeOtherJournal.x.y", "aps"),
    # Science family.
    ("10.1126/science.1157846", "science"),
    ("10.1126/sciadv.abc1234", "science-advances"),
    # Nature family.
    ("10.1038/s41586-024-07000-z", "nature"),
    ("10.1038/s41561-024-09999-z", "nature-geoscience"),
    ("10.1038/s41550-024-12345-y", "nature-astronomy"),
    ("10.1038/nature14365", "nature"),
    ("10.1038/ngeo2614", "nature-geoscience"),
    ("10.1038/somethingelse", "nature"),
    # Elsevier — sub-journal in the j.<journal> suffix.
    ("10.1016/j.icarus.2026.116970", "elsevier-icarus"),
    ("10.1016/j.epsl.2024.118735", "elsevier-epsl"),
    ("10.1016/j.gca.2023.05.001", "elsevier-gca"),
    ("10.1016/j.something.2024.001", "elsevier"),
    # Society / publisher catch-alls.
    ("10.1029/2009JE003539", "agu"),
    ("10.1073/pnas.6.8.449", "pnas"),
    ("10.1063/1.5135507", "aip"),
    ("10.1098/rsta.2023.0012", "royal-society"),
    ("10.5194/cp-19-1-2023", "copernicus"),
    ("10.1093/mnras/stab1234", "oup"),
    ("10.1002/wcms.1493", "wiley"),
    ("10.3847/1538-4357/abcdef", "iop-aas"),
    ("10.1088/0004-637X/123/4/56", "iop"),
    # Case-insensitive match.
    ("10.1103/PHYSREVLETT.103.225501", "aps-prl"),
    # Misses.
    ("10.9999/unknown.123", None),
    ("not-a-doi", None),
    ("", None),
    (None, None),
])
def test_resolve_journal_slug(doi, expected):
    assert mf._resolve_journal_slug(doi) == expected


def test_journal_slug_populated_offline(tmp_path):
    """resolve() with allow_network=False still sets journal_slug from
    a DOI extracted from the PDF text."""
    pdf = _make_pdf_with_text(
        tmp_path / "test.pdf",
        "Title\n\ndoi.org/10.1126/science.1157846\n\nAbstract...",
    )
    m = mf.resolve(pdf, allow_network=False,
                   cache_path=tmp_path / "cache.json")
    assert m.doi == "10.1126/science.1157846"
    assert m.journal_slug == "science"
    assert "journal-slug:science" in m.provenance


def test_journal_slug_in_manifest_dict():
    m = mf.ArticleMetadata(
        doi="10.1103/physrevlett.103.225501",
        journal_slug="aps-prl",
        license="cc-by-4.0",
    )
    d = m.manifest_dict()
    assert d["doi"] == "10.1103/physrevlett.103.225501"
    assert d["journal_slug"] == "aps-prl"


def test_journal_slug_in_yaml():
    m = mf.ArticleMetadata(
        doi="10.1016/j.icarus.2026.116970",
        journal_slug="elsevier-icarus",
        license="cc-by-4.0",
    )
    body = "\n".join(m.to_yaml_lines())
    assert "journal_slug: elsevier-icarus" in body


def test_journal_slug_in_to_dict():
    m = mf.ArticleMetadata(
        doi="10.1029/2009JE003539",
        journal_slug="agu",
    )
    d = m.to_dict()
    assert d["journal_slug"] == "agu"


# ---------------------------------------------------------------------------
# Reference-list fetch (Crossref -> OpenAlex)
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status_code: int = 200, payload: dict = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, route_table: dict):
        """route_table: dict mapping URL substring -> _FakeResp."""
        self.route_table = route_table
        self.calls: list[str] = []

    def get(self, url, timeout=None, headers=None):
        self.calls.append(url)
        for needle, resp in self.route_table.items():
            if needle in url:
                return resp
        return _FakeResp(status_code=404, payload={})


def test_format_crossref_unstructured_used_when_present():
    entry = {"unstructured": "Smith J. et al., Phys Rev B 1, 1 (2020)."}
    out = mf._format_crossref_ref(entry, 7)
    assert out == "- 7. Smith J. et al., Phys Rev B 1, 1 (2020)."


def test_format_crossref_composes_when_unstructured_missing():
    entry = {
        "author": "Smith",
        "year": "2020",
        "article-title": "On a topic",
        "journal-title": "Phys Rev B",
        "volume": "1",
        "first-page": "1",
        "DOI": "10.1234/x",
    }
    out = mf._format_crossref_ref(entry, 3)
    assert out.startswith("- 3. Smith, (2020), On a topic, *Phys Rev B*")
    assert "10.1234/x" in out


def test_format_crossref_handles_minimal_entry():
    out = mf._format_crossref_ref({}, 1)
    assert out == "- 1. (incomplete reference)"


def test_format_openalex_with_authors_and_venue():
    work = {
        "display_name": "A study of stuff",
        "publication_year": 2021,
        "authorships": [
            {"author": {"display_name": "Alice Author"}},
            {"author": {"display_name": "Bob Coauthor"}},
        ],
        "host_venue": {"display_name": "Nature"},
        "doi": "https://doi.org/10.1038/n.1234",
    }
    out = mf._format_openalex_ref(work, 5)
    assert out.startswith("- 5. Alice Author, Bob Coauthor")
    assert "*Nature*" in out
    assert "10.1038/n.1234" in out


def test_format_openalex_truncates_long_author_lists():
    work = {
        "display_name": "Title",
        "authorships": [{"author": {"display_name": f"A{i}"}} for i in range(10)],
    }
    out = mf._format_openalex_ref(work, 1)
    assert "et al." in out


def test_query_crossref_refs_returns_list():
    payload = {
        "message": {
            "reference": [
                {"unstructured": "ref 1"},
                {"unstructured": "ref 2"},
            ],
        }
    }
    session = _FakeSession({"api.crossref.org": _FakeResp(200, payload)})
    refs = mf._query_crossref_refs("10.1/x", None, 5.0, session)
    assert refs is not None
    assert len(refs) == 2


def test_query_crossref_refs_returns_none_on_404():
    session = _FakeSession({"api.crossref.org": _FakeResp(404, {})})
    assert mf._query_crossref_refs("10.1/x", None, 5.0, session) is None


def test_query_crossref_refs_returns_none_on_empty_field():
    session = _FakeSession({
        "api.crossref.org": _FakeResp(200, {"message": {}}),
    })
    assert mf._query_crossref_refs("10.1/x", None, 5.0, session) is None


def test_query_openalex_refs_batches_and_orders():
    work_payload = {
        "id": "https://openalex.org/W0",
        "referenced_works": [
            "https://openalex.org/W1",
            "https://openalex.org/W2",
            "https://openalex.org/W3",
        ],
    }
    batch_payload = {
        "results": [
            {"id": "https://openalex.org/W3", "display_name": "Three"},
            {"id": "https://openalex.org/W1", "display_name": "One"},
            {"id": "https://openalex.org/W2", "display_name": "Two"},
        ],
    }

    class _OARouter:
        def __init__(self):
            self.calls = []

        def get(self, url, timeout=None, headers=None):
            self.calls.append(url)
            if "doi:" in url:
                return _FakeResp(200, work_payload)
            if "filter=ids.openalex" in url:
                return _FakeResp(200, batch_payload)
            return _FakeResp(404, {})

    session = _OARouter()
    refs = mf._query_openalex_refs("10.1/x", None, 5.0, session)
    assert refs is not None
    assert [r["display_name"] for r in refs] == ["One", "Two", "Three"]


def test_fetch_references_no_doi_returns_none(tmp_path):
    cache = tmp_path / "refs.json"
    assert mf.fetch_references("", cache_path=cache) is None


def test_fetch_references_network_disabled_with_no_cache(tmp_path):
    cache = tmp_path / "refs.json"
    assert mf.fetch_references("10.1/x", allow_network=False,
                               cache_path=cache) is None


def test_fetch_references_uses_cache(tmp_path):
    cache = tmp_path / "refs.json"
    payload = {
        "10.1/x": {
            "source": "crossref",
            "refs": ["- 1. cached entry."],
            "fetched_at": "2026-05-04T00:00:00Z",
        }
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload))
    result = mf.fetch_references("10.1/x", allow_network=False,
                                 cache_path=cache)
    assert result is not None
    assert result["source"] == "crossref"
    assert result["refs"] == ["- 1. cached entry."]


def test_fetch_references_crossref_success_writes_cache(tmp_path):
    cache = tmp_path / "refs.json"
    cr_payload = {
        "message": {
            "reference": [
                {"unstructured": "Author A. 2020. Title. Journal 1, 1."},
                {"unstructured": "Author B. 2021. Title. Journal 2, 2."},
            ]
        }
    }
    fake_session = _FakeSession({
        "api.crossref.org": _FakeResp(200, cr_payload),
    })
    with patch.object(mf.requests, "Session", return_value=fake_session):
        result = mf.fetch_references("10.1/x", cache_path=cache)
    assert result is not None
    assert result["source"] == "crossref"
    assert len(result["refs"]) == 2
    assert "Author A" in result["refs"][0]
    assert cache.exists()
    cached = json.loads(cache.read_text())
    assert "10.1/x" in cached


def test_fetch_references_falls_back_to_openalex(tmp_path):
    cache = tmp_path / "refs.json"

    class _Router:
        def __init__(self):
            self.calls = []

        def get(self, url, timeout=None, headers=None):
            self.calls.append(url)
            if "api.crossref.org" in url:
                return _FakeResp(200, {"message": {}})
            if "doi:" in url and "openalex" in url:
                return _FakeResp(200, {
                    "id": "https://openalex.org/W0",
                    "referenced_works": ["https://openalex.org/W1"],
                })
            if "filter=ids.openalex" in url:
                return _FakeResp(200, {
                    "results": [{
                        "id": "https://openalex.org/W1",
                        "display_name": "OA Source",
                        "publication_year": 2019,
                    }],
                })
            return _FakeResp(404, {})

    fake_session = _Router()
    with patch.object(mf.requests, "Session", return_value=fake_session):
        result = mf.fetch_references("10.1/x", cache_path=cache)
    assert result is not None
    assert result["source"] == "openalex"
    assert len(result["refs"]) == 1
    assert "OA Source" in result["refs"][0]


def test_fetch_references_returns_none_when_both_apis_empty(tmp_path):
    cache = tmp_path / "refs.json"
    fake_session = _FakeSession({})

    with patch.object(mf.requests, "Session", return_value=fake_session):
        assert mf.fetch_references("10.1/x", cache_path=cache) is None
    assert not cache.exists()


def test_fetch_references_handles_network_exception(tmp_path):
    cache = tmp_path / "refs.json"

    class _Boom:
        def get(self, *a, **kw):
            raise mf.requests.ConnectionError("offline")

    with patch.object(mf.requests, "Session", return_value=_Boom()):
        assert mf.fetch_references("10.1/x", cache_path=cache) is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf_with_text(path: Path, text: str) -> Path:
    """Build a minimal one-page PDF with the given text. Used by tests that
    need the DOI extractor to find something."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path
