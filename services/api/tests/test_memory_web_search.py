import unittest
from unittest.mock import patch


class FakeResponse:
    def __init__(self, data=None, text=""):
        self._data = data
        self.text = text

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


class FakeAsyncClient:
    gets = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params):
        self.gets.append({"url": url, "params": params})
        return FakeResponse(
            {
                "Heading": "Lung cancer",
                "AbstractText": "Lung cancer is a disease with tumor microenvironment interactions.",
                "AbstractURL": "https://example.org/lung-cancer",
                "RelatedTopics": [
                    {
                        "Topics": [
                            {
                                "Text": "Tumor microenvironment context",
                                "FirstURL": "https://example.org/tme",
                            }
                        ]
                    }
                ],
            }
        )


class FakeEutilsAsyncClient:
    gets = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params):
        self.gets.append({"url": url, "params": params})
        db = params["db"]
        if url.endswith("/esearch.fcgi"):
            ids = ["123"] if db == "pubmed" else ["456"]
            return FakeResponse({"esearchresult": {"idlist": ids}})
        if db == "pubmed":
            return FakeResponse(
                text="""
                <PubmedArticleSet>
                  <PubmedArticle>
                    <MedlineCitation>
                      <PMID>123</PMID>
                      <Article>
                        <ArticleTitle>CAF HGF MET signaling in lung cancer</ArticleTitle>
                        <Abstract>
                          <AbstractText>CAF-derived HGF activates MET in lung cancer.</AbstractText>
                        </Abstract>
                      </Article>
                    </MedlineCitation>
                    <PubmedData>
                      <ArticleIdList>
                        <ArticleId IdType="pubmed">123</ArticleId>
                        <ArticleId IdType="pmc">PMC123</ArticleId>
                      </ArticleIdList>
                    </PubmedData>
                  </PubmedArticle>
                </PubmedArticleSet>
                """
            )
        return FakeResponse(
            text="""
            <pmc-articleset>
              <article>
                <front>
                  <article-meta>
                    <article-id pub-id-type="pmc">PMC456</article-id>
                    <article-id pub-id-type="pmid">789</article-id>
                    <title-group><article-title>PMC tumor microenvironment context</article-title></title-group>
                    <abstract><p>PMC full-text abstract context for tumor microenvironment.</p></abstract>
                  </article-meta>
                </front>
              </article>
            </pmc-articleset>
            """
        )


class FakePubTatorAsyncClient:
    gets = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params):
        self.gets.append({"url": url, "params": params})
        return FakeResponse(
            {
                "results": [
                    {
                        "pmid": 38720352,
                        "pmcid": "PMC111",
                        "title": "Function of alveolar macrophages in lung cancer microenvironment.",
                        "text_hl": "Function of alveolar macrophages in <m>lung</m> @DISEASE_Neoplasms @@@<m>cancer</m>@@@.",
                        "score": 301.08,
                    }
                ]
            }
        )


class FakeLitSenseAsyncClient:
    gets = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params):
        self.gets.append({"url": url, "params": params})
        if "/sentences/" in url:
            return FakeResponse(
                [
                    {
                        "score": 0.91,
                        "text": "CAF-derived HGF can activate MET signaling in lung cancer.",
                        "pmid": 123,
                        "pmcid": "PMC123",
                        "section": "RESULTS",
                        "annotations": ["0|3|gene|HGF"],
                    }
                ]
            )
        return FakeResponse(
            [
                {
                    "score": 0.88,
                    "text": "A mechanistic paragraph links stromal signals and immune escape.",
                    "pmid": 456,
                    "pmcid": "PMC456",
                    "section": "DISCUSS",
                    "annotations": [],
                }
            ]
        )


class FakeEmptyEutilsAsyncClient(FakeEutilsAsyncClient):
    async def get(self, url, params):
        self.gets.append({"url": url, "params": params})
        if url.endswith("/esearch.fcgi"):
            return FakeResponse({"esearchresult": {"idlist": []}})
        raise AssertionError("empty ESearch must not fetch XML")


class MemoryWebSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_duckduckgo_search_redacts_query_and_flattens_related_topics(self):
        from app.memory.web_search import duckduckgo_search

        FakeAsyncClient.gets = []
        with patch("httpx.AsyncClient", FakeAsyncClient):
            result = await duckduckgo_search("lung cancer TME for researcher@example.org", k=2)

        self.assertTrue(result["redacted"])
        self.assertNotIn("researcher@example.org", result["query"])
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["title"], "Lung cancer")
        self.assertEqual(result["results"][1]["snippet"], "Tumor microenvironment context")
        self.assertEqual(FakeAsyncClient.gets[0]["url"], "https://api.duckduckgo.com/")
        self.assertIn("[email]", FakeAsyncClient.gets[0]["params"]["q"])

    async def test_duckduckgo_search_skips_secret_query_without_http(self):
        from app.memory.web_search import duckduckgo_search

        FakeAsyncClient.gets = []
        with patch("httpx.AsyncClient", FakeAsyncClient):
            result = await duckduckgo_search("token=do-not-send lung cancer", k=2)

        self.assertEqual(result["results"], [])
        self.assertEqual(FakeAsyncClient.gets, [])

    async def test_pubmed_pmc_search_uses_redacted_query_and_tops_up_from_pmc(self):
        from app.memory.web_search import pubmed_pmc_search

        FakeEutilsAsyncClient.gets = []
        with patch("httpx.AsyncClient", FakeEutilsAsyncClient):
            result = await pubmed_pmc_search("lung cancer TME for researcher@example.org", k=2)

        self.assertTrue(result["redacted"])
        self.assertEqual(result["results"][0]["source"], "pubmed")
        self.assertEqual(result["results"][0]["pmcid"], "PMC123")
        self.assertEqual(result["results"][0]["snippet"], "CAF-derived HGF activates MET in lung cancer.")
        self.assertEqual(result["results"][1]["source"], "pmc")
        self.assertEqual(result["results"][1]["pmid"], "789")
        self.assertEqual(result["results"][1]["title"], "PMC tumor microenvironment context")
        self.assertEqual(len(FakeEutilsAsyncClient.gets), 4)
        self.assertIn("[email]", FakeEutilsAsyncClient.gets[0]["params"]["term"])

    async def test_pubmed_pmc_search_skips_secret_query_without_http(self):
        from app.memory.web_search import pubmed_pmc_search

        FakeEutilsAsyncClient.gets = []
        with patch("httpx.AsyncClient", FakeEutilsAsyncClient):
            result = await pubmed_pmc_search("token=do-not-send lung cancer", k=2)

        self.assertEqual(result["results"], [])
        self.assertEqual(FakeEutilsAsyncClient.gets, [])

    async def test_pubmed_pmc_search_returns_empty_when_ncbi_search_has_no_ids(self):
        from app.memory.web_search import pubmed_pmc_search

        FakeEmptyEutilsAsyncClient.gets = []
        with patch("httpx.AsyncClient", FakeEmptyEutilsAsyncClient):
            result = await pubmed_pmc_search("body pH parasite cancer", k=2)

        self.assertEqual(result["results"], [])
        self.assertEqual(len(FakeEmptyEutilsAsyncClient.gets), 2)

    async def test_pubtator3_search_normalizes_highlighted_biomedical_result(self):
        from app.memory.web_search import pubtator3_search

        FakePubTatorAsyncClient.gets = []
        with patch("httpx.AsyncClient", FakePubTatorAsyncClient):
            result = await pubtator3_search("lung cancer TME for researcher@example.org", k=1)

        self.assertTrue(result["redacted"])
        self.assertEqual(result["results"][0]["source"], "pubtator3")
        self.assertEqual(result["results"][0]["pmid"], "38720352")
        self.assertEqual(result["results"][0]["pmcid"], "PMC111")
        self.assertEqual(result["results"][0]["snippet"], "Function of alveolar macrophages in lung cancer.")
        self.assertEqual(
            FakePubTatorAsyncClient.gets[0]["url"],
            "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/",
        )
        self.assertIn("[email]", FakePubTatorAsyncClient.gets[0]["params"]["text"])

    async def test_pubtator3_search_skips_secret_query_without_http(self):
        from app.memory.web_search import pubtator3_search

        FakePubTatorAsyncClient.gets = []
        with patch("httpx.AsyncClient", FakePubTatorAsyncClient):
            result = await pubtator3_search("token=do-not-send lung cancer", k=1)

        self.assertEqual(result["results"], [])
        self.assertEqual(FakePubTatorAsyncClient.gets, [])

    async def test_litsense2_search_redacts_query_and_tops_up_with_passages(self):
        from app.memory.web_search import litsense2_search

        FakeLitSenseAsyncClient.gets = []
        with patch("httpx.AsyncClient", FakeLitSenseAsyncClient):
            result = await litsense2_search("lung cancer TME for researcher@example.org", k=2)

        self.assertTrue(result["redacted"])
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["source"], "litsense2_sentence")
        self.assertEqual(result["results"][0]["pmid"], "123")
        self.assertEqual(result["results"][0]["score"], 0.91)
        self.assertEqual(result["results"][0]["section"], "RESULTS")
        self.assertEqual(result["results"][1]["source"], "litsense2_passage")
        self.assertEqual(result["results"][1]["pmcid"], "PMC456")
        self.assertEqual(len(FakeLitSenseAsyncClient.gets), 2)
        self.assertIn("/sentences/", FakeLitSenseAsyncClient.gets[0]["url"])
        self.assertIn("/passages/", FakeLitSenseAsyncClient.gets[1]["url"])
        self.assertIn("[email]", FakeLitSenseAsyncClient.gets[0]["params"]["query"])

    async def test_litsense2_search_skips_secret_query_without_http(self):
        from app.memory.web_search import litsense2_search

        FakeLitSenseAsyncClient.gets = []
        with patch("httpx.AsyncClient", FakeLitSenseAsyncClient):
            result = await litsense2_search("token=do-not-send lung cancer", k=2)

        self.assertEqual(result["results"], [])
        self.assertEqual(FakeLitSenseAsyncClient.gets, [])

    def test_external_merge_keeps_pubmed_abstract_and_pubtator_semantic_slot(self):
        from app.memory.policy import _merge_external_results

        merged = _merge_external_results(
            [
                {"pmid": "123", "source": "pubmed"},
                {"pmid": "456", "source": "pubmed"},
                {"pmid": "789", "source": "pubmed"},
            ],
            [
                {"pmid": "123", "source": "pubtator3"},
                {"pmid": "999", "source": "pubtator3"},
            ],
            3,
        )

        self.assertEqual([item["pmid"] for item in merged], ["123", "456", "999"])
        self.assertEqual(merged[-1]["source"], "pubtator3")

    def test_external_merge_reserves_litsense_sentence_slot(self):
        from app.memory.policy import _merge_external_results

        merged = _merge_external_results(
            [{"pmid": "123", "source": "pubmed"}, {"pmid": "456", "source": "pubmed"}],
            [{"pmid": "789", "source": "pubtator3"}],
            3,
            [{"pmid": "999", "source": "litsense2_sentence"}],
        )

        self.assertEqual([item["source"] for item in merged], ["pubmed", "litsense2_sentence", "pubtator3"])

    def test_external_query_variants_translate_fungal_tumorigenesis_terms(self):
        from app.memory.policy import _external_query_variants

        variants = _external_query_variants(
            "what fungi are described as playing essential roles in tumorigenesis and how it happens",
            limit=3,
        )
        joined = " ".join(variants).lower()

        self.assertGreaterEqual(len(variants), 2)
        self.assertIn("mycobiome", joined)
        self.assertIn("dysbiosis", joined)
        self.assertIn("tumorigenesis", joined)

    def test_external_query_variants_prioritize_candida_mechanism_bridge(self):
        from app.memory.policy import _external_query_variants

        variants = _external_query_variants("Is candida a fungi promoting tumorgenesis?", limit=4)
        joined = " ".join(variants).lower()

        self.assertIn("candida albicans promotes tumorigenesis", joined)
        self.assertIn("pge2", joined)

    def test_external_ranking_promotes_semantic_pubtator_title(self):
        from app.memory.policy import _merge_external_results

        merged = _merge_external_results(
            [
                {"source": "pmc", "pmid": "42148290", "title": "Microbial extracellular vesicles in the lung", "snippet": "Respiratory inflammation."},
            ],
            [
                {"source": "pubtator3", "pmid": "34298645", "title": "Micro- and Mycobiota Dysbiosis in Pancreatic Ductal Adenocarcinoma Development", "snippet": "Micro- and Mycobiota Dysbiosis in Pancreatic Ductal Adenocarcinoma Development"},
            ],
            2,
            [],
            "what fungi are described as playing essential roles in tumorigenesis and how it happens",
        )

        self.assertEqual(merged[0]["pmid"], "34298645")
        self.assertGreater(merged[0]["external_rank_score"], merged[1]["external_rank_score"])


if __name__ == "__main__":
    unittest.main()
