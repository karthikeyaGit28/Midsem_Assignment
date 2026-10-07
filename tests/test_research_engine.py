import unittest
import numpy as np
from src.research_engine import ResearchEngine, parse_query, research_brief
from src.evaluate_research import query_metrics
from src.passage_retrieval import highlight_matched_terms


class ResearchEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = [
            dict(doc_id='b', case_name='Tenant dispute', date='2020', category='Property',
                 text='The tenant receives notice under the rent control act. Section 30 applies.',
                 passages=['The tenant receives notice under the rent control act. Section 30 applies.']),
            dict(doc_id='a', case_name='Criminal appeal', date='2021', category='Criminal',
                 text='Murder conviction under section 302. Bail was refused.',
                 passages=['Murder conviction under section 302. Bail was refused.']),
            dict(doc_id='c', case_name='Tax dispute', date='2022', category='Tax',
                 text='Tax on the rent levy is disputed. A tenant claims a deduction.',
                 passages=['Tax on the rent levy is disputed. A tenant claims a deduction.']),
            dict(doc_id='d', case_name='Promotion dispute', date='2023', category='Service',
                 text='A request for promotion after denial based on seniority.',
                 passages=['A request for promotion after denial based on seniority.']),
        ]
        cls.engine = ResearchEngine(cls.docs)

    def test_score_alignment_is_independent_of_json_order(self):
        self.assertEqual(self.engine.doc_ids, self.engine.tfidf.doc_ids)
        self.assertEqual(self.engine.ranked_ids('murder', 'TF-IDF')[0], 'a')

    def test_filter_happens_before_top_k(self):
        rows = self.engine.search('tenant rent', top_k=1, category='Tax')['results']
        self.assertEqual(rows[0]['doc_id'], 'c')

    def test_literal_phrase_keeps_stopwords(self):
        self.assertEqual(self.engine.ranked_ids('"rent control act"'), ['b'])
        self.assertEqual(self.engine.ranked_ids('"rent act"'), [])
        self.assertEqual(self.engine.ranked_ids('"under the rent"'), ['b'])

    def test_literal_phrase_is_not_stemmed(self):
        self.assertEqual(self.engine.ranked_ids('"tenants"'), [])

    def test_exclusion_removes_source(self):
        ids = self.engine.ranked_ids('tenant rent -tax')
        self.assertIn('b', ids)
        self.assertNotIn('c', ids)

    def test_section_number_does_not_match_prefix(self):
        self.assertEqual(self.engine.ranked_ids('section 30', strict_anchors=True), ['b'])
        self.assertEqual(self.engine.ranked_ids('section 302', strict_anchors=True), ['a'])
        self.assertEqual(self.engine.ranked_ids('section 30 section 302', strict_anchors=True), [])

    def test_term_audit_reconstructs_bm25(self):
        query = 'tenant tenant rent notice'
        total = sum(r['contribution'] for r in self.engine.term_contributions(query, 'b'))
        raw = self.engine.bm25.get_all_scores(query)[self.engine.doc_ids.index('b')]
        self.assertAlmostEqual(total, float(raw), places=5)

    def test_exact_evidence_is_a_source_substring(self):
        evidence = self.engine.evidence_passage('rent notice', 'b')
        self.assertIn(evidence['text'], self.docs[0]['passages'][0])
        self.assertEqual(evidence['passage_number'], 1)

    def test_passage_after_word_120_is_retrievable(self):
        passage = ' '.join(['filler'] * 140 + ['rareterm'] + ['filler'] * 20)
        engine = ResearchEngine([dict(doc_id='x', case_name='Long source', date='2020', text=passage, passages=[passage])])
        evidence = engine.evidence_passage('rareterm', 'x')
        self.assertIn('rareterm', evidence['text'])
        self.assertIn(evidence['text'], passage)
        self.assertGreater(evidence['word_start'], 0)

    def test_no_arbitrary_zero_score_results(self):
        for method in self.engine.modes:
            self.assertEqual(self.engine.ranked_ids('zzzzunfindable', method), [])
            self.assertEqual(self.engine.ranked_ids('', method), [])
            self.assertEqual(self.engine.ranked_ids('tenant', method, 0), [])

    def test_constraints_survive_counterfactuals(self):
        rows = self.engine.counterfactuals('tenant "rent control act" -tax notice')
        self.assertTrue(rows)
        for row in rows:
            self.assertIn('"rent control act"', row['query'])
            self.assertIn('-tax', row['query'])
            self.assertGreaterEqual(row['overlap'], 0)
            self.assertLessEqual(row['overlap'], 1)

    def test_parse_errors_and_minus_in_quote(self):
        with self.assertRaises(ValueError):
            parse_query('"unclosed phrase')
        self.assertEqual(parse_query('"tenant -tax"').excluded, ())

    def test_rank_fusion_is_reproducible(self):
        a = self.engine.ranked_ids('tenant rent', 'Rank fusion (BM25 + TF-IDF)')
        self.assertEqual(a, self.engine.ranked_ids('tenant rent', 'Rank fusion (BM25 + TF-IDF)'))

    def test_rank_fusion_ties_use_document_id_order(self):
        engine = ResearchEngine([dict(doc_id=d, case_name='Equal source', date='2020',
                                      text='tenant notice', passages=['tenant notice']) for d in ['z', 'a']])
        self.assertEqual(engine.ranked_ids('tenant notice', 'Rank fusion (BM25 + TF-IDF)'), ['a', 'z'])

    def test_evidence_brief_has_provenance(self):
        brief = research_brief(self.engine.search('rent notice'))
        self.assertIn('not full judgments', brief)
        self.assertIn('passage 1', brief)

    def test_metric_known_rank(self):
        metrics = query_metrics(['a', 'b', 'c'], 'b')
        self.assertEqual(metrics['P@5'], .2)
        self.assertEqual(metrics['R@10'], 1)
        self.assertEqual(metrics['MRR@10'], .5)
        self.assertAlmostEqual(metrics['nDCG@10'], 1 / np.log2(3))
        self.assertEqual(query_metrics(['a'], 'b')['MRR@10'], 0)

    def test_highlight_escapes_source_without_rewriting_its_own_tags(self):
        result = highlight_matched_terms('<script>style mark</script>', 'style mark')
        self.assertNotIn('<script>', result)
        self.assertEqual(result.count('<mark '), 2)
        self.assertIn('&lt;script&gt;', result)

    def test_metrics_ignore_relevance_beyond_depth_ten(self):
        metrics = query_metrics([str(i) for i in range(11)], '10')
        self.assertEqual(metrics['MRR@10'], 0)
        self.assertEqual(metrics['R@10'], 0)
        self.assertEqual(metrics['nDCG@10'], 0)

    def test_phrase_cannot_cross_two_original_answer_passages(self):
        engine = ResearchEngine([dict(doc_id='x', case_name='Boundary fixture', date='2020',
                                      text='alpha beta gamma delta', passages=['alpha beta', 'gamma delta'])])
        self.assertEqual(engine.ranked_ids('"beta gamma"'), [])
        self.assertEqual(engine.ranked_ids('"gamma delta"'), ['x'])


if __name__ == '__main__':
    unittest.main()
