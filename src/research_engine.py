"""Inspectable retrieval, literal constraints, and counterfactual ranking experiments.

The default path runs entirely offline on the shipped IndicLegalQA passages.
No model scores are labelled as probabilities or legal authority.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import heapq
import json
import math
import re

import numpy as np

from src.bm25_retriever import BM25Retriever
from src.preprocessing import tokenize
from src.tfidf_retriever import TFIDFRetriever
from src.utils import min_max_normalize

ROOT = Path(__file__).resolve().parents[1]
MODES = ["Evidence-aware BM25", "BM25", "TF-IDF", "Rank fusion (BM25 + TF-IDF)"]


def surface_tokens(text):
    """Literal positional tokens: keep stopwords, numbers, and inflections."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


@dataclass(frozen=True)
class QueryPlan:
    original: str
    scoring_query: str
    phrases: tuple
    excluded: tuple
    anchors: tuple


def parse_query(query):
    """Quoted phrases and -word are constraints; section/article numbers are anchors."""
    if query.count('"') % 2:
        raise ValueError('Close the quotation mark to search an exact phrase.')
    phrases = tuple(re.findall(r'"([^"\n]+)"', query))
    # A minus inside a quoted phrase is a literal separator, not an exclusion.
    unquoted = re.sub(r'"[^"\n]*"', '', query)
    excluded = tuple(re.findall(r'(?<!\S)-([a-zA-Z][a-zA-Z0-9-]*)', unquoted))
    scoring = re.sub(r'(?<!\S)-[a-zA-Z][a-zA-Z0-9-]*', '', re.sub(r'"[^"\n]*"', ' ', query))
    scoring = ' '.join((scoring + ' ' + ' '.join(phrases)).split())
    anchors = tuple(dict.fromkeys(
        f"{kind.lower()} {number.lower()}"
        for kind, number in re.findall(r'\b(section|article)\s+(\d+[a-z]?)\b', scoring, re.I)
    ))
    return QueryPlan(query, scoring, phrases, excluded, anchors)


def contiguous_positions(tokens, phrase):
    words = surface_tokens(phrase)
    if not words:
        return []
    width = len(words)
    return [i for i in range(len(tokens) - width + 1) if tokens[i:i + width] == words]


class ResearchEngine:
    def __init__(self, documents):
        self.documents = {d['doc_id']: d for d in documents}
        self.bm25 = BM25Retriever()
        self.bm25.fit(documents)
        self.tfidf = TFIDFRetriever()
        # Align score arrays explicitly, independent of input JSON order.
        self.doc_ids = self.bm25.doc_ids
        self.tfidf.fit([self.documents[d] for d in self.doc_ids])
        self.surfaces = {d: surface_tokens(self.documents[d]['text']) for d in self.doc_ids}
        # A quoted phrase must occur inside one real source field/passage,
        # never across two answers concatenated during corpus construction.
        self.literal_segments = {
            d: [surface_tokens(value) for value in (
                [self.documents[d].get(key, '') for key in ('case_name', 'date', 'court')]
                + (self.documents[d].get('passages') or [self.documents[d]['text']]))]
            for d in self.doc_ids
        }
        self.literal_postings = defaultdict(set)
        for doc_id, tokens in self.surfaces.items():
            for term in set(tokens):
                self.literal_postings[term].add(doc_id)
        self.semantic = None
        self.modes = list(MODES)

    @classmethod
    def from_project(cls):
        with (ROOT / 'data/processed/documents.json').open(encoding='utf-8') as f:
            return cls(json.load(f))

    def enable_semantic(self):
        """Opt-in MiniLM; validate metadata and load a locally cached model only."""
        from src.semantic_retriever import SemanticRetriever
        from sentence_transformers import SentenceTransformer
        semantic = SemanticRetriever.load(
            str(ROOT / 'data/processed/doc_embeddings.npy'),
            str(ROOT / 'data/processed/semantic_metadata.pkl'),
        )
        if set(semantic.doc_ids) != set(self.doc_ids):
            raise ValueError('Dense index does not match the current document corpus.')
        if len(semantic.doc_embeddings) != len(semantic.doc_ids):
            raise ValueError('Dense embeddings and metadata have different lengths.')
        mapping = {d: i for i, d in enumerate(semantic.doc_ids)}
        semantic.doc_embeddings = semantic.doc_embeddings[[mapping[d] for d in self.doc_ids]]
        semantic.doc_ids = list(self.doc_ids)
        semantic.model = SentenceTransformer(semantic.model_name, local_files_only=True)
        self.semantic = semantic
        self.modes = list(MODES) + ['Hybrid (BM25 + MiniLM)', 'MiniLM']

    def candidates(self, plan, category=None, strict_anchors=False):
        allowed = set(self.doc_ids)
        if category:
            allowed &= {d for d in self.doc_ids if self.documents[d].get('category') == category}
        for excluded in plan.excluded:
            allowed -= self.literal_postings.get(excluded.lower(), set())
        for phrase in plan.phrases + (plan.anchors if strict_anchors else ()):
            words = surface_tokens(phrase)
            if not words:
                return set()
            # Intersect rare terms first, then verify literal positional adjacency.
            for word in sorted(set(words), key=lambda w: len(self.literal_postings.get(w, ()))):
                allowed &= self.literal_postings.get(word, set())
            allowed = {d for d in allowed if any(contiguous_positions(segment, phrase)
                                                 for segment in self.literal_segments[d])}
        return allowed

    def score(self, query, mode='Evidence-aware BM25', alpha=0.75):
        if mode not in self.modes:
            raise ValueError(f'Retrieval mode unavailable: {mode}')
        plan = parse_query(query)
        bm25 = self.bm25.get_all_scores(plan.scoring_query)
        lexical = min_max_normalize(bm25)
        evidence = lexical.copy()
        # Bounded structural bonus; no invented citation links or authority scores.
        for i, doc_id in enumerate(self.doc_ids):
            if bm25[i] <= 0:
                continue
            if plan.anchors:
                coverage = sum(bool(contiguous_positions(self.surfaces[doc_id], a)) for a in plan.anchors) / len(plan.anchors)
                evidence[i] += 0.15 * coverage
        secondary = np.zeros(len(self.doc_ids))
        if mode == 'Evidence-aware BM25':
            scores = evidence
        elif mode == 'BM25':
            scores = bm25
        elif mode == 'TF-IDF':
            secondary = self.tfidf.get_all_scores(plan.scoring_query)
            scores = secondary
        elif mode == 'Rank fusion (BM25 + TF-IDF)':
            secondary = self.tfidf.get_all_scores(plan.scoring_query)
            scores = np.zeros(len(self.doc_ids))
            for component in (bm25, secondary):
                active = sorted((i for i, s in enumerate(component) if s > 0), key=lambda i: (-float(component[i]), self.doc_ids[i]))
                for rank, idx in enumerate(active, 1):
                    scores[idx] += 1.0 / (60 + rank)
        else:
            secondary = self.semantic.get_all_scores(plan.scoring_query)
            weight = float(np.clip(alpha, 0, 1))
            scores = secondary if mode == 'MiniLM' else weight * lexical + (1 - weight) * min_max_normalize(secondary)
        return plan, np.asarray(scores), bm25, secondary, lexical

    def ranked_ids(self, query, mode='Evidence-aware BM25', top_k=10, category=None, strict_anchors=False, alpha=0.75):
        if top_k <= 0 or not tokenize(parse_query(query).scoring_query):
            return []
        plan, scores, *_ = self.score(query, mode, alpha)
        allowed = self.candidates(plan, category, strict_anchors)
        valid = [i for i, d in enumerate(self.doc_ids) if d in allowed and scores[i] > 0]
        ranked = heapq.nsmallest(top_k, valid, key=lambda i: (-float(scores[i]), self.doc_ids[i]))
        return [self.doc_ids[i] for i in ranked]

    def term_contributions(self, query, doc_id):
        """Exact per-term BM25 arithmetic, including the existing 1.3 title weight."""
        idx = self.bm25.index
        rows = []
        for term, qtf in Counter(tokenize(query)).items():
            posting = next((p for p in idx.postings.get(term, ()) if p[0] == doc_id), None)
            if not posting:
                continue
            _, tf, positions = posting
            idf = idx.get_idf(term)
            length = 1 - self.bm25.b + self.bm25.b * idx.doc_lens[doc_id] / (idx.avg_doc_len or 1)
            title_weight = 1.3 if doc_id in idx.case_name_postings.get(term, ()) else 1.0
            contribution = idf * tf * (self.bm25.k1 + 1) / (tf + self.bm25.k1 * length) * qtf * title_weight
            rows.append(dict(term=term, tf=tf, df=idx.doc_freq[term], idf=idf, title_weight=title_weight, contribution=contribution, positions=positions[:12]))
        return sorted(rows, key=lambda r: (-r['contribution'], r['term']))

    def evidence_passage(self, query, doc_id):
        """Return an actual source excerpt and its passage number; never generate text."""
        doc = self.documents[doc_id]
        passages = doc.get('passages') or [doc['text']]
        terms = set(tokenize(query))
        best = None
        for number, passage in enumerate(passages, 1):
            # Rank overlapping windows so a match after word 120 remains visible.
            spans = list(re.finditer(r'\S+', passage))
            for start in range(0, max(1, len(spans)), 60):
                end = min(start + 120, len(spans))
                if end <= start:
                    continue
                excerpt = passage[spans[start].start():spans[end - 1].end()]
                found = terms & set(tokenize(excerpt))
                value = sum(self.bm25.index.get_idf(t) for t in found) / math.sqrt(max(1, len(tokenize(excerpt))))
                item = (value, number, excerpt, sorted(found), start, end)
                if best is None or value > best[0]:
                    best = item
                if end == len(spans):
                    break
        if best is None:
            return dict(text='', passage_number=1, matched_terms=[], word_start=0, word_end=0)
        _, number, excerpt, matched, start, end = best
        return dict(text=excerpt, passage_number=number, matched_terms=matched, word_start=start, word_end=end)

    def search(self, query, mode='Evidence-aware BM25', top_k=10, category=None, strict_anchors=False, alpha=0.75):
        plan = parse_query(query)
        if top_k <= 0 or not tokenize(plan.scoring_query):
            return dict(query=query, plan=plan.__dict__, results=[], diagnostics={'coverage': 0, 'overlap': 0, 'margin': 0})
        plan, scores, bm25, secondary, normalized = self.score(query, mode, alpha)
        allowed = self.candidates(plan, category, strict_anchors)
        valid = [i for i, d in enumerate(self.doc_ids) if d in allowed and scores[i] > 0]
        indices = heapq.nsmallest(top_k, valid, key=lambda i: (-float(scores[i]), self.doc_ids[i]))
        rows = []
        query_terms = set(tokenize(plan.scoring_query))
        for idx in indices:
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            matched = sorted(query_terms & set(tokenize(doc['text'])))
            rows.append(dict(
                doc_id=doc_id, case_name=doc.get('case_name', 'Unknown case'), date=doc.get('date', 'Unknown'),
                court=doc.get('court', 'Unknown'), category=doc.get('category', 'General Law'),
                score=float(scores[idx]), bm25=float(bm25[idx]), lexical_normalized=float(normalized[idx]),
                secondary=float(secondary[idx]), matched_terms=matched,
                anchor_matches=[a for a in plan.anchors if contiguous_positions(self.surfaces[doc_id], a)],
                coverage=len(matched) / len(query_terms) if query_terms else 0,
                evidence=self.evidence_passage(plan.scoring_query, doc_id),
            ))
        lexical_ids = self.ranked_ids(query, 'BM25', top_k, category, strict_anchors)
        tfidf_ids = self.ranked_ids(query, 'TF-IDF', top_k, category, strict_anchors)
        union = set(lexical_ids) | set(tfidf_ids)
        return dict(query=query, mode=mode, alpha=alpha, strict_anchors=strict_anchors, category=category,
                    plan=plan.__dict__, results=rows,
                    diagnostics=dict(coverage=rows[0]['coverage'] if rows else 0,
                                     overlap=len(set(lexical_ids) & set(tfidf_ids)) / len(union) if union else 0,
                                     margin=rows[0]['score'] - rows[1]['score'] if len(rows) > 1 else 0,
                                     eligible=len(allowed)))

    def counterfactuals(self, query, mode='Evidence-aware BM25', top_k=5, category=None, strict_anchors=False, alpha=0.75):
        """Leave one unquoted, non-excluded term out; preserve literal constraints."""
        original = self.ranked_ids(query, mode, top_k, category, strict_anchors, alpha)
        if not original:
            return []
        protected = [m.span() for m in re.finditer(r'"[^"\n]*"|(?<!\S)-[a-zA-Z][a-zA-Z0-9-]*', query)]
        terms = {}
        for m in re.finditer(r'\b[a-zA-Z][a-zA-Z0-9-]*\b', query):
            if any(a <= m.start() < b for a, b in protected):
                continue
            token = tokenize(m.group())
            if token:
                terms.setdefault(token[0], []).append(m)
        rows = []
        # Test the eight highest-IDF term groups to bound interactive latency.
        groups = sorted(terms.items(), key=lambda item: (-self.bm25.index.get_idf(item[0]), item[0]))[:8]
        for term, matches in groups:
            changed = query
            for m in reversed(matches):
                changed = changed[:m.start()] + changed[m.end():]
            changed = ' '.join(changed.split())
            after = self.ranked_ids(changed, mode, top_k, category, strict_anchors, alpha)
            union = set(original) | set(after)
            rows.append(dict(removed=matches[0].group(), query=changed, top_case=self.documents[after[0]]['case_name'] if after else 'No matching case',
                             original_winner_rank=after.index(original[0]) + 1 if original[0] in after else None,
                             overlap=len(set(original) & set(after)) / len(union) if union else 0,
                             winner_changed=not after or after[0] != original[0]))
        return rows


def research_brief(snapshot):
    lines = ['# LegalLens evidence brief', '', f"Query: {snapshot['query']}", f"Method: {snapshot.get('mode', '')}",
             '', 'Source: IndicLegalQA case-linked answer passages, not full judgments.',
             'Scores express retrieval relevance, not legal correctness. Categories are heuristic.', '']
    for rank, row in enumerate(snapshot['results'], 1):
        lines.extend([f"## {rank}. {row['case_name']}", f"Document: {row['doc_id']} | Date: {row['date']}",
                      f"Score: {row['score']:.6f} | BM25: {row['bm25']:.6f}",
                      f"Evidence: passage {row['evidence']['passage_number']}, words {row['evidence']['word_start'] + 1}-{row['evidence']['word_end']}",
                      '', '> ' + row['evidence']['text'], '', 'Matched tokens: ' + ', '.join(row['matched_terms']), ''])
    return '\n'.join(lines)
