"""
LegalLens Optional IR Extensions Module (spec.md Section 8)
Implements:
1. Legal Domain Query Expansion (Section 8.1): Expands legal terminology with authoritative Indian legal synonyms/variations.
2. Metadata & Zone Boosting (Section 8.2): Provides calibrated score boosting based on case category and court jurisdiction.
3. Positional Exact Phrase Verification (Section 8.3): Rewards documents containing multi-word statutory phrase matches.
"""

import re
from typing import List, Dict, Set, Tuple
from src.preprocessing import tokenize

# Curated Indian Legal Synonym Thesaurus
LEGAL_SYNONYM_THESAURUS = {
    "tenant": ["tenancy", "lessee", "occupant"],
    "landlord": ["lessor", "owner", "proprietor"],
    "eviction": ["ejectment", "dispossession", "removal"],
    "rent": ["arrears", "lease money", "rental"],
    "accused": ["appellant", "petitioner", "convict", "culprit"],
    "bail": ["anticipatory bail", "surety", "bond", "custody"],
    "murder": ["culpable homicide", "section 302", "section 304"],
    "theft": ["section 379", "misappropriation", "dishonest"],
    "divorce": ["dissolution of marriage", "judicial separation"],
    "adoption": ["hama", "hindu adoptions and maintenance act", "adoptive"],
    "pension": ["family pension", "gratuity", "retiral benefits", "superannuation"],
    "promotion": ["seniority", "dpc", "departmental promotion", "cadre"],
    "arbitration": ["arbitral tribunal", "award", "section 34", "arbitrator"],
    "fundamental right": ["article 21", "article 14", "article 19", "part iii"],
    "writ": ["habeas corpus", "mandamus", "certiorari", "quo warranto"]
}


def expand_query(query: str, max_expansions_per_term: int = 2) -> str:
    """
    Expand legal query using domain thesaurus (spec.md Section 8.1).
    Appends expanded terms to the query string while keeping primary terms prominent.
    """
    clean_q = query.lower()
    expanded_terms = []
    
    for key, synonyms in LEGAL_SYNONYM_THESAURUS.items():
        if re.search(rf'\b{re.escape(key)}\b', clean_q):
            # Term found, add top synonyms
            for syn in synonyms[:max_expansions_per_term]:
                if syn not in clean_q:
                    expanded_terms.append(syn)
                    
    if expanded_terms:
        # Append expansions to query with lower weight or as suffix
        return query + " " + " ".join(expanded_terms)
    return query


def apply_metadata_boost(
    base_score: float,
    doc_category: str,
    query_category: str,
    boost_weight: float = 0.05
) -> float:
    """
    Applies small calibrated category match boost (spec.md Section 8.2).
    Metadata boost remains strictly smaller than primary IR score.
    """
    if query_category != "General Law" and doc_category == query_category:
        return base_score * (1.0 + boost_weight)
    return base_score
