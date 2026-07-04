"""
text_analytics.py
=================
Analise profunda do par (query, response) de cada turno de dialogo.

Cinco dimensoes analisadas:
  [1] Superficie      -- contagens brutas: chars, palavras, sentencas
  [2] Riqueza lexical -- TTR, MATTR, hapax, Brunet W
  [3] Legibilidade    -- Flesch, Gunning Fog, Coleman-Liau
  [4] Estrutura       -- POS tagging, retorica, hedges, sentimento, NER
  [5] Alinhamento     -- similaridade semantica, coverage, novelty,
                         information gain, follow-up detection

Dependencias (adicionar ao requirements.txt):
    spacy>=3.7.0
    pt_core_news_sm   (python -m spacy download pt_core_news_sm)
    textstat>=0.7.3
    sentence-transformers>=2.2.2
    nltk>=3.8.1
    textblob>=0.17.1

Fail-safe por camada: se uma dependencia faltar, os campos daquela
camada retornam 0.0/False/"" sem quebrar o restante do pipeline.

Uso:
    metrics = analyze(query, response, prev_query=None, session_position=1)
    # metrics.semantic_similarity, metrics.r_flesch, etc.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Versao do modulo -- registrada em TextMetrics para rastreabilidade
# ---------------------------------------------------------------------------
_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Modelos utilizados -- definidos aqui para facil troca
# ---------------------------------------------------------------------------
_SPACY_MODEL     = "pt_core_news_sm"
_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------------------
# Conectivos e marcadores PT-BR -- listas extensiveis
# ---------------------------------------------------------------------------

_LOGICAL_CONN = [
    "portanto", "logo", "assim", "dessa forma", "desse modo",
    "por isso", "consequentemente", "por conseguinte", "entao",
]
_ADVERSATIVE_CONN = [
    "mas", "porem", "entretanto", "todavia", "contudo",
    "no entanto", "ainda assim", "apesar disso", "embora",
]
_ADDITIVE_CONN = [
    "alem disso", "tambem", "ademais", "ainda", "igualmente",
    "do mesmo modo", "outrossim", "bem como",
]
_CAUSAL_CONN = [
    "porque", "pois", "ja que", "dado que", "uma vez que",
    "visto que", "tendo em vista", "em virtude de",
]
_ENUM_MARKERS = [
    "primeiro", "segundo", "terceiro", "por fim", "finalmente",
    "em primeiro lugar", "em segundo lugar", "por ultimo",
    "inicialmente", "posteriormente",
]
_HEDGES = [
    "talvez", "possivelmente", "provavelmente", "pode ser",
    "acredita-se", "parece", "aparentemente", "possivelmente",
    "em geral", "normalmente", "costuma", "tende a",
    "pode", "poderia", "seria", "estaria",
]
_CERTAINTY = [
    "certamente", "definitivamente", "com certeza", "sem duvida",
    "indubitavelmente", "claramente", "obviamente", "evidentemente",
    "sempre", "nunca", "jamais",
]
_ASSERTIVE_VERBS_PATTERNS = [
    r"\bfaca\b", r"\buse\b", r"\bevite\b", r"\bdefina\b",
    r"\bconfigure\b", r"\binstale\b", r"\bexecute\b", r"\brode\b",
    r"\bcrie\b", r"\badote\b", r"\bimplemente\b", r"\badicione\b",
]

# Tipos de query -- classificacao heuristica
_QUERY_PATTERNS = {
    "factual":       [r"\bqual\b", r"\bquem\b", r"\bonde\b", r"\bquando\b",
                      r"\bo que e\b", r"\bquanto\b", r"\bcomo se chama\b"],
    "procedimental": [r"\bcomo fazer\b", r"\bcomo funciona\b", r"\bcomo usar\b",
                      r"\bcomo criar\b", r"\bcomo configurar\b", r"\bpasso a passo\b",
                      r"\bcomo implementar\b", r"\bcomo instalar\b"],
    "comparativa":   [r"\bdiferenca entre\b", r"\bcomparar\b", r"\bmelhor entre\b",
                      r"\bvantagens e desvantagens\b", r"\bversus\b", r"\bvs\b"],
    "opinativa":     [r"\bo que voce acha\b", r"\bsua opiniao\b", r"\brecomenda\b",
                      r"\bvale a pena\b", r"\bvoce prefere\b", r"\bmelhor opcao\b"],
    "criativa":      [r"\bescreva\b", r"\bcrie\b", r"\bgere\b", r"\bproduza\b",
                      r"\belabore\b", r"\bredija\b", r"\bformule\b"],
}


# ---------------------------------------------------------------------------
# Singleton lazy -- carrega modelos pesados uma unica vez por processo
# ---------------------------------------------------------------------------
_nlp         = None   # spaCy
_embedder    = None   # SentenceTransformer
_textblob_ok = False
_spacy_ok    = False
_embed_ok    = False


def _get_nlp():
    global _nlp, _spacy_ok
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load(_SPACY_MODEL)
            _spacy_ok = True
        except Exception:
            _spacy_ok = False
    return _nlp


def _get_embedder():
    global _embedder, _embed_ok
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(_EMBEDDING_MODEL)
            _embed_ok = True
        except Exception:
            _embed_ok = False
    return _embedder


def _check_textblob():
    global _textblob_ok
    try:
        from textblob import TextBlob
        _textblob_ok = True
    except Exception:
        _textblob_ok = False
    return _textblob_ok


# ---------------------------------------------------------------------------
# Dataclass principal
# ---------------------------------------------------------------------------

@dataclass
class TextMetrics:
    """
    Snapshot completo de metricas textuais de um turno.
    Armazenado no TurnRecord e exibido na tabela de observabilidade.
    Campos zerados (0 / 0.0 / False / "") quando a dependencia nao esta disponivel.
    """

    # ── Superficie — Query ───────────────────────────────────────────────────
    q_chars:               int   = 0    # total de caracteres
    q_chars_no_space:      int   = 0    # sem espacos
    q_words:               int   = 0    # total de palavras
    q_words_unique:        int   = 0    # palavras unicas (vocabulario)
    q_words_long:          int   = 0    # palavras com > 6 chars (complexidade)
    q_sentences:           int   = 0    # total de sentencas
    q_sent_len_mean:       float = 0.0  # comprimento medio de sentenca (palavras)
    q_sent_len_std:        float = 0.0  # desvio padrao do comprimento
    q_sent_questions:      int   = 0    # sentencas interrogativas
    q_punctuation_commas:  int   = 0    # virgulas (proxy de complexidade sintatica)
    q_punctuation_excl:    int   = 0    # exclamacoes (intensidade emocional)
    q_subquestions:        int   = 0    # sub-perguntas detectadas na query

    # ── Superficie — Response ────────────────────────────────────────────────
    r_chars:               int   = 0
    r_chars_no_space:      int   = 0
    r_words:               int   = 0
    r_words_unique:        int   = 0
    r_words_long:          int   = 0
    r_sentences:           int   = 0
    r_sent_len_mean:       float = 0.0
    r_sent_len_std:        float = 0.0
    r_paragraphs:          int   = 0    # paragrafos (separados por \n\n)
    r_list_markers:        int   = 0    # marcadores de lista (-, *, 1.)
    r_punctuation_commas:  int   = 0
    r_punctuation_excl:    int   = 0

    # ── Riqueza lexical ──────────────────────────────────────────────────────
    q_ttr:                 float = 0.0  # Type-Token Ratio query
    r_ttr:                 float = 0.0  # Type-Token Ratio response
    r_mattr:               float = 0.0  # Moving Average TTR (janela 50 tokens)
    r_hapax_ratio:         float = 0.0  # hapax legomena / total palavras
    r_brunet_w:            float = 0.0  # Brunet W -- riqueza robusta

    # ── Legibilidade ─────────────────────────────────────────────────────────
    q_flesch:              float = 0.0  # Flesch Reading Ease (0-100, maior=mais facil)
    r_flesch:              float = 0.0
    r_gunning_fog:         float = 0.0  # Gunning Fog (anos de escolaridade)
    r_coleman_liau:        float = 0.0  # Coleman-Liau Index
    flesch_gap:            float = 0.0  # r_flesch - q_flesch (complexidade relativa)

    # ── POS Tagging — Response ───────────────────────────────────────────────
    r_noun_ratio:          float = 0.0  # substantivos / total tokens
    r_verb_ratio:          float = 0.0  # verbos / total tokens
    r_adj_ratio:           float = 0.0  # adjetivos / total tokens
    r_adv_ratio:           float = 0.0  # adverbios / total tokens
    r_pronoun_ratio:       float = 0.0  # pronomes / total (alto = texto vago)
    r_content_word_ratio:  float = 0.0  # palavras de conteudo / total

    # ── Estrutura retorica — Response ────────────────────────────────────────
    r_logical_conn:        int   = 0    # portanto, logo, assim...
    r_adversative_conn:    int   = 0    # mas, porem, entretanto...
    r_additive_conn:       int   = 0    # alem disso, tambem...
    r_causal_conn:         int   = 0    # porque, pois, ja que...
    r_enum_markers:        int   = 0    # primeiro, segundo, por fim...
    r_hedge_count:         int   = 0    # talvez, possivelmente...
    r_certainty_count:     int   = 0    # certamente, definitivamente...
    r_hedge_ratio:         float = 0.0  # hedges / (hedges + certainty)
    r_assertive_verbs:     int   = 0    # verbos imperativos: faca, use, evite...

    # ── Sentimento e tom ─────────────────────────────────────────────────────
    q_sentiment_polarity:      float = 0.0  # -1.0 (neg) a +1.0 (pos)
    q_sentiment_subjectivity:  float = 0.0  # 0.0 (obj) a 1.0 (subj)
    r_sentiment_polarity:      float = 0.0
    r_sentiment_subjectivity:  float = 0.0
    r_formality_score:         float = 0.0  # 0.0 (informal) a 1.0 (formal)

    # ── NER — Response ───────────────────────────────────────────────────────
    r_ner_persons:         int   = 0
    r_ner_orgs:            int   = 0
    r_ner_locations:       int   = 0
    r_ner_dates:           int   = 0
    r_ner_values:          int   = 0    # valores monetarios, percentuais
    r_ner_total:           int   = 0

    # ── Query intent e tipo ──────────────────────────────────────────────────
    q_type:                str   = ""   # factual|procedimental|comparativa|etc
    q_type_confidence:     float = 0.0  # confianca da classificacao (0-1)
    q_has_context:         bool  = False # query forneceu contexto suficiente?
    q_specificity_score:   float = 0.0  # 0.0 (vaga) a 1.0 (especifica)

    # ── Alinhamento query/response ───────────────────────────────────────────
    semantic_similarity:   float = 0.0  # cosine similarity dos embeddings
    lexical_overlap:       float = 0.0  # palavras query em response / palavras query
    keyword_coverage:      float = 0.0  # keywords query cobertas na response
    topic_coverage_ratio:  float = 0.0  # sub-perguntas enderecadas / total
    response_novelty:      float = 0.0  # 1 - lexical_overlap (conteudo novo)
    effort_ratio:          float = 0.0  # r_words / q_words
    complexity_gap:        float = 0.0  # r_flesch - q_flesch
    information_gain:      float = 0.0  # entropia response - entropia query

    # ── Follow-up e contexto de sessao ───────────────────────────────────────
    is_followup:           bool  = False  # query similar a anterior?
    followup_similarity:   float = 0.0   # similaridade com query anterior
    session_position:      int   = 1     # posicao do turno na sessao (1, 2, 3...)
    topics_introduced:     int   = 0     # topicos novos trazidos pelo modelo

    # ── Metadados de processamento ───────────────────────────────────────────
    analytics_version:     str   = _VERSION
    processing_ms:         float = 0.0   # tempo de processamento em ms
    spacy_available:       bool  = False
    embedding_available:   bool  = False


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _count_syllables_pt(word: str) -> int:
    """
    Estimativa de silabas em portugues.
    Heuristica: conta grupos de vogais consecutivas.
    Nao e perfeita mas e boa o suficiente para indices de legibilidade.
    """
    word  = word.lower().strip(".,!?;:")
    vowels = "aeiouaeiouaeiou"  # inclui acentuadas basicas
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(count, 1)


def _tokenize_sentences(text: str) -> list[str]:
    """Tokeniza sentencas por pontuacao terminal."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if s.strip()]


def _tokenize_words(text: str) -> list[str]:
    """Tokeniza palavras removendo pontuacao."""
    return re.findall(r'\b[a-zA-ZÀ-ÿ]{2,}\b', text.lower())


def _entropy(words: list[str]) -> float:
    """Entropia de Shannon sobre distribuicao de palavras."""
    if not words:
        return 0.0
    from collections import Counter
    counts = Counter(words)
    total  = len(words)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _count_pattern_list(text: str, patterns: list[str]) -> int:
    """Conta ocorrencias de uma lista de padroes/strings no texto."""
    text_lower = text.lower()
    count = 0
    for p in patterns:
        if re.search(r'\b' + re.escape(p) + r'\b', text_lower):
            count += 1
    return count


def _mattr(words: list[str], window: int = 50) -> float:
    """
    Moving Average Type-Token Ratio.
    Divide o texto em janelas de `window` tokens e calcula TTR medio.
    Normaliza o TTR para textos de comprimentos diferentes.
    """
    if len(words) < window:
        unique = len(set(words))
        return round(unique / len(words), 4) if words else 0.0
    ttrs = []
    for i in range(len(words) - window + 1):
        window_words = words[i:i + window]
        ttrs.append(len(set(window_words)) / window)
    return round(sum(ttrs) / len(ttrs), 4)


def _brunet_w(n_words: int, n_unique: int) -> float:
    """
    Brunet's W: W = N ^ (V ^ -0.165)
    N = total words, V = unique words.
    Menor valor = maior riqueza lexical.
    Range tipico: 10-20 para textos ricos, > 30 para textos repetitivos.
    """
    if n_words <= 0 or n_unique <= 0:
        return 0.0
    try:
        return round(n_words ** (n_unique ** -0.165), 2)
    except Exception:
        return 0.0


def _flesch_pt(text: str) -> float:
    """
    Flesch Reading Ease adaptado para portugues.
    Formula: 206.835 - 1.015*(palavras/sentencas) - 84.6*(silabas/palavras)
    Resultado: 0-100 (maior = mais facil de ler)
    """
    words = _tokenize_words(text)
    sents = _tokenize_sentences(text)
    if not words or not sents:
        return 0.0
    avg_sent_len  = len(words) / len(sents)
    avg_syllables = sum(_count_syllables_pt(w) for w in words) / len(words)
    score = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_syllables)
    return round(max(0.0, min(100.0, score)), 2)


def _gunning_fog(text: str) -> float:
    """
    Gunning Fog Index.
    Formula: 0.4 * ((palavras/sentencas) + 100 * (palavras_complexas/palavras))
    Palavras complexas: 3+ silabas. Resultado = anos de escolaridade necessarios.
    """
    words = _tokenize_words(text)
    sents = _tokenize_sentences(text)
    if not words or not sents:
        return 0.0
    complex_words = [w for w in words if _count_syllables_pt(w) >= 3]
    avg_sent = len(words) / len(sents)
    pct_complex = len(complex_words) / len(words)
    return round(0.4 * (avg_sent + 100 * pct_complex), 2)


def _coleman_liau(text: str) -> float:
    """
    Coleman-Liau Index.
    Formula: 0.0588*L - 0.296*S - 15.8
    L = media de letras por 100 palavras, S = media de sentencas por 100 palavras.
    """
    words = _tokenize_words(text)
    sents = _tokenize_sentences(text)
    if not words or not sents:
        return 0.0
    letters = sum(len(w) for w in words)
    L = (letters / len(words)) * 100
    S = (len(sents) / len(words)) * 100
    return round(0.0588 * L - 0.296 * S - 15.8, 2)


def _classify_query_type(query: str) -> tuple[str, float]:
    """
    Classifica o tipo de query por correspondencia de padroes heuristicos.
    Retorna (tipo, confianca).
    Confianca = numero de padroes que bateram / total de padroes do tipo.
    """
    q_lower  = query.lower()
    best_type  = "ambigua"
    best_score = 0.0
    for qtype, patterns in _QUERY_PATTERNS.items():
        matches = sum(1 for p in patterns if re.search(p, q_lower))
        score   = matches / len(patterns)
        if score > best_score:
            best_score = score
            best_type  = qtype
    return best_type, round(best_score, 3)


def _specificity_score(query: str, words: list[str]) -> float:
    """
    Estimativa de especificidade da query.
    Heuristica: combina comprimento, presenca de termos concretos
    (numeros, nomes proprios) e ausencia de pronomes vagos.
    Range: 0.0 (muito vaga) a 1.0 (muito especifica).
    """
    score = 0.0
    # Comprimento contribui para especificidade (ate 0.3)
    score += min(len(words) / 30, 0.3)
    # Presenca de numeros (0.2)
    if re.search(r'\d+', query):
        score += 0.2
    # Ausencia de pronomes vagos (0.2)
    vague_pronouns = ['isso', 'ele', 'ela', 'eles', 'aquilo', 'algo', 'alguem']
    if not any(p in query.lower() for p in vague_pronouns):
        score += 0.2
    # Presenca de substantivos concretos via maiusculas (0.15)
    proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
    if proper_nouns:
        score += 0.15
    # Presenca de aspas ou termos tecnicos (0.15)
    if '"' in query or "'" in query or re.search(r'[A-Z]{2,}', query):
        score += 0.15
    return round(min(score, 1.0), 3)


def _formality_score(text: str) -> float:
    """
    Estimativa de formalidade do texto.
    Heuristica: penaliza girias, contrações informais, exclamacoes excessivas.
    Range: 0.0 (muito informal) a 1.0 (muito formal).
    """
    informal_markers = [
        r'\bvc\b', r'\bpq\b', r'\bq\b', r'\bsim\b', r'\bblz\b',
        r'\bvaleu\b', r'\bflw\b', r'\bohh\b', r'\bahh\b', r'\bkkkk+\b',
        r'\brsrs+\b', r'\bein\b', r'\bhuhu+\b', r'\bnao\b',
    ]
    text_lower = text.lower()
    informal_count = sum(1 for p in informal_markers if re.search(p, text_lower))
    excl_excess = max(0, text.count('!') - 1)
    penalty = (informal_count * 0.1) + (excl_excess * 0.05)
    return round(max(0.0, min(1.0, 1.0 - penalty)), 3)


def _cosine_similarity(vec_a, vec_b) -> float:
    """Cosine similarity entre dois vetores numpy."""
    try:
        import numpy as np
        dot   = np.dot(vec_a, vec_b)
        norm  = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        return round(float(dot / norm) if norm > 0 else 0.0, 4)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Dimensoes de analise -- funcoes privadas
# ---------------------------------------------------------------------------

def _surface(text: str) -> dict:
    """Metricas de superficie: contagens brutas."""
    words = _tokenize_words(text)
    sents = _tokenize_sentences(text)
    sent_lengths = [len(_tokenize_words(s)) for s in sents]
    mean_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0.0
    variance = (
        sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
        if len(sent_lengths) > 1 else 0.0
    )
    return {
        "chars":              len(text),
        "chars_no_space":     len(text.replace(" ", "")),
        "words":              len(words),
        "words_unique":       len(set(words)),
        "words_long":         sum(1 for w in words if len(w) > 6),
        "sentences":          len(sents),
        "sent_len_mean":      round(mean_len, 2),
        "sent_len_std":       round(math.sqrt(variance), 2),
        "sent_questions":     sum(1 for s in sents if "?" in s),
        "punctuation_commas": text.count(","),
        "punctuation_excl":   text.count("!"),
        "subquestions":       len(re.findall(r'\?', text)),
    }


def _lexical_richness(words: list[str]) -> dict:
    """TTR, MATTR, hapax, Brunet W."""
    if not words:
        return {"ttr": 0.0, "mattr": 0.0, "hapax_ratio": 0.0, "brunet_w": 0.0}
    from collections import Counter
    counts  = Counter(words)
    n       = len(words)
    v       = len(counts)
    hapax   = sum(1 for c in counts.values() if c == 1)
    return {
        "ttr":        round(v / n, 4),
        "mattr":      _mattr(words),
        "hapax_ratio": round(hapax / n, 4),
        "brunet_w":   _brunet_w(n, v),
    }


def _readability(text: str) -> dict:
    """Indices de legibilidade."""
    try:
        import textstat
        textstat.set_lang("en")   # textstat em PT tem limitacoes -- usamos formulas proprias
    except Exception:
        pass
    return {
        "flesch":       _flesch_pt(text),
        "gunning_fog":  _gunning_fog(text),
        "coleman_liau": _coleman_liau(text),
    }


def _pos_tagging(text: str, nlp) -> dict:
    """POS tagging com spaCy."""
    if nlp is None:
        return {k: 0.0 for k in [
            "noun_ratio", "verb_ratio", "adj_ratio",
            "adv_ratio", "pronoun_ratio", "content_word_ratio",
        ]}
    try:
        doc    = nlp(text[:5000])  # limita para performance
        total  = len([t for t in doc if not t.is_punct and not t.is_space])
        if total == 0:
            return {k: 0.0 for k in [
                "noun_ratio","verb_ratio","adj_ratio",
                "adv_ratio","pronoun_ratio","content_word_ratio",
            ]}
        nouns   = sum(1 for t in doc if t.pos_ in ("NOUN","PROPN"))
        verbs   = sum(1 for t in doc if t.pos_ == "VERB")
        adjs    = sum(1 for t in doc if t.pos_ == "ADJ")
        advs    = sum(1 for t in doc if t.pos_ == "ADV")
        prons   = sum(1 for t in doc if t.pos_ == "PRON")
        content = nouns + verbs + adjs + advs
        return {
            "noun_ratio":         round(nouns   / total, 4),
            "verb_ratio":         round(verbs   / total, 4),
            "adj_ratio":          round(adjs    / total, 4),
            "adv_ratio":          round(advs    / total, 4),
            "pronoun_ratio":      round(prons   / total, 4),
            "content_word_ratio": round(content / total, 4),
        }
    except Exception:
        return {k: 0.0 for k in [
            "noun_ratio","verb_ratio","adj_ratio",
            "adv_ratio","pronoun_ratio","content_word_ratio",
        ]}


def _rhetoric(text: str) -> dict:
    """Estrutura retorica: conectivos, hedges, assertividade."""
    t = text.lower()
    hedge   = _count_pattern_list(t, _HEDGES)
    certain = _count_pattern_list(t, _CERTAINTY)
    total_hc = hedge + certain
    assertive = sum(
        len(re.findall(p, t)) for p in _ASSERTIVE_VERBS_PATTERNS
    )
    return {
        "logical_conn":     _count_pattern_list(t, _LOGICAL_CONN),
        "adversative_conn": _count_pattern_list(t, _ADVERSATIVE_CONN),
        "additive_conn":    _count_pattern_list(t, _ADDITIVE_CONN),
        "causal_conn":      _count_pattern_list(t, _CAUSAL_CONN),
        "enum_markers":     _count_pattern_list(t, _ENUM_MARKERS),
        "hedge_count":      hedge,
        "certainty_count":  certain,
        "hedge_ratio":      round(hedge / total_hc, 4) if total_hc > 0 else 0.0,
        "assertive_verbs":  assertive,
    }


def _sentiment(text: str) -> dict:
    """Polaridade e subjetividade via TextBlob."""
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        return {
            "polarity":     round(float(blob.sentiment.polarity),     4),
            "subjectivity": round(float(blob.sentiment.subjectivity), 4),
        }
    except Exception:
        return {"polarity": 0.0, "subjectivity": 0.0}


def _ner(text: str, nlp) -> dict:
    """Named Entity Recognition com spaCy."""
    zero = {
        "persons":0,"orgs":0,"locations":0,
        "dates":0,"values":0,"total":0,
    }
    if nlp is None:
        return zero
    try:
        doc = nlp(text[:5000])
        persons   = sum(1 for e in doc.ents if e.label_ in ("PER","PERSON"))
        orgs      = sum(1 for e in doc.ents if e.label_ in ("ORG","ORGANIZATION"))
        locations = sum(1 for e in doc.ents if e.label_ in ("LOC","GPE","LOCATION"))
        dates     = sum(1 for e in doc.ents if e.label_ in ("DATE","TIME"))
        values    = sum(1 for e in doc.ents if e.label_ in ("MONEY","PERCENT","QUANTITY","CARDINAL"))
        total     = persons + orgs + locations + dates + values
        return {
            "persons":persons,"orgs":orgs,"locations":locations,
            "dates":dates,"values":values,"total":total,
        }
    except Exception:
        return zero


def _alignment(
    q_words: list[str],
    r_words: list[str],
    q_text:  str,
    r_text:  str,
    q_subquestions: int,
    embedder,
) -> dict:
    """
    Alinhamento semantico e lexical entre query e response.
    Calcula: similaridade, coverage, novelty, information gain.
    """
    # Overlap lexical
    q_set = set(q_words)
    r_set = set(r_words)
    overlap = len(q_set & r_set) / len(q_set) if q_set else 0.0

    # Effort ratio
    effort = len(r_words) / len(q_words) if q_words else 0.0

    # Information gain (entropia)
    h_q = _entropy(q_words)
    h_r = _entropy(r_words)
    info_gain = round(h_r - h_q, 4)

    # Topic coverage: heuristica baseada em overlap de keywords
    # Keywords = palavras longas (> 5 chars) da query
    keywords = [w for w in q_words if len(w) > 5]
    kw_covered = sum(1 for k in keywords if k in r_set)
    kw_coverage = round(kw_covered / len(keywords), 4) if keywords else 0.0

    # Sub-perguntas cobertas: proporcional ao keyword coverage
    subq_covered = round(q_subquestions * kw_coverage) if q_subquestions > 0 else 0
    topic_coverage = round(subq_covered / q_subquestions, 4) if q_subquestions > 0 else kw_coverage

    # Topicos novos na response (palavras unicas na response nao na query)
    new_words = r_set - q_set
    topics_introduced = len([w for w in new_words if len(w) > 6])

    # Similaridade semantica via embeddings
    sem_sim = 0.0
    if embedder is not None:
        try:
            vecs = embedder.encode([q_text[:512], r_text[:512]])
            sem_sim = _cosine_similarity(vecs[0], vecs[1])
        except Exception:
            sem_sim = 0.0

    return {
        "semantic_similarity": sem_sim,
        "lexical_overlap":     round(overlap, 4),
        "keyword_coverage":    kw_coverage,
        "topic_coverage_ratio": topic_coverage,
        "response_novelty":    round(1.0 - overlap, 4),
        "effort_ratio":        round(effort, 2),
        "information_gain":    info_gain,
        "topics_introduced":   topics_introduced,
    }


def _followup(query: str, prev_query: Optional[str], embedder) -> dict:
    """Detecta se a query atual e reformulacao da anterior."""
    if prev_query is None or embedder is None:
        return {"is_followup": False, "followup_similarity": 0.0}
    try:
        vecs = embedder.encode([query[:512], prev_query[:512]])
        sim  = _cosine_similarity(vecs[0], vecs[1])
        return {
            "is_followup":         sim >= 0.85,
            "followup_similarity": sim,
        }
    except Exception:
        return {"is_followup": False, "followup_similarity": 0.0}


# ---------------------------------------------------------------------------
# Funcao publica -- unica interface usada pelo app.py
# ---------------------------------------------------------------------------

def analyze(
    query:            str,
    response:         str,
    prev_query:       Optional[str] = None,
    session_position: int           = 1,
) -> TextMetrics:
    """
    Executa o pipeline completo de analise textual sobre o par (query, response).

    Parametros
    ----------
    query            : texto da mensagem do usuario
    response         : texto da resposta da LLM (ja sanitizado)
    prev_query       : query do turno anterior (para follow-up detection)
    session_position : posicao deste turno na sessao (1 = primeiro turno)

    Retorna
    -------
    TextMetrics com todos os campos preenchidos ou zerados (fail-safe).
    """
    t0 = time.time()

    # Carrega modelos (lazy -- so na primeira chamada)
    nlp      = _get_nlp()
    embedder = _get_embedder()
    _check_textblob()

    # Tokenizacao base -- usada por multiplas dimensoes
    q_words = _tokenize_words(query)
    r_words = _tokenize_words(response)

    # ── Dimensao 1: Superficie ───────────────────────────────────────────────
    q_surf = _surface(query)
    r_surf = _surface(response)
    r_surf_extra = {
        "paragraphs":   len([p for p in response.split("\n\n") if p.strip()]),
        "list_markers": len(re.findall(r'^\s*[-*•]\s|^\s*\d+\.\s', response, re.MULTILINE)),
    }

    # ── Dimensao 2: Riqueza lexical ──────────────────────────────────────────
    q_lex = _lexical_richness(q_words)
    r_lex = _lexical_richness(r_words)

    # ── Dimensao 3: Legibilidade ─────────────────────────────────────────────
    q_read = _readability(query)
    r_read = _readability(response)

    # ── Dimensao 4a: POS Tagging ─────────────────────────────────────────────
    r_pos = _pos_tagging(response, nlp)

    # ── Dimensao 4b: Estrutura retorica ──────────────────────────────────────
    r_rhet = _rhetoric(response)

    # ── Dimensao 4c: Sentimento ───────────────────────────────────────────────
    q_sent = _sentiment(query)
    r_sent = _sentiment(response)

    # ── Dimensao 4d: NER ─────────────────────────────────────────────────────
    r_ner = _ner(response, nlp)

    # ── Dimensao 4e: Query intent ────────────────────────────────────────────
    q_type, q_conf  = _classify_query_type(query)
    q_has_ctx       = len(q_words) >= 5 and any(len(w) > 6 for w in q_words)
    q_specificity   = _specificity_score(query, q_words)
    r_formality     = _formality_score(response)

    # ── Dimensao 5: Alinhamento ───────────────────────────────────────────────
    align = _alignment(
        q_words       = q_words,
        r_words       = r_words,
        q_text        = query,
        r_text        = response,
        q_subquestions= q_surf["subquestions"],
        embedder      = embedder,
    )

    # ── Follow-up detection ───────────────────────────────────────────────────
    fu = _followup(query, prev_query, embedder)

    # ── Tempo de processamento ────────────────────────────────────────────────
    processing_ms = round((time.time() - t0) * 1000, 1)

    return TextMetrics(
        # Superficie — Query
        q_chars               = q_surf["chars"],
        q_chars_no_space      = q_surf["chars_no_space"],
        q_words               = q_surf["words"],
        q_words_unique        = q_surf["words_unique"],
        q_words_long          = q_surf["words_long"],
        q_sentences           = q_surf["sentences"],
        q_sent_len_mean       = q_surf["sent_len_mean"],
        q_sent_len_std        = q_surf["sent_len_std"],
        q_sent_questions      = q_surf["sent_questions"],
        q_punctuation_commas  = q_surf["punctuation_commas"],
        q_punctuation_excl    = q_surf["punctuation_excl"],
        q_subquestions        = q_surf["subquestions"],
        # Superficie — Response
        r_chars               = r_surf["chars"],
        r_chars_no_space      = r_surf["chars_no_space"],
        r_words               = r_surf["words"],
        r_words_unique        = r_surf["words_unique"],
        r_words_long          = r_surf["words_long"],
        r_sentences           = r_surf["sentences"],
        r_sent_len_mean       = r_surf["sent_len_mean"],
        r_sent_len_std        = r_surf["sent_len_std"],
        r_paragraphs          = r_surf_extra["paragraphs"],
        r_list_markers        = r_surf_extra["list_markers"],
        r_punctuation_commas  = r_surf["punctuation_commas"],
        r_punctuation_excl    = r_surf["punctuation_excl"],
        # Riqueza lexical
        q_ttr                 = q_lex["ttr"],
        r_ttr                 = r_lex["ttr"],
        r_mattr               = r_lex["mattr"],
        r_hapax_ratio         = r_lex["hapax_ratio"],
        r_brunet_w            = r_lex["brunet_w"],
        # Legibilidade
        q_flesch              = q_read["flesch"],
        r_flesch              = r_read["flesch"],
        r_gunning_fog         = r_read["gunning_fog"],
        r_coleman_liau        = r_read["coleman_liau"],
        flesch_gap            = round(r_read["flesch"] - q_read["flesch"], 2),
        # POS tagging
        r_noun_ratio          = r_pos["noun_ratio"],
        r_verb_ratio          = r_pos["verb_ratio"],
        r_adj_ratio           = r_pos["adj_ratio"],
        r_adv_ratio           = r_pos["adv_ratio"],
        r_pronoun_ratio       = r_pos["pronoun_ratio"],
        r_content_word_ratio  = r_pos["content_word_ratio"],
        # Estrutura retorica
        r_logical_conn        = r_rhet["logical_conn"],
        r_adversative_conn    = r_rhet["adversative_conn"],
        r_additive_conn       = r_rhet["additive_conn"],
        r_causal_conn         = r_rhet["causal_conn"],
        r_enum_markers        = r_rhet["enum_markers"],
        r_hedge_count         = r_rhet["hedge_count"],
        r_certainty_count     = r_rhet["certainty_count"],
        r_hedge_ratio         = r_rhet["hedge_ratio"],
        r_assertive_verbs     = r_rhet["assertive_verbs"],
        # Sentimento e tom
        q_sentiment_polarity      = q_sent["polarity"],
        q_sentiment_subjectivity  = q_sent["subjectivity"],
        r_sentiment_polarity      = r_sent["polarity"],
        r_sentiment_subjectivity  = r_sent["subjectivity"],
        r_formality_score         = r_formality,
        # NER
        r_ner_persons         = r_ner["persons"],
        r_ner_orgs            = r_ner["orgs"],
        r_ner_locations       = r_ner["locations"],
        r_ner_dates           = r_ner["dates"],
        r_ner_values          = r_ner["values"],
        r_ner_total           = r_ner["total"],
        # Query intent
        q_type                = q_type,
        q_type_confidence     = q_conf,
        q_has_context         = q_has_ctx,
        q_specificity_score   = q_specificity,
        # Alinhamento
        semantic_similarity   = align["semantic_similarity"],
        lexical_overlap       = align["lexical_overlap"],
        keyword_coverage      = align["keyword_coverage"],
        topic_coverage_ratio  = align["topic_coverage_ratio"],
        response_novelty      = align["response_novelty"],
        effort_ratio          = align["effort_ratio"],
        complexity_gap        = round(r_read["flesch"] - q_read["flesch"], 2),
        information_gain      = align["information_gain"],
        # Follow-up e sessao
        is_followup           = fu["is_followup"],
        followup_similarity   = fu["followup_similarity"],
        session_position      = session_position,
        topics_introduced     = align["topics_introduced"],
        # Metadados
        analytics_version     = _VERSION,
        processing_ms         = processing_ms,
        spacy_available       = _spacy_ok,
        embedding_available   = _embed_ok,
    )


# ---------------------------------------------------------------------------
# Utilitario de diagnostico
# ---------------------------------------------------------------------------

def analytics_status() -> dict:
    """Retorna disponibilidade de cada dependencia para exibir na sidebar."""
    nlp      = _get_nlp()
    embedder = _get_embedder()
    tb_ok    = _check_textblob()
    return {
        "spacy":               _spacy_ok,
        "sentence_transformers": _embed_ok,
        "textblob":            tb_ok,
        "spacy_model":         _SPACY_MODEL,
        "embedding_model":     _EMBEDDING_MODEL,
        "version":             _VERSION,
    }
