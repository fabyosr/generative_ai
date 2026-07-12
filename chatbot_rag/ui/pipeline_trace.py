"""
=============================================================================
ui/pipeline_trace.py — Thinking Trace: Visualização do Pipeline em Tempo Real
=============================================================================
Responsabilidade:
    Renderizar no chat uma caixa de "pensamento" que mostra, passo a passo
    e em tempo real, tudo que o agente está processando.

Design (v2):
    - Fundo escuro (slate-900) com borda esquerda índigo vibrante
      → contraste claro contra o fundo branco/cinza do chat do Streamlit
    - Tipografia monospace com texto claro sobre fundo escuro
    - Seção colapsável para bloco <think> do modelo (CoT interno)
    - Badges coloridos por tipo de intenção/status
    - Linha de resumo ao finalizar
=============================================================================
"""

import time
import streamlit as st


# ---------------------------------------------------------------------------
# CSS — injetado uma vez por sessão
# ---------------------------------------------------------------------------

THINKING_BOX_CSS = """
<style>
/* ============================================================
   Caixa principal de pensamento — fundo escuro para contrastar
   com o fundo claro do chat do Streamlit
   ============================================================ */
.thinking-box {
    background: #0f172a;                  /* slate-900 */
    border: 1px solid #334155;            /* slate-700 */
    border-left: 4px solid #6366f1;       /* indigo-500 — acento vibrante */
    border-radius: 10px;
    padding: 14px 18px;
    margin: 4px 0 14px 0;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
    font-size: 0.79rem;
    line-height: 1.75;
    color: #cbd5e1;                       /* slate-300 */
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    position: relative;
}

/* Header — título da caixa */
.thinking-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    font-size: 0.75rem;
    color: #818cf8;                       /* indigo-400 */
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e293b;     /* slate-800 */
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Cada linha de etapa */
.thinking-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 2px 0;
    animation: fadeSlideIn 0.25s ease-out;
}

.thinking-step-icon {
    min-width: 20px;
    font-size: 0.82rem;
    margin-top: 2px;
    opacity: 0.9;
}

.thinking-step-content { flex: 1; }

.thinking-step-label {
    font-weight: 600;
    color: #e2e8f0;                       /* slate-200 */
}

.thinking-step-detail {
    color: #94a3b8;                       /* slate-400 */
    font-size: 0.74rem;
    margin-top: 2px;
}

/* Divisória */
.thinking-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 8px 0;
}

/* Linha de resumo final */
.thinking-summary {
    font-size: 0.73rem;
    color: #64748b;                       /* slate-500 */
    font-style: italic;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #1e293b;
}

/* ============================================================
   Bloco <think> do modelo — CoT interno
   ============================================================ */
.thinking-cot-block {
    background: #1e293b;                  /* slate-800 */
    border: 1px solid #334155;
    border-left: 3px solid #a855f7;       /* purple-500 */
    border-radius: 6px;
    margin: 8px 0 4px 0;
    overflow: hidden;
}

.thinking-cot-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 7px 12px;
    cursor: pointer;
    font-size: 0.74rem;
    font-weight: 700;
    color: #c084fc;                       /* purple-400 */
    letter-spacing: 0.05em;
    text-transform: uppercase;
    user-select: none;
}

.thinking-cot-header:hover {
    background: rgba(168, 85, 247, 0.08);
}

.thinking-cot-content {
    padding: 10px 14px;
    font-size: 0.73rem;
    color: #94a3b8;
    line-height: 1.65;
    white-space: pre-wrap;
    border-top: 1px solid #334155;
    max-height: 260px;
    overflow-y: auto;
}

/* ============================================================
   Badges coloridos inline
   ============================================================ */
.thinking-badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 999px;
    font-size: 0.70rem;
    font-weight: 700;
    margin-left: 5px;
    vertical-align: middle;
}

.badge-hit    { background: #064e3b; color: #6ee7b7; }   /* emerald */
.badge-miss   { background: #450a0a; color: #fca5a5; }   /* red */
.badge-rag    { background: #1e1b4b; color: #a5b4fc; }   /* indigo */
.badge-chat   { background: #052e16; color: #86efac; }   /* green */
.badge-follow { background: #1c1917; color: #fbbf24; }   /* amber */
.badge-cot    { background: #2e1065; color: #c084fc; }   /* purple */

/* ============================================================
   Animação de entrada das etapas
   ============================================================ */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(-3px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
"""


# ---------------------------------------------------------------------------
# Helpers de HTML
# ---------------------------------------------------------------------------

def _step_html(icon: str, label: str, detail: str = "") -> str:
    """Monta o HTML de uma etapa do trace."""
    detail_html = (
        f'<div class="thinking-step-detail">{detail}</div>'
        if detail else ""
    )
    return (
        f'<div class="thinking-step">'
        f'  <div class="thinking-step-icon">{icon}</div>'
        f'  <div class="thinking-step-content">'
        f'    <div class="thinking-step-label">{label}</div>'
        f'    {detail_html}'
        f'  </div>'
        f'</div>'
    )


def _cot_block_html(think_content: str, think_tokens: int, think_lines: int) -> str:
    """
    Monta o HTML do bloco <think> colapsável.

    Exibe o raciocínio interno do modelo (CoT) numa caixa roxa
    separada, indicando claramente que é pensamento interno, não
    parte da resposta ao usuário.

    Args:
        think_content: Texto do raciocínio interno.
        think_tokens:  Estimativa de tokens do bloco.
        think_lines:   Número de linhas do bloco.

    Returns:
        str: HTML do bloco colapsável.
    """
    # Escapa HTML básico para evitar injeção
    safe = (
        think_content
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    badge = '<span class="thinking-badge badge-cot">CoT</span>'
    return (
        f'<div class="thinking-cot-block">'
        f'  <div class="thinking-cot-header">'
        f'    🧠 Raciocínio interno do modelo {badge}'
        f'    <span style="margin-left:auto;font-weight:400;color:#64748b;">'
        f'      ~{think_tokens} tok · {think_lines} linhas'
        f'    </span>'
        f'  </div>'
        f'  <div class="thinking-cot-content">{safe}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# PipelineTrace
# ---------------------------------------------------------------------------

class PipelineTrace:
    """
    Renderização incremental do pipeline trace no chat.

    Usa st.empty() para atualizar o placeholder a cada etapa,
    criando o efeito de "pensamento ao vivo" em tempo real.
    """

    def __init__(self):
        if "thinking_css_injected" not in st.session_state:
            st.markdown(THINKING_BOX_CSS, unsafe_allow_html=True)
            st.session_state.thinking_css_injected = True

        self._placeholder = st.empty()
        self._steps_html  = []
        self._start_time  = time.perf_counter()
        self._active      = True

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------

    def run_step(self, label: str, icon: str, fn, detail_fn=None,
                 skip_if: bool = False, skip_msg: str = ""):
        """
        Executa uma etapa e atualiza o trace em tempo real.

        Mostra ⏳ durante execução → ✅ com detalhe ao concluir.

        Args:
            label:     Texto da etapa.
            icon:      Emoji da etapa.
            fn:        Callable que executa a lógica da etapa.
            detail_fn: Callable(resultado) → str HTML de detalhe.
            skip_if:   Se True, pula a etapa mostrando skip_msg.
            skip_msg:  Mensagem quando etapa é pulada.

        Returns:
            Resultado de fn(), ou None se pulada.
        """
        if not self._active:
            return fn() if not skip_if else None

        if skip_if:
            self._add_step("⏭️", label, skip_msg)
            return None

        self._add_step("⏳", label, "executando…")
        result = fn()
        detail = detail_fn(result) if detail_fn else ""
        self._steps_html[-1] = _step_html("✅", label, detail)
        self._render()
        return result

    def add_info(self, icon: str, label: str, detail: str = "") -> None:
        """Adiciona linha informativa sem executar função."""
        self._add_step(icon, label, detail)

    def add_cot_block(self, think_result) -> None:
        """
        Adiciona o bloco de raciocínio interno (<think>) ao trace.

        Deve ser chamado após o LLM responder, passando o ThinkResult
        retornado por core.think_parser.parse_think().

        Args:
            think_result: ThinkResult de core.think_parser.
        """
        if not think_result.used_cot or not think_result.think_content:
            return

        cot_html = _cot_block_html(
            think_result.think_content,
            think_result.think_tokens,
            think_result.think_lines,
        )
        self._steps_html.append(cot_html)
        self._render()

    def add_divider(self) -> None:
        """Linha divisória horizontal."""
        self._steps_html.append('<hr class="thinking-divider"/>')
        self._render()

    def finish(self, summary: str = "") -> None:
        """
        Finaliza o trace com linha de resumo.

        Args:
            summary: Texto do rodapé (latência, status, custo).
        """
        elapsed = time.perf_counter() - self._start_time
        if summary:
            self._steps_html.append(
                f'<div class="thinking-summary">'
                f'⏱ {elapsed:.2f}s · {summary}'
                f'</div>'
            )
        self._render()
        self._active = False

    # -----------------------------------------------------------------------
    # Privados
    # -----------------------------------------------------------------------

    def _add_step(self, icon: str, label: str, detail: str = "") -> None:
        self._steps_html.append(_step_html(icon, label, detail))
        self._render()

    def _render(self) -> None:
        """Atualiza o placeholder com o HTML acumulado."""
        steps_joined = "\n".join(self._steps_html)
        html = (
            f'<div class="thinking-box">'
            f'  <div class="thinking-header">'
            f'    <span>⚙️</span>'
            f'    <span>Processamento do Agente</span>'
            f'  </div>'
            f'  {steps_joined}'
            f'</div>'
        )
        self._placeholder.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers de detail_fn para uso em app.py
# ---------------------------------------------------------------------------

def intent_detail(result) -> str:
    intent_map = {
        "chitchat":  ("💬 CHITCHAT",  "badge-chat"),
        "rag_query": ("🔍 RAG QUERY", "badge-rag"),
        "followup":  ("🔄 FOLLOWUP",  "badge-follow"),
    }
    label, cls = intent_map.get(result.intent.value, (result.intent.value, "badge-rag"))
    badge = f'<span class="thinking-badge {cls}">{label}</span>'
    return (
        f"Método: {result.method} · "
        f"Confiança: {result.confidence*100:.0f}% · "
        f"{result.latency_ms:.0f}ms {badge}"
    )


def cache_detail(result) -> str:
    if result.hit:
        return (
            f'Similaridade: <strong>{result.similarity:.4f}</strong> '
            f'<span class="thinking-badge badge-hit">HIT ✅</span> · '
            f'{result.latency_ms:.0f}ms · '
            f'Query: "<em>{(result.matched_query or "")[:55]}…</em>"'
        )
    return (
        f'Sim. máx: {result.similarity:.4f} '
        f'<span class="thinking-badge badge-miss">MISS ❌</span> · '
        f'{result.latency_ms:.0f}ms'
    )


def rerank_detail(result) -> str:
    scores_str = " · ".join(f"#{i+1}: {s:.3f}" for i, s in enumerate(result.scores))
    return (
        f"{result.docs_before} → {result.docs_after} chunks · "
        f"Top: <strong>{result.top_score:.4f}</strong> · "
        f"{result.latency_ms:.0f}ms<br>"
        f"<span style='font-size:0.71rem;color:#64748b;'>{scores_str}</span>"
    )


def rag_retrieval_detail(context_docs: list) -> str:
    import os
    sources = list({os.path.basename(d.metadata.get("source", "?")) for d in context_docs})
    return (
        f"{len(context_docs)} chunks · "
        f"Fontes: {', '.join(f'<code>{s}</code>' for s in sources)}"
    )


def cot_detail(think_result) -> str:
    """Detail para etapa de parsing do think."""
    if think_result.used_cot:
        return (
            f'<span class="thinking-badge badge-cot">CoT DETECTADO</span> · '
            f'~{think_result.think_tokens} tokens de raciocínio · '
            f'{think_result.think_ratio*100:.0f}% do output total'
        )
    return "Modelo não usou chain-of-thought explícito"
