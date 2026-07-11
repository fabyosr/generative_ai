"""
=============================================================================
ui/pipeline_trace.py — Thinking Trace: Visualização do Pipeline em Tempo Real
=============================================================================
Responsabilidade:
    Renderizar no chat uma caixa de "pensamento" que mostra, passo a passo
    e em tempo real, tudo que o agente está processando — do recebimento
    do prompt até a geração da resposta final.

Design:
    - Caixa com fundo âmbar/dourado, ícone de engrenagem pulsante
    - Cada etapa aparece sequencialmente conforme é executada
    - Ícones de status: ⏳ (em execução) → ✅ (concluído) / ⚠️ (aviso)
    - Detalhes inline: scores, tokens, cache hit/miss, intenção detectada
    - Ao final, a caixa colapsa automaticamente em um resumo de uma linha

Padrão de uso (em app.py):
    trace = PipelineTrace()
    trace.start()

    with trace.step("🎯 Classificando intenção…"):
        intent_result = classify_intent(query, history, llm)
    trace.complete_step(f"Intenção: {intent_result.intent.value}")

    with trace.step("🗃 Verificando cache semântico…"):
        cache_result = cache.lookup(query)
    trace.complete_step(f"Cache: {'HIT ✅' if cache_result.hit else 'MISS ❌'}")

    # ... demais etapas

    trace.finish()
=============================================================================
"""

import time
import streamlit as st


# ---------------------------------------------------------------------------
# CSS da caixa de pensamento — injetado uma vez por sessão
# ---------------------------------------------------------------------------

THINKING_BOX_CSS = """
<style>
/* Caixa de pensamento do agente */
.thinking-box {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 1px solid #f59e0b;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0 12px 0;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 0.80rem;
    line-height: 1.7;
    color: #78350f;
    box-shadow: 0 2px 8px rgba(245, 158, 11, 0.12);
    position: relative;
}

.thinking-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    font-size: 0.82rem;
    color: #92400e;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px dashed #fcd34d;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.thinking-step {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 3px 0;
    animation: fadeIn 0.3s ease-in;
}

.thinking-step-icon {
    min-width: 18px;
    font-size: 0.85rem;
    margin-top: 1px;
}

.thinking-step-content {
    flex: 1;
}

.thinking-step-label {
    font-weight: 600;
    color: #92400e;
}

.thinking-step-detail {
    color: #a16207;
    font-size: 0.76rem;
    margin-top: 1px;
}

.thinking-divider {
    border: none;
    border-top: 1px dashed #fcd34d;
    margin: 8px 0;
}

.thinking-summary {
    font-size: 0.76rem;
    color: #a16207;
    font-style: italic;
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px dashed #fcd34d;
}

.thinking-badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-left: 4px;
}

.badge-hit   { background: #d1fae5; color: #065f46; }
.badge-miss  { background: #fee2e2; color: #991b1b; }
.badge-rag   { background: #e0e7ff; color: #3730a3; }
.badge-chat  { background: #f0fdf4; color: #166534; }
.badge-follow{ background: #fef3c7; color: #92400e; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
"""


# ---------------------------------------------------------------------------
# Helpers de formatação HTML
# ---------------------------------------------------------------------------

def _badge(text: str, kind: str = "rag") -> str:
    """Retorna um badge HTML colorido inline."""
    return f'<span class="thinking-badge badge-{kind}">{text}</span>'


def _step_html(icon: str, label: str, detail: str = "") -> str:
    """Monta o HTML de uma etapa do trace."""
    detail_html = f'<div class="thinking-step-detail">{detail}</div>' if detail else ""
    return (
        f'<div class="thinking-step">'
        f'  <div class="thinking-step-icon">{icon}</div>'
        f'  <div class="thinking-step-content">'
        f'    <div class="thinking-step-label">{label}</div>'
        f'    {detail_html}'
        f'  </div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# PipelineTrace — contexto de rastreamento em tempo real
# ---------------------------------------------------------------------------

class PipelineTrace:
    """
    Gerencia a renderização incremental do trace do pipeline no chat.

    Usa st.empty() para atualizar o placeholder conforme cada etapa
    do pipeline é concluída, criando o efeito de "pensamento ao vivo".

    Uso típico em app.py:
        trace = PipelineTrace()

        intent_result = trace.run_step(
            label   = "Classificando intenção",
            icon    = "🎯",
            fn      = lambda: classify_intent(query, history, llm),
            detail_fn = lambda r: f"→ {r.intent.value} ({r.method}, {r.latency_ms:.0f}ms)"
        )
    """

    def __init__(self):
        # Injeta CSS uma vez por sessão
        if "thinking_css_injected" not in st.session_state:
            st.markdown(THINKING_BOX_CSS, unsafe_allow_html=True)
            st.session_state.thinking_css_injected = True

        # Placeholder Streamlit que será atualizado incrementalmente
        self._placeholder  = st.empty()
        self._steps_html   = []          # HTML acumulado das etapas
        self._start_time   = time.perf_counter()
        self._active       = True

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------

    def run_step(
        self,
        label:     str,
        icon:      str,
        fn,
        detail_fn=None,
        skip_if:   bool = False,
        skip_msg:  str  = "",
    ):
        """
        Executa uma etapa do pipeline e atualiza o trace em tempo real.

        Mostra ⏳ enquanto `fn` executa, depois substitui por ✅ com detalhe.

        Args:
            label:     Texto descritivo da etapa (ex: "Verificando cache").
            icon:      Emoji da etapa (ex: "🗃").
            fn:        Callable sem argumentos que executa a etapa.
            detail_fn: Callable que recebe o resultado de fn() e retorna
                       string HTML de detalhe para exibir abaixo do label.
            skip_if:   Se True, pula a etapa e exibe skip_msg.
            skip_msg:  Mensagem exibida quando a etapa é pulada.

        Returns:
            Resultado de fn(), ou None se skip_if=True.
        """
        if not self._active:
            return fn() if not skip_if else None

        if skip_if:
            self._add_step("⏭️", label, skip_msg, skipped=True)
            return None

        # Mostra etapa em execução
        self._add_step("⏳", label, "executando…", running=True)

        # Executa
        result = fn()

        # Atualiza com resultado
        detail = detail_fn(result) if detail_fn else ""
        self._steps_html[-1] = _step_html("✅", label, detail)
        self._render()

        return result

    def add_info(self, icon: str, label: str, detail: str = "") -> None:
        """
        Adiciona uma linha informativa ao trace sem executar função.

        Útil para exibir metadados (ex: "Modelo: gpt-4o-mini").

        Args:
            icon:   Emoji.
            label:  Texto principal.
            detail: Texto secundário (opcional).
        """
        self._add_step(icon, label, detail)

    def add_divider(self) -> None:
        """Adiciona linha divisória horizontal no trace."""
        self._steps_html.append('<hr class="thinking-divider"/>')
        self._render()

    def finish(self, summary: str = "") -> None:
        """
        Finaliza o trace, colapsa para modo summary e congela o placeholder.

        Args:
            summary: Linha de resumo exibida no rodapé da caixa
                     (ex: "Pipeline completo em 3.2s · RAG · Reranker · Cache miss").
        """
        elapsed = time.perf_counter() - self._start_time

        if summary:
            self._steps_html.append(
                f'<div class="thinking-summary">⏱ {elapsed:.2f}s total · {summary}</div>'
            )

        self._render()
        self._active = False

    # -----------------------------------------------------------------------
    # Métodos privados
    # -----------------------------------------------------------------------

    def _add_step(
        self,
        icon:    str,
        label:   str,
        detail:  str  = "",
        running: bool = False,
        skipped: bool = False,
    ) -> None:
        """Adiciona etapa à lista interna e re-renderiza."""
        if skipped:
            effective_icon = "⏭️"
        elif running:
            effective_icon = "⏳"
        else:
            effective_icon = icon

        self._steps_html.append(_step_html(effective_icon, label, detail))
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
# Helpers de detail_fn prontos para uso em app.py
# ---------------------------------------------------------------------------

def intent_detail(result) -> str:
    """Detail HTML para a etapa de classificação de intenção."""
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
        f"{result.latency_ms:.0f}ms "
        f"{badge}"
    )


def cache_detail(result) -> str:
    """Detail HTML para a etapa de lookup no cache."""
    if result.hit:
        return (
            f'Similaridade: <strong>{result.similarity:.4f}</strong> '
            f'<span class="thinking-badge badge-hit">HIT ✅</span> · '
            f'{result.latency_ms:.0f}ms · '
            f'Query casada: "<em>{result.matched_query[:60]}…</em>"'
        )
    return (
        f'Similaridade máx: {result.similarity:.4f} '
        f'<span class="thinking-badge badge-miss">MISS ❌</span> · '
        f'{result.latency_ms:.0f}ms'
    )


def rerank_detail(result) -> str:
    """Detail HTML para a etapa de reranking."""
    scores_str = " · ".join(f"#{i+1}: {s:.3f}" for i, s in enumerate(result.scores))
    return (
        f"{result.docs_before} → {result.docs_after} chunks · "
        f"Top score: <strong>{result.top_score:.4f}</strong> · "
        f"{result.latency_ms:.0f}ms<br>"
        f"<span style='font-size:0.72rem;'>{scores_str}</span>"
    )


def rag_retrieval_detail(context_docs: list) -> str:
    """Detail HTML para a etapa de recuperação RAG."""
    import os
    sources = list({os.path.basename(d.metadata.get("source", "?")) for d in context_docs})
    return (
        f"{len(context_docs)} chunks recuperados · "
        f"Fontes: {', '.join(f'<code>{s}</code>' for s in sources)}"
    )
