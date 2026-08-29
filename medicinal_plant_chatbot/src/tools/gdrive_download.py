"""
Utilitário de download de pastas públicas do Google Drive via gdown.

Extraído do padrão já usado em tools/vendor/botanical_search.py
(DataManager.download_assets) para reutilização por outros artefatos
baixados sob demanda do Drive — em particular, o índice vetorial do RAG
(ver tools/rag.py, a ser implementado).

O conteúdo de tools/vendor/botanical_search.py NÃO é alterado — ele
mantém sua própria cópia da lógica de download, testada e funcional.
Este módulo é uma extração para NOVOS usos, não uma refatoração do
código existente.

Mantém a mesma correção de comportamento do gdown que
tools/vendor/botanical_search.py já aplica: em alguns casos o gdown
"envelopa" o conteúdo baixado em uma subpasta extra com o mesmo nome —
esta função achata essa estrutura automaticamente.
"""

from __future__ import annotations

import os
import shutil

import gdown


def _achatar_se_necessario(diretorio: str) -> None:
    """Corrige o comportamento do gdown que às vezes cria uma subpasta
    extra envolvendo o conteúdo baixado (mesmo problema documentado em
    tools/vendor/botanical_search.py::DataManager._flatten_if_needed)."""
    itens = os.listdir(diretorio)
    if len(itens) == 1:
        subpasta = os.path.join(diretorio, itens[0])
        if os.path.isdir(subpasta):
            for item in os.listdir(subpasta):
                shutil.move(os.path.join(subpasta, item), os.path.join(diretorio, item))
            os.rmdir(subpasta)


def baixar_pasta_do_drive(
    google_drive_folder_id: str,
    destino: str,
    forcar: bool = False,
) -> None:
    """Baixa uma pasta pública do Google Drive para `destino`, se ainda
    não existir localmente.

    Args:
        google_drive_folder_id: ID da pasta pública no Google Drive
            (o mesmo padrão de Config.MODEL_COMPONENTS_GD_ID em
            tools/vendor/botanical_search.py).
        destino: diretório local onde o conteúdo será salvo.
        forcar: se True, baixa novamente mesmo que `destino` já tenha
            conteúdo — útil para atualizar um artefato manualmente.

    Levanta RuntimeError com mensagem clara em caso de falha de rede ou
    permissão, em vez de deixar a exceção original do gdown se propagar
    sem contexto.
    """
    if os.path.exists(destino) and os.listdir(destino) and not forcar:
        return

    os.makedirs(destino, exist_ok=True)
    try:
        gdown.download_folder(
            id=google_drive_folder_id,
            output=destino,
            quiet=True,
            use_cookies=False,
        )
        _achatar_se_necessario(destino)
    except Exception as e:
        raise RuntimeError(
            f"Erro ao baixar pasta do Google Drive (id={google_drive_folder_id}) "
            f"para '{destino}': {e}"
        ) from e
