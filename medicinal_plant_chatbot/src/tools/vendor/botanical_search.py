"""
Módulo de Busca Semântica Botânica com Dual-Encoder.

Este módulo encapsula uma arquitetura dual-encoder (imagem + texto) treinada
com aprendizado contrastivo para realizar buscas cross-modais em um acervo
de plantas medicinais. Ele foi extraído de uma aplicação Streamlit e projetado
para ser importado e utilizado em qualquer outro projeto Python.

Principais capacidades:
    - Busca Texto → Imagem (query textual encontra protótipos visuais)
    - Busca Imagem → Texto (query visual encontra protótipos textuais)
    - Carregamento automático de pesos, protótipos e metadados via Google Drive
    - Normalização L2 e similaridade de cosseno para ranking

Dependências principais:
    torch, torchvision, transformers, pandas, numpy, pillow, gdown, scikit-learn

Exemplo de uso básico:

    from botanical_search import BotanicalSearchEngine

    engine = BotanicalSearchEngine()          # faz download (se necessário) e carrega tudo
    resultados = engine.search_by_text("folhas alongadas digestivas", top_k=5)
    for r in resultados:
        print(r["plant_name"], r["similarity"])

"""

from __future__ import annotations

import os
import shutil
from typing import List, Dict, Any, Union, Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from PIL import Image
import gdown
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES
# =============================================================================

class Config:
    """
    Centraliza todos os caminhos, hiperparâmetros e IDs de recursos externos.

    Motivo da existência:
        Evita "números mágicos" espalhados pelo código e facilita a manutenção.
        Qualquer mudança de caminho, dimensão de embedding ou modelo de
        tokenização é feita em um único lugar.
    """

    # Diretórios locais onde os artefatos serão gravados após o download
    SAVE_DIR: str = "./saved_model_components"
    EMBEDDINGS_DIR: str = "./saved_embeddings_and_metadata"
    PLANT_IMAGES_DIR: str = os.path.join(EMBEDDINGS_DIR, "plant_images")

    # IDs de pastas públicas no Google Drive
    # (contêm os pesos do dual-encoder e os protótipos + metadados)
    MODEL_COMPONENTS_GD_ID: str = "1vPnnFsO_IsDs_I4oE73KGD5Y_BpM0dac"
    EMBEDDINGS_GD_ID: str = "10NFY8TiwMwlBnnfBwZ2hVx98M3kfQGg5"

    # Modelo de linguagem base (BERT em português)
    TOKENIZER_MODEL: str = "neuralmind/bert-base-portuguese-cased"

    # Dimensão do espaço de embedding compartilhado (imagem e texto)
    EMBED_DIM: int = 512

    # Dispositivo de execução (GPU se disponível, senão CPU)
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# =============================================================================
# 2. GERENCIAMENTO DE DADOS (DOWNLOADS)
# =============================================================================

class DataManager:
    """
    Responsável por baixar e organizar os artefatos necessários ao funcionamento
    do dual-encoder (pesos do modelo + protótipos + metadados + imagens).

    Motivo da existência:
        Separar a lógica de I/O e download da lógica de inferência. Assim o
        SearchEngine pode assumir que os arquivos já existem localmente.
    """

    @staticmethod
    def _flatten_if_needed(target_dir: str) -> None:
        """
        Corrige o comportamento do gdown que, em alguns casos, cria uma
        subpasta extra com o mesmo nome do arquivo/pasta baixada.

        Parâmetros
        ----------
        target_dir : str
            Diretório onde o download foi realizado.

        Motivo:
            O gdown às vezes "envelopa" o conteúdo em uma pasta intermediária.
            Esta rotina detecta o caso (exatamente um item e esse item é pasta)
            e move o conteúdo para o nível superior, removendo a pasta vazia.
        """
        items = os.listdir(target_dir)

        # Só age se existir exatamente um item e esse item for um diretório
        if len(items) == 1:
            subfolder = os.path.join(target_dir, items[0])
            if os.path.isdir(subfolder):
                for item in os.listdir(subfolder):
                    shutil.move(
                        os.path.join(subfolder, item),
                        os.path.join(target_dir, item),
                    )
                os.rmdir(subfolder)

    @classmethod
    def download_assets(cls) -> None:
        """
        Garante que os componentes do modelo e os embeddings/protótipos
        estejam presentes localmente. Caso contrário, faz o download via
        Google Drive.

        Motivo:
            A aplicação precisa dos arquivos .pth, .pt, .csv e das imagens
            de plantas. Esta rotina torna o módulo auto-contido: na primeira
            execução os dados são baixados; nas seguintes eles já estão em disco.
        """
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        os.makedirs(Config.EMBEDDINGS_DIR, exist_ok=True)

        try:
            # --- Componentes arquiteturais (pesos do dual-encoder) ---
            weights_path = os.path.join(
                Config.SAVE_DIR, "dual_encoder_model_weights.pth"
            )
            if not os.path.exists(weights_path):
                print("🌿 Baixando componentes arquiteturais do modelo...")
                gdown.download_folder(
                    id=Config.MODEL_COMPONENTS_GD_ID,
                    output=Config.SAVE_DIR,
                    quiet=True,
                    use_cookies=False,
                )
                cls._flatten_if_needed(Config.SAVE_DIR)

            # --- Protótipos, matrizes e acervo botânico ---
            metadata_path = os.path.join(Config.EMBEDDINGS_DIR, "metadata.csv")
            if not os.path.exists(metadata_path):
                print("🍃 Baixando protótipos, matrizes e acervo botânico...")
                gdown.download_folder(
                    id=Config.EMBEDDINGS_GD_ID,
                    output=Config.EMBEDDINGS_DIR,
                    quiet=True,
                    use_cookies=False,
                )
                cls._flatten_if_needed(Config.EMBEDDINGS_DIR)

        except Exception as e:
            raise RuntimeError(
                f"❌ Erro crítico de conexão com a nuvem: {str(e)}"
            ) from e


# =============================================================================
# 3. ARQUITETURA DO MODELO (DUAL-ENCODER)
# =============================================================================

class ImageEncoder(nn.Module):
    """
    Encoder de imagem baseado em ResNet-50 (ou outro backbone torchvision).

    O backbone extrai features visuais; uma camada linear de projeção mapeia
    essas features para o espaço de embedding compartilhado de dimensão
    `embed_dim`.

    Parâmetros
    ----------
    model_name : str
        Nome do backbone torchvision (padrão: 'resnet50').
    embed_dim : int
        Dimensão do vetor de embedding de saída.
    freeze_backbone : bool
        Se True, congela os pesos do backbone (apenas a projeção é treinável).
        Útil em fine-tuning contrastivo ou em inferência pura.

    Motivo do design:
        Em arquiteturas dual-encoder contrastivas é comum congelar o backbone
        pré-treinado e treinar apenas a cabeça de projeção, reduzindo
        overfitting e custo computacional.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        embed_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        # Carrega o backbone com pesos ImageNet
        self.backbone = models.__dict__[model_name](pretrained=True)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remove a cabeça de classificação original e substitui por Identity
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Camada de projeção para o espaço compartilhado
        self.projection = nn.Linear(in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parâmetros
        ----------
        x : torch.Tensor
            Batch de imagens normalizadas (B, C, H, W).

        Retorno
        -------
        torch.Tensor
            Embeddings de dimensão (B, embed_dim).
        """
        features = self.backbone(x)
        return self.projection(features)


class TextEncoder(nn.Module):
    """
    Encoder de texto baseado em BERT (neuralmind/bert-base-portuguese-cased).

    Utiliza mean-pooling mascarado sobre o last_hidden_state e depois projeta
    para o espaço de embedding compartilhado.

    Parâmetros
    ----------
    model_name : str
        Nome do modelo Hugging Face.
    embed_dim : int
        Dimensão do vetor de embedding de saída.
    freeze_backbone : bool
        Se True, congela os pesos do BERT.

    Motivo do mean-pooling mascarado:
        Evita que tokens de padding influenciem o vetor final. A média é
        calculada apenas sobre as posições onde attention_mask == 1.
    """

    def __init__(
        self,
        model_name: str = Config.TOKENIZER_MODEL,
        embed_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.config.hidden_size
        self.projection = nn.Linear(in_features, embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass com mean-pooling mascarado.

        Parâmetros
        ----------
        input_ids : torch.Tensor
            IDs dos tokens (B, seq_len).
        attention_mask : torch.Tensor
            Máscara de atenção (B, seq_len).

        Retorno
        -------
        torch.Tensor
            Embeddings de dimensão (B, embed_dim).
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Aplica a máscara para zerar posições de padding
        masked_hidden = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        # Mean-pooling
        pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
        return self.projection(pooled)


class DualEncoder(nn.Module):
    """
    Arquitetura dual-encoder completa: ImageEncoder + TextEncoder.

    Os dois ramos projetam para o mesmo espaço de dimensão `embed_dim`,
    permitindo cálculo de similaridade de cosseno cross-modal.

    Parâmetros
    ----------
    embed_dim : int
        Dimensão do espaço compartilhado.
    freeze_encoders : bool
        Se True, congela ambos os backbones.

    Motivo:
        Em sistemas de recuperação multimodal (CLIP-like) cada modalidade
        possui seu próprio encoder, mas ambos aprendem a mapear para um
        espaço comum onde pares positivos ficam próximos.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        freeze_encoders: bool = True,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            embed_dim=embed_dim,
            freeze_backbone=freeze_encoders,
        )
        self.text_encoder = TextEncoder(
            embed_dim=embed_dim,
            freeze_backbone=freeze_encoders,
        )


# =============================================================================
# 4. MOTOR DE BUSCA (LÓGICA DE PROTÓTIPOS)
# =============================================================================

class SearchEngine:
    """
    Motor de busca semântica que utiliza protótipos pré-computados.

    Em vez de comparar a query contra todas as imagens/textos individuais,
    o sistema mantém um protótipo (média) por classe de planta. Isso reduz
    drasticamente o custo de busca e estabiliza o ranking.

    Fluxos suportados:
        - Texto  → protótipos de imagem  (cross-modal)
        - Imagem → protótipos de texto   (cross-modal)

    Motivo do uso de protótipos:
        Em bases com múltiplas imagens por espécie, o protótipo captura a
        "essência" visual/textual da planta, tornando a recuperação mais
        robusta a variações de ângulo, iluminação e formulação textual.
    """

    def __init__(self, auto_download: bool = True):
        """
        Inicializa o motor: opcionalmente baixa os artefatos e carrega
        modelo + dados em memória.

        Parâmetros
        ----------
        auto_download : bool
            Se True (padrão), chama DataManager.download_assets() automaticamente.
        """
        if auto_download:
            DataManager.download_assets()

        self.tokenizer, self.transform, self.model = self._load_model()
        (
            self.metadata_df,
            self.proto_names,
            self.proto_img_norm,
            self.proto_txt_norm,
        ) = self._load_data()

    @staticmethod
    def _load_model():
        """
        Carrega o tokenizer, as transformações de imagem e os pesos do
        dual-encoder.

        Retorno
        -------
        tuple
            (tokenizer, transform, model) já no dispositivo correto e em modo eval.

        Motivo de ser estático e separado:
            Facilita o cache em ambientes que suportam (ex.: Streamlit) e
            torna o carregamento testável isoladamente.
        """
        tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_MODEL)

        # Transformações padrão ImageNet (mesmas usadas no treinamento)
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        model = DualEncoder(
            embed_dim=Config.EMBED_DIM,
            freeze_encoders=True,
        )
        weights_path = os.path.join(
            Config.SAVE_DIR, "dual_encoder_model_weights.pth"
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(Config.DEVICE).eval()

        return tokenizer, transform, model

    @staticmethod
    def _load_data():
        """
        Carrega o CSV de metadados e o arquivo de protótipos (.pt).

        Constrói e normaliza (L2) as matrizes de protótipos de imagem e de
        texto, preparando-as para cálculo rápido de similaridade de cosseno.

        Retorno
        -------
        tuple
            (metadata_df, proto_names, proto_img_norm, proto_txt_norm)
        """
        metadata_df = pd.read_csv(
            os.path.join(Config.EMBEDDINGS_DIR, "metadata.csv")
        )

        checkpoint = torch.load(
            os.path.join(Config.EMBEDDINGS_DIR, "prototypes.pt"),
            map_location="cpu",
        )
        prototypes = checkpoint["prototypes"]
        proto_names = checkpoint["proto_names"]

        # Matriz de protótipos de imagem → normalização L2
        proto_image_matrix = np.stack(
            [prototypes[p]["image_proto"].cpu().numpy() for p in proto_names]
        )
        proto_img_norm = proto_image_matrix / np.linalg.norm(
            proto_image_matrix, axis=1, keepdims=True
        )

        # Matriz de protótipos de texto → normalização L2
        proto_text_matrix = np.stack(
            [prototypes[p]["text_proto"].cpu().numpy() for p in proto_names]
        )
        proto_txt_norm = proto_text_matrix / np.linalg.norm(
            proto_text_matrix, axis=1, keepdims=True
        )

        return metadata_df, proto_names, proto_img_norm, proto_txt_norm

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Busca cross-modal: texto → protótipos de imagem.

        A query textual é codificada pelo TextEncoder, normalizada e comparada
        (similaridade de cosseno) com todos os protótipos visuais.

        Parâmetros
        ----------
        query_text : str
            Descrição textual da planta ou de suas propriedades.
        top_k : int
            Número máximo de resultados a retornar.

        Retorno
        -------
        List[Dict[str, Any]]
            Lista de dicionários ordenados por similaridade decrescente.
            Cada dicionário contém:
                - rank (int)
                - plant_name (str)
                - description (str)
                - similarity (float)
                - image_filename (str)
        """
        tokens = self.tokenizer(
            query_text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(Config.DEVICE)

        with torch.no_grad():
            query_emb = self.model.text_encoder(
                tokens["input_ids"],
                tokens["attention_mask"],
            ).cpu().numpy()

        # Normalização L2 da query
        query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

        # Similaridade de cosseno contra protótipos de imagem
        similarities = cosine_similarity(query_norm, self.proto_img_norm).flatten()
        return self._format_results(similarities, top_k)

    def search_by_image(
        self,
        image_input: Union[str, Image.Image],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Busca cross-modal: imagem → protótipos de texto.

        A imagem de consulta é codificada pelo ImageEncoder, normalizada e
        comparada com os protótipos textuais.

        Parâmetros
        ----------
        image_input : str | PIL.Image.Image
            Caminho para o arquivo de imagem OU objeto PIL já aberto.
        top_k : int
            Número máximo de resultados a retornar.

        Retorno
        -------
        List[Dict[str, Any]]
            Mesma estrutura retornada por `search_by_text`.
        """
        # Aceita tanto caminho (str) quanto objeto PIL
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        transformed = self.transform(image).unsqueeze(0).to(Config.DEVICE)

        with torch.no_grad():
            query_emb = self.model.image_encoder(transformed).cpu().numpy()

        query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

        # Similaridade de cosseno contra protótipos de texto
        similarities = cosine_similarity(query_norm, self.proto_txt_norm).flatten()
        return self._format_results(similarities, top_k)

    def _format_results(
        self,
        similarities: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Converte o vetor de similaridades em uma lista de resultados
        estruturados, enriquecidos com metadados do CSV.

        Parâmetros
        ----------
        similarities : np.ndarray
            Vetor de similaridades de cosseno (um valor por protótipo).
        top_k : int
            Quantidade de resultados desejados.

        Retorno
        -------
        List[Dict[str, Any]]
            Resultados ordenados por similaridade decrescente.
        """
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for rank, idx in enumerate(top_indices):
            plant_name = self.proto_names[idx]
            # Localiza a linha correspondente no DataFrame de metadados
            row = self.metadata_df[
                self.metadata_df["plant_name"] == plant_name
            ].iloc[0]

            results.append(
                {
                    "rank": rank + 1,
                    "plant_name": plant_name,
                    "description": row["description"],
                    "similarity": float(similarities[idx]),
                    "image_filename": row["image_filename"],
                }
            )

        return results

    def get_image_path(self, image_filename: str) -> str:
        """
        Utilitário conveniente: retorna o caminho completo de uma imagem
        do acervo a partir do nome do arquivo.

        Parâmetros
        ----------
        image_filename : str
            Nome do arquivo (como aparece no metadata.csv).

        Retorno
        -------
        str
            Caminho absoluto/relativo completo.
        """
        return os.path.join(Config.PLANT_IMAGES_DIR, image_filename)


# =============================================================================
# 5. FACADE / API PÚBLICA SIMPLIFICADA
# =============================================================================

class BotanicalSearchEngine(SearchEngine):
    """
    Alias público mais descritivo para a classe SearchEngine.

    Motivo:
        Em bibliotecas é comum oferecer um nome "de fachada" que deixa claro
        o domínio da ferramenta. Internamente é exatamente o mesmo motor.
    """
    pass


# =============================================================================
# 6. EXEMPLO DE USO (quando o módulo é executado diretamente)
# =============================================================================

if __name__ == "__main__":
    print("Inicializando BotanicalSearchEngine...")
    engine = BotanicalSearchEngine()

    # Exemplo 1: busca por texto
    print("\n--- Busca por texto ---")
    text_results = engine.search_by_text(
        "folhas alongadas com propriedades digestivas",
        top_k=3,
    )
    for r in text_results:
        print(
            f"#{r['rank']} {r['plant_name']} "
            f"(sim={r['similarity']:.3f})"
        )

    print("\nMódulo carregado com sucesso. Importe e use em suas aplicações.")
