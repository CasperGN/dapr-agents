from dapr_agents.document.embedder.base import EmbedderBase
from typing import List, Any, Optional, Union, Literal
from sentence_transformers import SentenceTransformer
from pydantic import ConfigDict, Field
import logging
import os

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(EmbedderBase):
    """
    SentenceTransformer-based embedder for generating text embeddings.
    Supports multi-process encoding for large datasets.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: SentenceTransformer = SentenceTransformer()

    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Name of the SentenceTransformer model to use.",
    )
    device: Literal["cpu", "cuda", "mps", "npu"] = Field(
        default="cpu", description="Device for computation."
    )
    normalize_embeddings: bool = Field(
        default=False, description="Whether to normalize embeddings."
    )
    multi_process: bool = Field(
        default=False, description="Whether to use multi-process encoding."
    )
    cache_dir: Optional[str] = Field(
        default=None, description="Directory to cache or load the model."
    )
    client: SentenceTransformer = Field(
        default=client, init=False, description="Loaded SentenceTransformer model."
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize the SentenceTransformer model after validation.
        """
        super().model_post_init(__context)

        # Determine whether to load from cache or download
        model_path = (
            self.cache_dir
            if self.cache_dir and os.path.exists(self.cache_dir)
            else self.model
        )
        # Attempt to load the model
        try:
            if os.path.exists(model_path):
                logger.info(
                    f"Loading SentenceTransformer model from local path: {model_path}"
                )
            else:
                logger.info(f"Downloading SentenceTransformer model: {self.model}")
                if self.cache_dir:
                    logger.info(f"Model will be cached to: {self.cache_dir}")
            self.client: SentenceTransformer = SentenceTransformer(
                model_name_or_path=model_path, device=self.device
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise
        # Save to cache directory if downloaded
        if (
            model_path == self.model
            and self.cache_dir
            and not os.path.exists(self.cache_dir)
        ):
            logger.info(f"Saving the downloaded model to: {self.cache_dir}")
            self.client.save(self.cache_dir)

    def embed(  # type: ignore[override]
        self, input: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for input text(s).

        Args:
            input (Union[str, List[str]]): Input text(s) to embed. Can be a single string or a list of strings.

        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s) for the input(s).
                - A single embedding vector (list of floats) for a single string input.
                - A list of embedding vectors for a list of string inputs.
        """
        if not input or (isinstance(input, list) and all(not q for q in input)):
            raise ValueError("Input must contain valid text.")

        input_strings: list[str] = [input] if isinstance(input, str) else input

        logger.info(f"Generating embeddings for {len(input_strings)} input(s).")

        if self.multi_process:
            logger.info("Starting multi-process pool for encoding.")
            pool = self.client.start_multi_process_pool()

            try:
                embeddings = self.client.encode_multi_process(
                    input_strings,
                    pool=pool,
                    normalize_embeddings=self.normalize_embeddings,
                )
            finally:
                logger.info("Stopping multi-process pool.")
                self.client.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(  # type: ignore[assignment]
                input_strings,  # this is a false positive as encode returns ndarray, mypy reads "Tensor"
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )

        return [float(emb) for emb in embeddings]

    def __call__(
        self, input: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Allows the instance to be called directly to embed text(s).

        Args:
            input (Union[str, List[str]]): The input text(s) to embed.

        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s) for the input(s).
        """
        return self.embed(input)
