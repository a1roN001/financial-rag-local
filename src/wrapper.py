from ragas.embeddings import BaseRagasEmbeddings


class CustomE5Embeddings(BaseRagasEmbeddings):

  def __init__(
      self, client, model_name: str = "intfloat/multilingual-e5-small"
  ):
    super().__init__()
    self.client = client  # Сам объект SentenceTransformer храним здесь
    self.model = model_name  # В self.model кладем именно СТРОКУ для валидации

  def embed_query(self, text: str) -> list[float]:
    return self.client.encode(
        f"query: {text}", normalize_embeddings=True
    ).tolist()

  def embed_documents(self, texts: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {t}" for t in texts]
    return self.client.encode(prefixed, normalize_embeddings=True).tolist()

  async def aembed_query(self, text: str) -> list[float]:
    return self.embed_query(text)

  async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
    return self.embed_documents(texts)
