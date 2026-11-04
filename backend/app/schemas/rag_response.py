from pydantic import BaseModel, Field, ConfigDict, model_validator


class Citation(BaseModel):
    """A citation pointing to a retrieved document chunk."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)


class RAGResponse(BaseModel):
    """Validated response returned by the RAG generation pipeline."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    is_answerable: bool

    @model_validator(mode="after")
    def validate_response_consistency(self):
        if self.is_answerable and not self.citations:
            raise ValueError(
                "An answerable response must contain at least one citation."
            )

        if not self.is_answerable and self.citations:
            raise ValueError(
                "An unanswerable response must not contain citations."
            )

        return self
