"""Dataset catalog and source-text routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from .state import _get_runtime

router = APIRouter()


@router.get("/experts")
def experts() -> dict[str, Any]:
    runtime = _get_runtime()
    return {
        "experts": [
            {
                "expert_id": expert["expert_id"],
                "role": expert["role"],
                "speech_style": expert["speech_style"],
            }
            for expert in runtime.dataset["experts"]
        ],
        "dataset_generation": runtime.generation,
    }


@router.get("/source/{article_id}/{section_id}")
def source(article_id: str, section_id: str,
           generation: str | None = None) -> dict[str, Any]:
    runtime = _get_runtime()
    if generation is not None and generation != runtime.generation:
        raise HTTPException(
            status_code=409,
            detail="Source belongs to a previous dataset generation; reload the dataset",
        )
    entry = runtime.sections.get((article_id, section_id))
    if entry is None:
        raise HTTPException(status_code=404, detail="Source not found")
    article, section = entry
    return {
        "article_id": article["article_id"],
        "article_title": article["title"],
        "section_id": section["section_id"],
        "section_title": section["title"],
        "text": section["text"],
        "authors": article["authors"],
        "year": article["year"],
        "doi": article["doi"],
        "license": article["license"],
        "dataset_generation": runtime.generation,
    }


