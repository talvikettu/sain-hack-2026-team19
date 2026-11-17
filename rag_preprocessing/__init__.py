"""Dependency-free corpus preparation and hybrid retrieval for case 4."""

from .pipeline import Config, build_index, load_articles, pack_context, search

__all__ = ["Config", "build_index", "load_articles", "pack_context", "search"]
