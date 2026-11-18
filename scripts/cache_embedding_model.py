from __future__ import annotations

import argparse
from pathlib import Path

from spend_engine.embeddings import ensure_embedding_model_registered


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/cache the embedding model before the blind-test freeze")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache_models"))
    args = parser.parse_args()
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise SystemExit("Install the optional model dependency first: pip install -e '.[embeddings]'") from exc

    ensure_embedding_model_registered(TextEmbedding, args.model)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(model_name=args.model, cache_dir=str(args.cache_dir.resolve()), cuda=False)
    # A tiny warm-up verifies the model can actually infer after download.
    vector = next(model.embed(["query: industrial pump spare part"]))
    print(f"Model cached and warm: {args.model}")
    print(f"Vector dimension: {len(vector)}")
    print(f"Cache directory: {args.cache_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
