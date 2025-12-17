import argparse
from .rag import load_config, build_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    stats = build_index(cfg)
    print(f"Indexed files: {stats['files']}, chunks: {stats['chunks']}")


if __name__ == "__main__":
    main()
