import argparse
import os
from typing import List

from colorama import Fore, Style
from llama_cpp import Llama

from .rag import load_config, load_index_and_embedder


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers strictly using the provided context. "
    "If the answer is not in the context, say you don't know. Be concise."
)

PROMPT_TEMPLATE = (
    "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n"
    "Answer succinctly, citing filenames when relevant. [/INST]"
)


def load_llm(cfg: dict) -> Llama:
    model_path = cfg.get("model_path")
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}. Place a GGUF file and update config.yaml"
        )
    n_ctx = int(cfg.get("n_ctx", 4096))
    n_threads = cfg.get("n_threads")
    n_threads = None if n_threads in (None, "", "null") else int(n_threads)
    n_gpu_layers = int(cfg.get("n_gpu_layers", 0))

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    return llm


def answer_question(cfg_path: str, question: str, top_k: int = None):
    cfg = load_config(cfg_path)
    store, embedder = load_index_and_embedder(cfg)
    top_k = top_k or int(cfg.get("top_k", 5))

    q_vec = embedder.encode([question])[0]
    hits = store.search(q_vec, top_k)

    context, used = None, []
    if hits:
        context, used = __build_context(store, hits)

    llm = load_llm(cfg)
    prompt = PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, context=context or "(no relevant context)", question=question)

    # Prefer chat completion if available
    try:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.2,
        )
        text = out["choices"][0]["message"]["content"].strip()
    except Exception:
        out = llm(prompt, temperature=0.2)
        text = out["choices"][0]["text"].strip()

    return text, used


def __build_context(store, hits):
    from .rag import make_context_from_hits

    ctx, used = make_context_from_hits(store, hits, max_chars=6000)
    # append minimal provenance (filenames)
    sources = {}
    for m in used:
        src = os.path.basename(m.get("source", ""))
        if not src:
            continue
        sources.setdefault(src, 0)
        sources[src] += 1
    if sources:
        provenance = "\n\nSources: " + ", ".join([f"{k}({v})" for k, v in sources.items()])
        ctx = ctx + provenance
    return ctx, used


def repl(cfg_path: str):
    print(Fore.CYAN + "Local RAG — interactive mode (type 'exit' to quit)" + Style.RESET_ALL)
    while True:
        try:
            q = input(Fore.YELLOW + "? " + Style.RESET_ALL).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            break
        try:
            ans, used = answer_question(cfg_path, q)
            print(Fore.GREEN + "\n" + ans + "\n" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--q", dest="question", default=None)
    parser.add_argument("--repl", action="store_true")
    args = parser.parse_args()

    if args.repl:
        repl(args.config)
        return

    if not args.question:
        raise SystemExit("Provide --q 'your question' or use --repl")

    ans, used = answer_question(args.config, args.question)
    print(ans)


if __name__ == "__main__":
    main()
