import argparse
import os
import gradio as gr

# Force offline mode to avoid network calls
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from .rag import load_config, load_index_and_embedder, make_context_from_hits
from .query import load_llm, SYSTEM_PROMPT


def create_ui(cfg_path: str):
    cfg = load_config(cfg_path)
    store, embedder = load_index_and_embedder(cfg)
    llm = load_llm(cfg)
    top_k = int(cfg.get("top_k", 5))

    def answer(question: str, history):
        if not question.strip():
            return "Please enter a question."

        # Retrieve
        q_vec = embedder.encode([question])[0]
        hits = store.search(q_vec, top_k)

        if not hits:
            context = "(no relevant context found)"
            sources_text = ""
        else:
            context, used = make_context_from_hits(store, hits, max_chars=6000)
            # Build sources summary
            import os
            sources = {}
            for m in used:
                src = os.path.basename(m.get("source", ""))
                if src:
                    sources.setdefault(src, 0)
                    sources[src] += 1
            if sources:
                sources_text = "\n\n**Sources**: " + ", ".join([f"{k} ({v} chunks)" for k, v in sources.items()])
            else:
                sources_text = ""

        # Generate
        try:
            out = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0.2,
                max_tokens=512,
            )
            text = out["choices"][0]["message"]["content"].strip()
        except Exception:
            # Fallback to completion
            prompt = (
                f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\n"
                f"Answer succinctly. [/INST]"
            )
            out = llm(prompt, temperature=0.2, max_tokens=512)
            text = out["choices"][0]["text"].strip()

        return text + sources_text

    interface = gr.ChatInterface(
        fn=answer,
        title="Local RAG — Document Q&A",
        description="Ask questions about your knowledge base. Powered by local embeddings + LLM.",
        examples=[
            "What are the main topics covered?",
            "Summarize the key takeaways.",
            "How do I set up the environment?",
        ],
    )
    return interface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    ui = create_ui(args.config)
    ui.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
