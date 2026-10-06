"""Shared multi-corpus picker used by the batch screens."""

from __future__ import annotations

import gradio as gr

from modules.utils import AITUBER_CORPUS_PRESET, describe_corpora, list_corpus_files


def build_corpus_selector(default: list[str] | None = None):
    """Multi-corpus picker (generation order = selection order) + AITuber preset."""
    corpus_files = list_corpus_files("ja")
    if default is None:
        default = corpus_files[:1]
    corpus = gr.Dropdown(
        choices=corpus_files,
        value=[name for name in default if name in corpus_files],
        multiselect=True,
        label="内蔵コーパス（複数可・選んだ順に生成）",
    )
    order = gr.Textbox(value=describe_corpora(default), label="生成順", interactive=False)
    preset = gr.Button("AITuberセット（感情100 → 配信200 → AIキャラ500）", variant="secondary", size="sm")
    gr.Markdown(
        "データを**追加**するときは、前回生成した文を先頭に残したまま後ろに足してください"
        "（例: 感情100で生成済み → 感情100＋配信200）。生成済みの行は再利用され、新しい文だけ生成されます。"
    )
    corpus.change(fn=describe_corpora, inputs=[corpus], outputs=[order])
    preset.click(fn=lambda: [name for name in AITUBER_CORPUS_PRESET if name in corpus_files],
                 outputs=[corpus])
    return corpus, order
