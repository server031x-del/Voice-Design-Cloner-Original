"""Reusable Gradio panel for dataset quality checks (see modules/audio_qc.py)."""

from __future__ import annotations

import logging

import gradio as gr

from config import OUTPUT_DIR
from modules import audio_qc
from modules.lora_pipeline import list_clone_sources

logger = logging.getLogger(__name__)


def build_qc_panel(default_folder: str | None = None) -> None:
    gr.Markdown(
        "### 品質チェック（学習前の推奨手順）\n"
        "生成した音声の **読み間違い・途中切れ・無音・音割れ・話速の異常** を検出し、"
        "LoRA学習のテキストリストから除外できます。\n"
        "信号チェックはCPUで数秒、ASR（Whisper）チェックはGPUで1文あたり約0.3〜1秒です"
        "（初回はモデルを自動でダウンロードします）。"
    )
    choices = list_clone_sources()
    with gr.Row():
        folder = gr.Dropdown(
            choices=choices,
            value=default_folder if default_folder in choices else (choices[0] if choices else None),
            label="対象フォルダ（output/ 以下）",
            scale=3,
        )
        refresh = gr.Button("更新", variant="secondary", scale=1)
    with gr.Row():
        use_asr = gr.Checkbox(value=True, label="ASR（Whisper）で原文と照合する")
        cer_threshold = gr.Slider(0.05, 1.0, value=audio_qc.DEFAULT_CER_THRESHOLD, step=0.05,
                                  label="CERしきい値（超えたらNG）")
    asr_model = gr.Textbox(value=audio_qc.DEFAULT_ASR_MODEL, label="ASRモデル（Hugging Face ID）")
    state_text = gr.Textbox(label="状態", interactive=False, lines=2)
    with gr.Row():
        run = gr.Button("品質チェックを実行", variant="primary", scale=2)
        apply = gr.Button("NGを学習リストから除外", variant="secondary")
        restore = gr.Button("除外を元に戻す", variant="secondary")
    table = gr.Dataframe(headers=audio_qc.REPORT_HEADERS, label="NG・注意のクリップ", wrap=True,
                         interactive=False)
    gr.Markdown(
        "NGの行は「一括クローン」で **「品質チェックNGの行だけ再生成」** にチェックして同じフォルダへ実行すると、"
        "別のseedで作り直せます。詳細は各フォルダの `qc_report.csv` を参照してください。"
    )

    def _status(selected):
        if not selected:
            return "", []
        return audio_qc.status_text(OUTPUT_DIR / selected), audio_qc.report_rows(OUTPUT_DIR / selected)

    def on_run(selected, asr, threshold, model_name, progress=gr.Progress()):
        if not selected:
            yield "エラー: フォルダを選択してください", []
            return
        try:
            for pct, payload in audio_qc.run_qc(
                OUTPUT_DIR / selected,
                use_asr=bool(asr),
                asr_model=(model_name or audio_qc.DEFAULT_ASR_MODEL).strip(),
                cer_threshold=float(threshold),
            ):
                if isinstance(payload, dict):
                    yield _status(selected)
                else:
                    progress(pct, desc=payload)
                    yield payload, []
        except Exception as exc:
            logger.exception("QC failed")
            yield f"エラー: {exc}", []

    def on_apply(selected):
        if not selected:
            return "エラー: フォルダを選択してください", []
        try:
            result = audio_qc.apply_filter(OUTPUT_DIR / selected)
        except Exception as exc:
            return f"エラー: {exc}", []
        status, rows = _status(selected)
        return (f"{result['excluded']}件を除外し、{result['kept']}件を学習リストに残しました。"
                f"（元のリスト: {result['backup']}）\n{status}"), rows

    def on_restore(selected):
        if not selected:
            return "エラー: フォルダを選択してください", []
        restored = audio_qc.restore_unfiltered(OUTPUT_DIR / selected)
        status, rows = _status(selected)
        return ("除外前のテキストリストに戻しました。\n" if restored else "除外は適用されていません。\n") + status, rows

    refresh.click(fn=lambda: gr.update(choices=list_clone_sources()), outputs=[folder])
    folder.change(fn=_status, inputs=[folder], outputs=[state_text, table])
    qc_event = run.click(fn=on_run, inputs=[folder, use_asr, cer_threshold, asr_model],
                         outputs=[state_text, table])
    apply.click(fn=on_apply, inputs=[folder], outputs=[state_text, table])
    restore.click(fn=on_restore, inputs=[folder], outputs=[state_text, table])
    return qc_event
