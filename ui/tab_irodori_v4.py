"""Dedicated Irodori-TTS v4 UI.

The legacy v2/v3 tabs remain untouched.  This screen exposes the unified v4
checkpoint's text + reference speech + caption conditioning, long/multiple
references, v4-specific LoRAs, and quantized checkpoint variants.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr
import soundfile as sf

from config import DEFAULT_SAMPLE_TEXT, DEFAULT_TARGET_SR
from modules.emoji_palette import EMOJI_CATEGORIES, append_emoji
from modules.gpu_gate import wait_messages
from modules.irodori_jobs import RELEASE_IDLE, RELEASE_IMMEDIATE, RELEASE_KEEP, format_batch_result
from modules.lora_pipeline import get_lora_adapter_path, list_loras
from modules.model_manager import ModelManager
from modules.utils import format_duration, list_corpus_files, load_corpus
from modules.voice_clone import batch_clone_irodori_v4
from modules.voice_design import (
    generate_irodori_v4,
    get_kept_voice_metadata_by_label,
    get_kept_voice_path_by_label,
    list_kept_voice_labels,
    list_kept_voice_labels_with_metadata,
    save_voice,
)
from ui.qc_panel import build_qc_panel
from ui.tab_gemini_voice import build_gemini_voice_tab
from ui.tab_lora import build_lora_tab

logger = logging.getLogger(__name__)

_LORA_NONE = "—"
_MODEL_CHOICES = [
    ("Full BF16/FP32（最高品質・2.85 GiB）", "full"),
    ("INT8 Weight-only（推奨軽量版・872 MiB）", "int8-weight-only"),
    ("INT8 Dynamic（872 MiB）", "int8-dynamic"),
    ("INT4 Weight-only（Ampere以降・813 MiB）", "int4-weight-only"),
    ("FP8 Weight-only（Ada以降・873 MiB）", "float8-weight-only"),
    ("FP8 Dynamic（Ada以降・906 MiB）", "float8-dynamic"),
]
_RELEASE_CHOICES = [
    ("数分操作がなければ解放（推奨・連続生成が速い）", RELEASE_IDLE),
    ("生成ごとにすぐ解放（VRAM最優先・毎回再読込）", RELEASE_IMMEDIATE),
    ("解放しない（手動で解放）", RELEASE_KEEP),
]


def _resolve_path(value) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("path") or value.get("name")
    return getattr(value, "path", None) or getattr(value, "name", None)


def _resolve_paths(values) -> list[str]:
    if not values:
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]
    paths: list[str] = []
    for value in values:
        path = _resolve_path(value)
        if path and Path(path).is_file():
            paths.append(str(Path(path)))
    return paths


def _combined_refs(saved_label: str | None, uploaded_files) -> list[str]:
    refs: list[str] = []
    if saved_label:
        saved = get_kept_voice_path_by_label(saved_label)
        if saved:
            refs.append(saved)
    for path in _resolve_paths(uploaded_files):
        if path not in refs:
            refs.append(path)
    return refs


def _metadata_summary(metadata: dict | None) -> str:
    if not metadata:
        return ""
    if metadata.get("source") == "gemini":
        return f"Geminiで作成: {metadata.get('voice_id')} / {metadata.get('description') or ''}"
    settings = metadata.get("settings", {})
    caption = settings.get("caption") or "（なし）"
    return (
        f"保存時の設定: seed={metadata.get('used_seed')} / {settings.get('model_variant', '?')} / "
        f"steps={settings.get('num_steps', '?')} / Caption: {caption}"
    )


def _reference_summary(saved_label, uploaded_files) -> str:
    refs = _combined_refs(saved_label, uploaded_files)
    if not refs:
        return "参照なし：CaptionのみのVoice Designとして生成します。"
    total = 0.0
    details: list[str] = []
    for path in refs:
        try:
            info = sf.info(path)
            seconds = info.frames / float(info.samplerate)
            total += seconds
            details.append(f"{Path(path).name} ({seconds:.1f}秒)")
        except Exception:
            details.append(f"{Path(path).name} (長さ不明)")
    warning = ""
    if total > 120.0:
        warning = "\n⚠ 合計120秒を超えた部分は推論時に切り詰められます。"
    meta = _metadata_summary(get_kept_voice_metadata_by_label(saved_label)) if saved_label else ""
    return f"{len(refs)}本 / 合計 {total:.1f}秒\n" + "\n".join(details) + warning + (f"\n{meta}" if meta else "")


def _effective_precision(model_variant: str, precision: str) -> str:
    return "bf16" if model_variant != "full" else precision


def _resolve_seed(seed) -> int | None:
    if seed is None or seed == "":
        return None
    return int(seed)


def _load_batch_texts(corpus_file, uploaded_file, count) -> list[str]:
    uploaded_path = _resolve_path(uploaded_file)
    if uploaded_path:
        with open(uploaded_path, encoding="utf-8") as handle:
            texts = [line.strip() for line in handle if line.strip()]
    elif corpus_file:
        texts = load_corpus(corpus_file, "ja")
    else:
        return []
    limit = int(count or 0)
    return texts[:limit] if limit > 0 else texts


def _release_v4_runtime_status() -> str:
    """Release the worker's active model and format allocator telemetry."""
    from modules.gpu_gate import GPUBusyError, gpu_session
    from modules.irodori_bridge import get_bridge

    try:
        with gpu_session("GPU解放", wait=False):
            bridge = get_bridge()
            response = bridge.release_runtime()
            bridge.shutdown()
    except GPUBusyError as exc:
        return f"解放できません: {exc}"
    memory = response.get("gpu_memory") or {}
    if response.get("runtime_released"):
        return (
            "V4モデル・CUDAキャッシュ・GPUワーカーを完全解放しました。"
            f" PyTorch確保 {float(memory.get('allocated_mib', 0.0)):.1f} MiB /"
            f" 予約 {float(memory.get('reserved_mib', 0.0)):.1f} MiB"
        )
    return "解放対象のV4モデルはありません（GPUワーカー停止済み）。"


def build_irodori_v4_tab(manager: ModelManager):
    gr.Markdown(
        "## Irodori-TTS V4.1 ワークスペース\n"
        "ボイスデザイン・統合生成・一括クローン・LoRA学習を下のタブで選べます。従来モデルは上の「V3系」へ切り替えて使用できます。"
        "V4.1 Small（発話長予測を改善）を使用します。新規LoRA学習もV4.1がベースです。"
    )
    gr.Markdown(
        "> **日本語専用 / 48kHz**　参照音声は同じ話者の短い綺麗なクリップを複数選び、"
        "合計30秒前後にするとV4の長時間参照を活かしやすくなります。"
        "長いテキストは文の区切りで自動分割して生成します（1回の生成上限は約30秒）。"
    )

    with gr.Tabs():
        with gr.Tab("ボイスデザイン"):
            _build_unified_generation(manager, design_only=True)
        with gr.Tab("Geminiボイスデザイン"):
            build_gemini_voice_tab(manager)
        with gr.Tab("統合生成"):
            _build_unified_generation(manager)
        with gr.Tab("一括クローン"):
            _build_batch_generation(manager)
        with gr.Tab("V4 LoRA学習"):
            gr.Markdown(
                "V4 LoRAは `output/lora_v4/` に保存され、既存V3 LoRAとは完全に分離されます。"
            )
            build_lora_tab(manager, profile="v4")
        with gr.Tab("モデル互換性"):
            gr.Markdown(
                """
### モデルの使い分け

- **Full**: 品質基準。RTX 3060ではBF16を推奨します。
- **INT8 Weight-only**: VRAMを抑えたい場合の第一候補です。
- **INT4 Weight-only**: Compute Capability 8.0以上が必要です。RTX 3060は対応、GTX 1660は非対応です。
- **FP8**: Compute Capability 8.9以上が必要なためRTX 3060／GTX 1660では使用できません。

V3用LoRAをV4へ適用することはできません。V4タブは `output/lora_v4/` のみ、既存タブは
`output/lora/` のみを表示します。量子化モデルへ動的に適用するV4 LoRAは、Fullモデルを基盤に学習してください。
"""
            )


def _build_unified_generation(manager: ModelManager, *, design_only: bool = False) -> None:
    state_audio = gr.State(None)
    state_info = gr.State(None)
    if design_only:
        gr.Markdown("### 声の説明から新しい声を作る\n声質・高さ・年齢感・話し方を日本語で入力してください。参照音声なしで生成し、気に入った声は保存してクローンに使えます。")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 1. 参照音声（任意・複数可）", visible=not design_only)
            with gr.Group(visible=not design_only):
                saved_choices = list_kept_voice_labels()
                saved_ref = gr.Dropdown(
                    choices=saved_choices,
                    value=None,
                    label="保存済みVoice Design（任意）",
                )
                uploaded_refs = gr.File(
                    label="追加参照音声（同じ話者・複数選択可）",
                    file_count="multiple",
                    file_types=["audio"],
                    type="filepath",
                )
                ref_summary = gr.Textbox(
                    value="参照なし：CaptionのみのVoice Designとして生成します。",
                    label="参照音声情報",
                    interactive=False,
                    lines=4,
                )
                refresh_refs = gr.Button("保存済み音声を更新", variant="secondary")

            if design_only:
                with gr.Accordion("保存済みボイスの設定を読み込む（再現・微調整用）", open=False):
                    with gr.Row():
                        preset_voice = gr.Dropdown(
                            choices=list_kept_voice_labels_with_metadata(),
                            value=None,
                            label="設定付きで保存されたボイス",
                            scale=3,
                        )
                        preset_refresh = gr.Button("更新", variant="secondary", scale=1)
                    preset_load = gr.Button("この設定を読み込む", variant="secondary")

            gr.Markdown("### 声の説明と試聴テキスト" if design_only else "### 2. Captionと読み上げテキスト")
            caption = gr.Textbox(
                label="作りたい声の説明" if design_only else "Voice Design Caption",
                placeholder="例: 深く傷つき、今にも泣き出しそうな若い女性。声を震わせて弱々しく話す。",
                lines=4,
            )
            text = gr.Textbox(
                label="読み上げテキスト（絵文字制御可・長文は自動分割）",
                value=DEFAULT_SAMPLE_TEXT,
                lines=4,
            )
            emoji_buttons: list[tuple[gr.Button, str]] = []
            with gr.Accordion("絵文字スタイル制御", open=False):
                for category, items in EMOJI_CATEGORIES.items():
                    gr.Markdown(f"**{category}**")
                    with gr.Row():
                        for emoji, _description in items:
                            button = gr.Button(emoji, size="sm", min_width=44)
                            emoji_buttons.append((button, emoji))

        with gr.Column(scale=2):
            gr.Markdown("### 3. V4モデル／推論設定")
            with gr.Group():
                model_variant = gr.Dropdown(
                    choices=_MODEL_CHOICES,
                    value="full",
                    label="V4モデル",
                )
                precision = gr.Dropdown(
                    choices=[("BF16（RTX推奨）", "bf16"), ("FP32", "fp32")],
                    value="bf16",
                    label="Fullモデル精度（量子化版はBF16固定）",
                )
                lora = gr.Dropdown(
                    choices=[_LORA_NONE, *list_loras("v4")],
                    value=_LORA_NONE,
                    label="V4 LoRA",
                    visible=not design_only,
                )
                refresh_lora = gr.Button("V4 LoRAを更新", variant="secondary", size="sm", visible=not design_only)
                release_mode = gr.Radio(
                    choices=_RELEASE_CHOICES,
                    value=RELEASE_IDLE,
                    label="生成後のGPUメモリ",
                )
                release_now = gr.Button("GPUメモリを今すぐ解放", variant="secondary", size="sm")
                with gr.Accordion("詳細パラメータ", open=False):
                    with gr.Row():
                        seed = gr.Number(label="Seed（空でランダム）", value=None, precision=0, scale=3)
                        seed_clear = gr.Button("ランダムに戻す", size="sm", scale=1)
                    num_steps = gr.Slider(4, 80, value=40, step=1, label="RF steps")
                    duration_scale = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Duration scale")
                    max_ref_seconds = gr.Slider(5, 120, value=120, step=5, label="最大参照秒数", visible=not design_only)
                    cfg_text = gr.Slider(0, 10, value=3.0, step=0.25, label="Text CFG")
                    cfg_caption = gr.Slider(0, 10, value=3.0, step=0.25, label="Caption CFG")
                    cfg_speaker = gr.Slider(0, 10, value=5.0, step=0.25, label="Speaker CFG", visible=not design_only)

            gr.Markdown("### 4. 生成・保存")
            audio_preview = gr.Audio(label="V4プレビュー", type="numpy")
            status = gr.Textbox(label="ステータス", interactive=False, lines=3)
            with gr.Row():
                generate = gr.Button("声をデザインして生成" if design_only else "V4で生成", variant="primary", scale=2)
                reroll = gr.Button("再生成（新しいseed）", variant="secondary")
            keep_seed = gr.Button("この音声のseedを固定する", variant="secondary", size="sm")
            save_name = gr.Textbox(label="保存名", value="irodori_v4_voice")
            save_button = gr.Button("Voice Designへ保存（seed・設定も保存）", variant="primary")

    if not design_only:
        saved_ref.change(
            fn=_reference_summary,
            inputs=[saved_ref, uploaded_refs],
            outputs=[ref_summary],
        )
        uploaded_refs.change(
            fn=_reference_summary,
            inputs=[saved_ref, uploaded_refs],
            outputs=[ref_summary],
        )
        refresh_refs.click(
            fn=lambda: gr.update(choices=list_kept_voice_labels(), value=None),
            outputs=[saved_ref],
        )
    refresh_lora.click(
        fn=lambda: gr.update(choices=[_LORA_NONE, *list_loras("v4")], value=_LORA_NONE),
        outputs=[lora],
    )
    model_variant.change(
        fn=lambda variant: gr.update(value="bf16", interactive=(variant == "full")),
        inputs=[model_variant],
        outputs=[precision],
    )
    seed_clear.click(fn=lambda: None, outputs=[seed])

    def _generate(
        saved,
        uploaded,
        caption_text,
        reading_text,
        variant,
        precision_value,
        lora_name,
        seed_value,
        steps,
        duration,
        max_ref,
        text_cfg,
        caption_cfg,
        speaker_cfg,
        release,
    ):
        if not reading_text or not reading_text.strip():
            yield None, None, None, "エラー: 読み上げテキストを入力してください"
            return
        if design_only and not (caption_text or "").strip():
            yield None, None, None, "エラー: 作りたい声の説明を入力してください"
            return
        refs = [] if design_only else _combined_refs(saved, uploaded)
        lora_path = None
        if not design_only and lora_name and lora_name != _LORA_NONE:
            lora_path = get_lora_adapter_path(lora_name, "v4")
        for message in wait_messages("V4生成"):
            yield gr.update(), gr.update(), gr.update(), message
        try:
            manager.unload_model()
            sr, audio, info = generate_irodori_v4(
                text=reading_text,
                caption=caption_text,
                ref_wavs=refs,
                model_variant=variant,
                model_precision=_effective_precision(variant, precision_value),
                lora_path=lora_path,
                seed=seed_value,
                num_steps=int(steps),
                duration_scale=float(duration),
                max_ref_seconds=float(max_ref),
                cfg_scale_text=float(text_cfg),
                cfg_scale_caption=float(caption_cfg),
                cfg_scale_speaker=float(speaker_cfg),
                release_mode=release,
            )
            mode = "Style Clone" if refs else "Voice Design"
            message = (
                f"生成完了: {mode} / {variant} / {sr}Hz / 参照{len(refs)}本 / "
                f"seed={info.get('used_seed')} / {info.get('duration_sec')}秒"
            )
            if info.get("chunks", 1) > 1:
                message += f" / 長文を{info['chunks']}分割して生成"
            if info.get("truncated_suspect"):
                message += "\n⚠ 生成上限（約30秒）に達しています。末尾が切れていないか確認してください。"
            if release == RELEASE_IMMEDIATE:
                message += " / GPUメモリ解放済み"
            yield (sr, audio), (sr, audio), info, message
        except Exception as exc:
            logger.exception("Irodori v4 generation failed")
            yield None, None, None, f"エラー: {exc}"

    def on_generate(saved, uploaded, caption_text, reading_text, variant, precision_value, lora_name,
                    seed_value, *rest):
        yield from _generate(saved, uploaded, caption_text, reading_text, variant, precision_value,
                             lora_name, _resolve_seed(seed_value), *rest)

    def on_reroll(saved, uploaded, caption_text, reading_text, variant, precision_value, lora_name,
                  seed_value, *rest):
        # Re-roll always explores a new voice, even if a seed is pinned.
        yield from _generate(saved, uploaded, caption_text, reading_text, variant, precision_value,
                             lora_name, None, *rest)

    generation_inputs = [
        saved_ref,
        uploaded_refs,
        caption,
        text,
        model_variant,
        precision,
        lora,
        seed,
        num_steps,
        duration_scale,
        max_ref_seconds,
        cfg_text,
        cfg_caption,
        cfg_speaker,
        release_mode,
    ]
    generation_outputs = [audio_preview, state_audio, state_info, status]
    generate.click(fn=on_generate, inputs=generation_inputs, outputs=generation_outputs)
    reroll.click(fn=on_reroll, inputs=generation_inputs, outputs=generation_outputs)
    release_now.click(fn=_release_v4_runtime_status, outputs=[status])

    def on_keep_seed(info):
        if not info or info.get("used_seed") is None:
            return gr.update(), "固定できるseedがありません（先に生成してください）"
        return int(info["used_seed"]), f"seed={info['used_seed']} を固定しました。「V4で生成」で同じ条件を再現できます。"

    keep_seed.click(fn=on_keep_seed, inputs=[state_info], outputs=[seed, status])

    def on_save(audio_data, info, name, reading_text):
        if audio_data is None:
            return "エラー: 保存する音声がありません"
        try:
            destination = save_voice(audio_data, name, sample_text=reading_text or "", metadata=info)
            return (
                f"保存完了: {destination}\n既存／V4両方の参照ショートカットから使用できます。"
                + (f"（seed={info.get('used_seed')} と生成設定を .json に保存）" if info else "")
            )
        except Exception as exc:
            logger.exception("Failed to save v4 voice")
            return f"エラー: {exc}"

    save_button.click(fn=on_save, inputs=[state_audio, state_info, save_name, text], outputs=[status])

    if design_only:
        preset_refresh.click(
            fn=lambda: gr.update(choices=list_kept_voice_labels_with_metadata(), value=None),
            outputs=[preset_voice],
        )

        def on_load_preset(label):
            metadata = get_kept_voice_metadata_by_label(label)
            if not metadata:
                return [gr.update()] * 10 + ["保存済みの設定が見つかりません"]
            s = metadata.get("settings", {})
            variant = s.get("model_variant", "full")
            return [
                s.get("caption") or "",
                metadata.get("text") or gr.update(),
                variant,
                gr.update(value=s.get("model_precision", "bf16"), interactive=(variant == "full")),
                metadata.get("used_seed"),
                s.get("num_steps", 40),
                s.get("duration_scale", 1.0),
                s.get("cfg_scale_text", 3.0),
                s.get("cfg_scale_caption", 3.0),
                label,
                f"「{label}」の設定を読み込みました。{_metadata_summary(metadata)}",
            ]

        preset_load.click(
            fn=on_load_preset,
            inputs=[preset_voice],
            outputs=[caption, text, model_variant, precision, seed, num_steps, duration_scale,
                     cfg_text, cfg_caption, save_name, status],
        )

    for button, emoji in emoji_buttons:
        button.click(
            fn=(lambda current, value=emoji: append_emoji(current, value)),
            inputs=[text],
            outputs=[text],
        )


def _build_batch_generation(manager: ModelManager) -> None:
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 1. 同一話者の参照音声")
            saved_choices = list_kept_voice_labels()
            saved_ref = gr.Dropdown(
                choices=saved_choices,
                value=None,
                label="保存済みVoice Design（任意）",
            )
            refresh_refs = gr.Button("保存済み音声を更新", variant="secondary")
            uploaded_refs = gr.File(
                label="参照音声（複数選択可）",
                file_count="multiple",
                file_types=["audio"],
                type="filepath",
            )
            ref_summary = gr.Textbox(label="参照音声情報", interactive=False, lines=4)

            gr.Markdown("### 2. 日本語コーパス")
            corpus_files = list_corpus_files("ja")
            corpus_file = gr.Dropdown(
                choices=corpus_files,
                value=corpus_files[0] if corpus_files else None,
                label="内蔵コーパス",
            )
            uploaded_text = gr.File(label="または .txt（1行1文）", file_types=[".txt"], type="filepath")
            count = gr.Number(label="生成文数（0=すべて）", value=0, minimum=0, precision=0)
            caption = gr.Textbox(
                label="全クリップ共通のStyle Caption（任意）",
                placeholder="例: 明るく元気に、少し早口で話す。",
                lines=3,
            )

        with gr.Column(scale=2):
            gr.Markdown("### 3. 出力・モデル設定")
            model_variant = gr.Dropdown(choices=_MODEL_CHOICES, value="full", label="V4モデル")
            precision = gr.Dropdown(
                choices=[("BF16（RTX推奨）", "bf16"), ("FP32", "fp32")],
                value="bf16",
                label="Fullモデル精度",
            )
            lora = gr.Dropdown(
                choices=[_LORA_NONE, *list_loras("v4")],
                value=_LORA_NONE,
                label="V4 LoRA",
            )
            release_mode = gr.Radio(
                choices=_RELEASE_CHOICES,
                value=RELEASE_IMMEDIATE,
                label="一括生成終了後のGPUメモリ",
            )
            output_folder = gr.Textbox(label="output/ 以下のフォルダ", value="irodori_v4_clone")
            wavs_folder = gr.Textbox(label="音声サブフォルダ", value="raw")
            esd_filename = gr.Textbox(label="テキストリスト", value="Neutral.txt")
            target_sr = gr.Dropdown(
                choices=[48000, 44100, 24000, 22050],
                value=DEFAULT_TARGET_SR,
                label="出力サンプルレート",
            )
            with gr.Group():
                resume = gr.Checkbox(
                    value=True,
                    label="続きから再開（同じ文・同じ設定で生成済みの行はスキップ）",
                )
                redo_qc = gr.Checkbox(
                    value=False,
                    label="品質チェックNGの行だけ再生成（別のseedで作り直す）",
                )
                continue_on_error = gr.Checkbox(
                    value=True,
                    label="失敗した行はスキップして続行（再試行2回の後）",
                )
            with gr.Accordion("詳細パラメータ", open=False):
                seed = gr.Number(label="開始Seed（空で各文ランダム・使用seedは記録されます）", value=None, precision=0)
                num_steps = gr.Slider(4, 80, value=40, step=1, label="RF steps")
                duration_scale = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Duration scale")
                max_ref_seconds = gr.Slider(5, 120, value=120, step=5, label="最大参照秒数")

            with gr.Row():
                start = gr.Button("V4一括生成開始", variant="primary", scale=3)
                stop = gr.Button("■ 停止", variant="secondary")
            progress_text = gr.Textbox(label="ステータス", interactive=False)
            result_text = gr.Textbox(label="結果", interactive=False, lines=8)

    with gr.Accordion("品質チェック・学習データの選別", open=False):
        build_qc_panel(default_folder="irodori_v4_clone")

    saved_ref.change(fn=_reference_summary, inputs=[saved_ref, uploaded_refs], outputs=[ref_summary])
    uploaded_refs.change(fn=_reference_summary, inputs=[saved_ref, uploaded_refs], outputs=[ref_summary])
    refresh_refs.click(
        fn=lambda: gr.update(choices=list_kept_voice_labels(), value=None),
        outputs=[saved_ref],
    )
    model_variant.change(
        fn=lambda variant: gr.update(value="bf16", interactive=(variant == "full")),
        inputs=[model_variant],
        outputs=[precision],
    )

    def on_batch(
        saved,
        uploads,
        selected_corpus,
        text_file,
        limit,
        caption_text,
        variant,
        precision_value,
        lora_name,
        folder,
        wav_folder,
        esd_name,
        sr,
        seed_value,
        steps,
        duration,
        max_ref,
        release,
        do_resume,
        do_redo_qc,
        keep_going,
        progress=gr.Progress(),
    ):
        refs = _combined_refs(saved, uploads)
        if not refs:
            yield "エラー: V4一括クローンには参照音声が必要です", ""
            return
        try:
            texts = _load_batch_texts(selected_corpus, text_file, limit)
        except Exception as exc:
            yield f"エラー: コーパスを読み込めません ({exc})", ""
            return
        if not texts:
            yield "エラー: 生成するテキストがありません", ""
            return

        redo_ids = None
        if do_redo_qc:
            from config import OUTPUT_DIR
            from modules import audio_qc
            from modules.dataset_io import sanitize_segment

            redo_ids = audio_qc.rejected_ids(OUTPUT_DIR / sanitize_segment(folder or "irodori_v4_clone", "irodori_v4_clone"))
            if not redo_ids:
                yield "品質チェックでNGの行はありません（先に品質チェックを実行してください）", ""
                return

        lora_path = None
        if lora_name and lora_name != _LORA_NONE:
            lora_path = get_lora_adapter_path(lora_name, "v4")
        try:
            manager.unload_model()
            for pct, payload in batch_clone_irodori_v4(
                ref_audios=refs,
                texts=texts,
                caption=caption_text,
                output_folder=(folder or "irodori_v4_clone"),
                wavs_folder=(wav_folder or "raw"),
                esd_filename=(esd_name or "Neutral.txt"),
                target_sr=int(sr),
                model_variant=variant,
                model_precision=_effective_precision(variant, precision_value),
                lora_path=lora_path,
                seed=_resolve_seed(seed_value),
                num_steps=int(steps),
                duration_scale=float(duration),
                max_ref_seconds=float(max_ref),
                release_mode=release,
                resume=bool(do_resume) or bool(redo_ids),
                redo_ids=redo_ids,
                continue_on_error=bool(keep_going),
            ):
                if isinstance(payload, dict):
                    yield "完了", format_batch_result(payload, format_duration)
                else:
                    progress(pct, desc=payload)
                    yield payload, ""
        except Exception as exc:
            logger.exception("Irodori v4 batch clone failed")
            yield f"エラー: {exc}（生成済みの行は保存済みです。再実行すると続きから再開します）", ""

    batch_event = start.click(
        fn=on_batch,
        inputs=[
            saved_ref,
            uploaded_refs,
            corpus_file,
            uploaded_text,
            count,
            caption,
            model_variant,
            precision,
            lora,
            output_folder,
            wavs_folder,
            esd_filename,
            target_sr,
            seed,
            num_steps,
            duration_scale,
            max_ref_seconds,
            release_mode,
            resume,
            redo_qc,
            continue_on_error,
        ],
        outputs=[progress_text, result_text],
    )
    stop.click(fn=None, cancels=[batch_event])
