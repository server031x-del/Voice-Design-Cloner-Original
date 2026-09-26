"""LoRA tab: fine-tune Irodori-TTS on a Voice Clone output folder."""

import logging
import gradio as gr

from config import OUTPUT_DIR
from lang import t
from modules import audio_qc
from modules.lora_pipeline import (
    LORA_OUTPUT_DIR,
    PRESET_TRAIN_OVERRIDES,
    STEP_PRESETS,
    V4_PRESET_TRAIN_OVERRIDES,
    compare_checkpoints,
    get_preferred_checkpoint,
    list_clone_sources,
    list_lora_checkpoints,
    list_loras,
    run_lora_pipeline,
    set_preferred_checkpoint,
)
from modules.model_manager import ModelManager

logger = logging.getLogger(__name__)


def build_lora_tab(manager: ModelManager, profile: str = "legacy"):
    # LoRA training runs in an Irodori subprocess regardless of which backend
    # vdc itself is using, so this tab stays fully interactive even when the
    # active backend is Qwen3 / faster. The trained adapter only becomes
    # usable in the Voice Clone / Inference tabs after switching to Irodori,
    # which is noted in the manual.
    train_overrides = (
        V4_PRESET_TRAIN_OVERRIDES if profile == "v4" else PRESET_TRAIN_OVERRIDES
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown(t("lora_source_section"))
            with gr.Group():
                source_dropdown = gr.Dropdown(
                    choices=list_clone_sources(),
                    label=t("lora_source_label"),
                )
                source_qc = gr.Textbox(label="学習データの品質チェック", interactive=False, lines=2)
                refresh_source_btn = gr.Button(t("lora_btn_refresh"), variant="secondary")

            gr.HTML("<div style='height: 12px'></div>")

            gr.Markdown(t("lora_settings_section"))
            with gr.Group():
                speaker_name = gr.Textbox(
                    label=t("lora_speaker_label"),
                    placeholder="honoka",
                )
                steps_radio = gr.Radio(
                    choices=[(f"{k} ({v})", k) for k, v in STEP_PRESETS.items()],
                    value="quick",
                    label=t("lora_steps_label"),
                )
                with gr.Accordion(t("lora_advanced_section"), open=False):
                    steps_override = gr.Number(
                        label=t("lora_steps_override_label"),
                        value=STEP_PRESETS["quick"], precision=0,
                    )
                    batch_size = gr.Number(
                        label=t("lora_batch_label"),
                        value=int(train_overrides["quick"]["batch_size"]),
                        precision=0,
                    )
                    num_workers = gr.Number(
                        label=t("lora_workers_label"),
                        value=int(train_overrides["quick"]["num_workers"]),
                        precision=0,
                    )
                    learning_rate = gr.Number(
                        label=t("lora_lr_label"),
                        value=0.0001,
                    )
                    if profile == "v4":
                        training_caption = gr.Textbox(
                            label="V4学習キャプション（任意・全クリップ共通）",
                            placeholder="例: 落ち着いた自然な声。空欄なら話者適応を優先します。",
                            lines=2,
                        )
                    else:
                        training_caption = gr.State(None)
                resume_training = gr.Checkbox(
                    value=False,
                    label="中断した学習を再開（同じ話者名の最新チェックポイントから）",
                )

        with gr.Column(scale=2):
            gr.Markdown(t("lora_existing_section"))
            with gr.Group():
                existing_list = gr.Textbox(
                    value="\n".join(list_loras(profile)) or t("lora_existing_none"),
                    label=("V4 " + t("lora_existing_label") if profile == "v4" else t("lora_existing_label")),
                    interactive=False,
                    lines=5,
                )
                refresh_existing_btn = gr.Button(t("lora_btn_refresh"), variant="secondary")

            gr.HTML("<div style='height: 12px'></div>")

            with gr.Row():
                start_btn = gr.Button(t("lora_btn_start"), variant="primary", scale=3)
                stop_btn = gr.Button(t("lora_btn_stop"), variant="secondary", scale=1)

            gr.Markdown(t("lora_progress_section"))
            with gr.Group():
                status_text = gr.Textbox(label=t("lora_status_label"), interactive=False)
                log_text = gr.Textbox(label=t("lora_log_label"), interactive=False, lines=12)

    # ── refresh handlers ──
    refresh_source_btn.click(
        fn=lambda: gr.update(choices=list_clone_sources()),
        outputs=[source_dropdown],
    )
    source_dropdown.change(
        fn=lambda source: audio_qc.status_text(OUTPUT_DIR / source) if source else "",
        inputs=[source_dropdown],
        outputs=[source_qc],
    )
    refresh_existing_btn.click(
        fn=lambda: "\n".join(list_loras(profile)) or t("lora_existing_none"),
        outputs=[existing_list],
    )

    # ── preset radio updates the advanced overrides to its defaults ──
    def _on_preset_change(preset):
        steps = STEP_PRESETS.get(preset, STEP_PRESETS["quick"])
        overrides = train_overrides.get(preset, train_overrides["quick"])
        return steps, int(overrides["batch_size"]), int(overrides["num_workers"])

    steps_radio.change(
        fn=_on_preset_change,
        inputs=[steps_radio],
        outputs=[steps_override, batch_size, num_workers],
    )

    # ── start training ──
    def on_start(source, speaker, steps, steps_n, batch, workers, lr, caption, resume,
                 progress=gr.Progress()):
        if not source:
            yield t("lora_err_no_source"), ""
            return
        if not speaker or not speaker.strip():
            yield t("lora_err_no_speaker"), ""
            return

        # Explicit step count from the advanced field wins over the preset.
        effective_steps = int(steps_n) if steps_n and int(steps_n) > 0 else steps
        log_buffer: list[str] = []
        try:
            for event in run_lora_pipeline(
                source=source,
                speaker=speaker.strip(),
                steps=effective_steps,
                batch_size=int(batch) if batch else None,
                num_workers=int(workers) if workers is not None else None,
                learning_rate=float(lr) if lr else None,
                profile=profile,
                caption=caption,
                resume=bool(resume),
            ):
                kind = event.get("event")
                if kind == "stage":
                    line = f"[{event['status']}] {event.get('message', '')}"
                    if event["status"] == "gpu_wait":
                        yield event.get("message", ""), "\n".join(log_buffer[-50:])
                        continue
                    log_buffer.append(line)
                    yield line, "\n".join(log_buffer[-50:])
                elif kind == "progress":
                    step = event.get("step", 0)
                    max_s = event.get("max_steps", 0)
                    loss = event.get("loss")
                    line = f"step {step}/{max_s}" + (f"  loss={loss:.4f}" if loss is not None else "")
                    if max_s:
                        progress(min(1.0, step / max_s), desc=line)
                    log_buffer.append(line)
                    yield line, "\n".join(log_buffer[-50:])
                elif kind == "log":
                    raw = event.get("raw", "")
                    log_buffer.append(raw)
                    yield "", "\n".join(log_buffer[-50:])
                elif kind == "done":
                    summary = t("lora_ok_done").format(
                        event.get("speaker"), event.get("output_dir"),
                    )
                    log_buffer.append(summary)
                    yield summary, "\n".join(log_buffer[-50:])
        except Exception as e:
            logger.exception("LoRA pipeline failed")
            log_buffer.append(f"ERROR: {e}")
            yield t("lora_err_failed").format(e), "\n".join(log_buffer[-50:])

    train_event = start_btn.click(
        fn=on_start,
        inputs=[
            source_dropdown, speaker_name, steps_radio,
            steps_override, batch_size, num_workers, learning_rate, training_caption,
            resume_training,
        ],
        outputs=[status_text, log_text],
    )
    stop_btn.click(fn=None, cancels=[train_event])

    _build_checkpoint_compare(profile)

    def _refresh_on_select():
        return (
            gr.update(choices=list_clone_sources()),
            "\n".join(list_loras(profile)) or t("lora_existing_none"),
        )

    return {"refresh": _refresh_on_select, "outputs": [source_dropdown, existing_list]}


_AUTO_CHECKPOINT = "（自動: final優先）"
_COMPARE_DEFAULT_TEXT = "こんにちは、今日はよろしくお願いします。\nえっ、本当に？それはちょっと驚きました。"


def _build_checkpoint_compare(profile: str) -> None:
    """Listen to the same lines rendered by every checkpoint of a LoRA."""
    with gr.Accordion("チェックポイント比較（どのstepが一番良いか聴き比べる）", open=False):
        gr.Markdown(
            "学習中に保存されたチェックポイントごとに、同じ文・同じseed・同じ参照音声で生成します。"
            "過学習（声が硬い・崩れる）していないstepを選び「採用」すると、以後このLoRAを選んだときにそのstepが使われます。"
        )
        with gr.Row():
            speaker = gr.Dropdown(choices=list_loras(profile), label="LoRA", scale=3)
            refresh = gr.Button("更新", variant="secondary", scale=1)
        checkpoints = gr.CheckboxGroup(choices=[], label="比較するチェックポイント（未選択=すべて）")
        texts = gr.Textbox(value=_COMPARE_DEFAULT_TEXT, label="比較用テキスト（1行1文）", lines=3)
        seed = gr.Number(value=1234, precision=0, label="固定seed")
        run = gr.Button("比較音声を生成", variant="primary")
        status = gr.Textbox(label="ステータス", interactive=False, lines=2)
        results = gr.Dataframe(headers=["チェックポイント", "step", "文", "テキスト"], interactive=False,
                               label="生成結果")
        rows_state = gr.State([])
        with gr.Row():
            pick = gr.Dropdown(choices=[], label="試聴する音声", scale=3)
            audio = gr.Audio(label="試聴", type="filepath", scale=3)
        with gr.Row():
            preferred = gr.Dropdown(choices=[], label="このLoRAで使うチェックポイント", scale=3)
            adopt = gr.Button("採用", variant="primary", scale=1)

    def _checkpoint_names(name):
        return [c["name"] for c in list_lora_checkpoints(name, profile)] if name else []

    def on_speaker(name):
        names = _checkpoint_names(name)
        current = get_preferred_checkpoint(name, profile) if name else None
        return (
            gr.update(choices=names, value=[]),
            gr.update(choices=[_AUTO_CHECKPOINT, *names], value=current or _AUTO_CHECKPOINT),
            (f"現在の採用: {current}" if current else "現在の採用: 自動（checkpoint_final 優先）") if name else "",
        )

    def on_run(name, selected, lines, seed_value, progress=gr.Progress()):
        if not name:
            yield "エラー: LoRAを選択してください", [], [], gr.update()
            return
        try:
            for pct, payload in compare_checkpoints(
                speaker=name,
                profile=profile,
                texts=(lines or "").splitlines(),
                seed=int(seed_value) if seed_value is not None else 1234,
                checkpoints=selected or None,
            ):
                if isinstance(payload, list):
                    table = [[r["checkpoint"], r["step"], r["line"], r["text"]] for r in payload]
                    labels = [f"{r['checkpoint']} / 文{r['line']}" for r in payload]
                    yield (
                        f"{len(payload)}本を生成しました。下の「試聴する音声」から選んで聴き比べてください。",
                        table,
                        payload,
                        gr.update(choices=labels, value=labels[0] if labels else None),
                    )
                else:
                    progress(pct, desc=payload)
                    yield payload, [], [], gr.update()
        except Exception as exc:
            logger.exception("Checkpoint comparison failed")
            yield f"エラー: {exc}", [], [], gr.update()

    def on_pick(label, rows):
        for row in rows or []:
            if f"{row['checkpoint']} / 文{row['line']}" == label:
                return row["path"]
        return None

    def on_adopt(name, choice):
        if not name:
            return "エラー: LoRAを選択してください"
        try:
            set_preferred_checkpoint(name, None if choice in (None, _AUTO_CHECKPOINT) else choice, profile)
        except Exception as exc:
            return f"エラー: {exc}"
        return f"「{name}」は今後 {choice} を使用します。"

    refresh.click(fn=lambda: gr.update(choices=list_loras(profile)), outputs=[speaker])
    speaker.change(fn=on_speaker, inputs=[speaker], outputs=[checkpoints, preferred, status])
    run.click(fn=on_run, inputs=[speaker, checkpoints, texts, seed], outputs=[status, results, rows_state, pick])
    pick.change(fn=on_pick, inputs=[pick, rows_state], outputs=[audio])
    adopt.click(fn=on_adopt, inputs=[speaker, preferred], outputs=[status])
