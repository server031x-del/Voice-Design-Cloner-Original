"""Gemini voice design → Irodori-TTS reference hand-off UI."""

from __future__ import annotations

import logging

import gradio as gr

from modules import gemini_tts
from modules.gemini_tts import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_REFERENCE_LINES,
    MAX_STORED_VOICES,
    MODELS,
    GeminiTTSClient,
    build_irodori_reference,
    wav_bytes_to_array,
)
from modules.irodori_jobs import format_batch_result
from modules.model_manager import ModelManager
from modules.utils import AITUBER_CORPUS_PRESET, format_duration, load_corpora
from modules.voice_clone import batch_clone_irodori_v4
from ui.corpus_selector import build_corpus_selector

logger = logging.getLogger(__name__)

_GENDER_CHOICES = [("女性", "female"), ("男性", "male")]


def _key_status() -> str:
    source = gemini_tts.api_key_source()
    return f"APIキー: 設定済み（{source}）" if source else "APIキー: 未設定（下に入力するか、環境変数 GEMINI_API_KEY を設定してください）"


def _voice_label(voice: dict) -> str:
    name = voice.get("display_name") or "(無名)"
    return f"{name}  [{voice.get('id')}]"


def build_gemini_voice_tab(manager: ModelManager) -> None:
    session_key = gr.State("")
    gr.Markdown(
        "### Geminiで声をデザインして、Irodori-TTSで複製する\n"
        "1. Gemini TTS（クラウド）で声を設計 → 2. その声で数文を読み上げてリファレンス（約30秒）を作成 → "
        "3. 「保存済みVoice Design」としてIrodori V4の統合生成・一括クローンでそのまま使えます。\n\n"
        "Geminiの出力は **24kHz** です（Irodoriは48kHz）。参照音声としては問題なく使えますが、高域の情報は含まれません。"
    )
    with gr.Accordion("Gemini APIキー", open=not bool(gemini_tts.get_api_key())):
        key_status = gr.Markdown(_key_status())
        with gr.Row():
            key_input = gr.Textbox(label="APIキー", type="password", placeholder="AIza...", scale=3)
            key_save = gr.Checkbox(value=False, label="config.jsonに保存（git管理外）", scale=1)
        key_apply = gr.Button("このキーを使う", variant="secondary")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("#### 1. 声の設計（Gemini）")
            description = gr.Textbox(
                label="声の説明（1〜2文が推奨・年齢／性別／声質／話し方など変わらない特徴）",
                placeholder="例: 20代前半の落ち着いた女性。少し低めで柔らかく、夜のラジオのように静かで親密に話す。",
                lines=3,
            )
            with gr.Row():
                gender = gr.Radio(choices=_GENDER_CHOICES, value="female", label="性別")
                language = gr.Textbox(value=DEFAULT_LANGUAGE, label="言語コード")
            with gr.Row():
                display_name = gr.Textbox(value="vdc_voice", label="表示名（Gemini側）")
                model = gr.Dropdown(choices=MODELS, value=DEFAULT_MODEL, label="モデル")
            create = gr.Button("Geminiで声を作る", variant="primary")
            gr.Markdown(
                f"作成した声はGeminiのプロジェクトに保存されます（最大{MAX_STORED_VOICES}件・1年で失効）。"
                "不要な声は右の一覧から削除できます。"
            )
            sample_audio = gr.Audio(label="Geminiのサンプル音声", type="numpy")

            gr.Markdown("#### 2. 試聴（任意）")
            trial_text = gr.Textbox(value="こんにちは、今日はよろしくお願いします。", label="試聴テキスト", lines=2)
            trial_style = gr.Textbox(label="話し方（任意・例: 明るく元気に）")
            trial = gr.Button("この声で読み上げ", variant="secondary")
            trial_audio = gr.Audio(label="Gemini 試聴", type="numpy")

        with gr.Column(scale=2):
            gr.Markdown("#### Geminiに保存された声")
            voice_select = gr.Dropdown(choices=[], label="声を選択", allow_custom_value=True)
            with gr.Row():
                refresh = gr.Button("一覧を更新", variant="secondary")
                delete = gr.Button("選択した声を削除", variant="stop")
            voice_id = gr.Textbox(label="使用するvoice_id", interactive=True)

            gr.Markdown("#### 3. Irodori用リファレンスを作る")
            ref_name = gr.Textbox(value="gemini_voice", label="保存名（output/voice_design/）")
            ref_lines = gr.Textbox(
                value="\n".join(DEFAULT_REFERENCE_LINES),
                label="リファレンス用テキスト（1行1文・合計30秒前後が目安）",
                lines=8,
            )
            ref_style = gr.Textbox(label="話し方（任意・空欄推奨: 素の声を参照にする）")
            build = gr.Button("リファレンスを作成して保存", variant="primary")
            reference_audio = gr.Audio(label="作成したリファレンス", type="numpy")

            gr.Markdown("#### 4. Irodori V4で複製を試す")
            irodori_text = gr.Textbox(value="はじめまして。この声、ちゃんと再現できているかな？", label="Irodoriで読ませるテキスト", lines=2)
            irodori_test = gr.Button("Irodori V4で生成", variant="secondary")
            irodori_audio = gr.Audio(label="Irodori V4（複製）", type="numpy")

    with gr.Accordion("5. LoRA用の学習データを作る（Irodori V4で一括複製）", open=False):
        gr.Markdown(
            "作成したリファレンスを参照にIrodori V4でコーパスを一括生成し、そのままLoRA学習に使えるフォルダを作ります"
            "（48kHzのIrodori出力で学習するため、24kHzのGemini音声を直接学習するより音質面で有利です）。\n"
            "完了後は **品質チェック → V4 LoRA学習タブ** で、このフォルダを選んで学習してください。途中で止めても続きから再開できます。\n\n"
            "**AITuber向けの既定セット**: 感情100（感情の幅）→ 配信200（挨拶・コメント反応・ゲーム実況・スパチャ感謝など）"
            "→ AIキャラ500（AICAコーパス・会話）＝800文。生成は約1時間、LoRAは `normal`（10000 step）以上を推奨します。"
        )
        lora_corpus, _lora_order = build_corpus_selector(default=AITUBER_CORPUS_PRESET)
        lora_count = gr.Number(value=0, precision=0, minimum=0, label="文数（0=すべて）")
        with gr.Row():
            lora_folder = gr.Textbox(value="gemini_voice_lora", label="出力フォルダ（output/ 以下）")
            lora_model = gr.Dropdown(
                choices=[("Full（推奨）", "full"), ("INT8 Weight-only", "int8-weight-only")],
                value="full",
                label="V4モデル",
            )
        lora_start = gr.Button("学習データを一括生成", variant="primary")
        lora_stop = gr.Button("■ 停止", variant="secondary")
        lora_result = gr.Textbox(label="結果", interactive=False, lines=6)

    status = gr.Textbox(label="ステータス", interactive=False, lines=3)
    reference_path = gr.State(None)
    gr.Markdown(
        "> **規約メモ**: 個人利用の範囲では問題になりにくいですが、Gemini API追加利用規約は「競合するモデルの開発」への使用を禁じています。"
        "Gemini由来の声で学習したLoRAや学習データを**配布・公開**する場合は規約を確認してください。"
        "実在の人物の声を模倣する説明は使わないでください。"
    )

    def _client(key: str) -> GeminiTTSClient:
        return GeminiTTSClient(api_key=key or None)

    def on_apply_key(key, save):
        key = (key or "").strip()
        if not key:
            return "", _key_status() + "\n\nエラー: キーが空です", ""
        if save:
            gemini_tts.save_api_key(key)
        return key, _key_status() if save else "APIキー: このセッションでのみ使用中", ""

    key_apply.click(fn=on_apply_key, inputs=[key_input, key_save], outputs=[session_key, key_status, key_input])

    def on_refresh(key):
        try:
            voices = _client(key).list_voices()
        except Exception as exc:
            return gr.update(), f"エラー: {exc}"
        choices = [(_voice_label(v), v.get("id")) for v in voices if v.get("id")]
        return gr.update(choices=choices), f"Geminiに保存された声: {len(choices)}件 / 上限{MAX_STORED_VOICES}件"

    refresh.click(fn=on_refresh, inputs=[session_key], outputs=[voice_select, status])

    def on_select(key, selected):
        if not selected:
            return gr.update(), None, ""
        try:
            details = _client(key).get_voice(selected)
        except Exception as exc:
            return selected, None, f"エラー: {exc}"
        sample = details.get("sample_wav")
        prompt = (details.get("prompted") or {}).get("input", "")
        return selected, (wav_bytes_to_array(sample) if sample else None), f"選択: {selected}\n説明: {prompt}"

    voice_select.change(fn=on_select, inputs=[session_key, voice_select], outputs=[voice_id, sample_audio, status])

    def on_create(key, desc, gen, lang, name, model_name):
        try:
            created = _client(key).create_voice(
                desc, display_name=name, gender=gen, language_code=(lang or DEFAULT_LANGUAGE).strip(),
                model=model_name,
            )
        except Exception as exc:
            logger.exception("Gemini voice creation failed")
            return gr.update(), None, f"エラー: {exc}"
        sample = created.get("sample_wav")
        return (
            created["id"],
            wav_bytes_to_array(sample) if sample else None,
            f"作成しました: {created['id']}（{created['display_name']}）\n"
            "気に入らなければもう一度「声を作る」。気に入ったら右の「リファレンスを作成して保存」へ。",
        )

    create.click(fn=on_create, inputs=[session_key, description, gender, language, display_name, model],
                 outputs=[voice_id, sample_audio, status])

    def on_delete(key, selected, current_id):
        target = selected or current_id
        if not target:
            return gr.update(), "エラー: 削除する声を選択してください"
        try:
            client = _client(key)
            client.delete_voice(target)
            voices = client.list_voices()
        except Exception as exc:
            return gr.update(), f"エラー: {exc}"
        choices = [(_voice_label(v), v.get("id")) for v in voices if v.get("id")]
        return gr.update(choices=choices, value=None), f"削除しました: {target}"

    delete.click(fn=on_delete, inputs=[session_key, voice_select, voice_id], outputs=[voice_select, status])

    def on_trial(key, vid, text, style, model_name):
        if not vid:
            return None, "エラー: 先に声を作成または選択してください"
        try:
            wav = _client(key).synthesize(text, voice=vid.strip(), model=model_name, style=style)
        except Exception as exc:
            return None, f"エラー: {exc}"
        return wav_bytes_to_array(wav), "試聴を生成しました"

    trial.click(fn=on_trial, inputs=[session_key, voice_id, trial_text, trial_style, model],
                outputs=[trial_audio, status])

    def on_build(key, vid, name, lines, style, model_name, desc, progress=gr.Progress()):
        if not vid:
            yield None, None, "エラー: 先に声を作成または選択してください"
            return
        try:
            for pct, payload in build_irodori_reference(
                _client(key), voice_id=vid.strip(), name=name, lines=(lines or "").splitlines(),
                model=model_name, style=style, description=desc,
            ):
                if isinstance(payload, dict):
                    import soundfile as sf

                    audio, sr = sf.read(payload["reference_path"], dtype="float32")
                    yield (sr, audio), payload["reference_path"], (
                        f"保存しました: {payload['reference_path']}（{payload['duration_sec']}秒・{payload['clips']}文）\n"
                        "統合生成／一括クローンの「保存済みVoice Design」に表示されます（一覧の「更新」を押してください）。"
                    )
                else:
                    progress(pct, desc=payload)
                    yield gr.update(), gr.update(), payload
        except Exception as exc:
            logger.exception("Gemini reference build failed")
            yield None, None, f"エラー: {exc}"

    build.click(fn=on_build, inputs=[session_key, voice_id, ref_name, ref_lines, ref_style, model, description],
                outputs=[reference_audio, reference_path, status])

    def on_irodori(ref, text):
        if not ref:
            return None, "エラー: 先に「リファレンスを作成して保存」を実行してください"
        try:
            from modules.voice_design import generate_irodori_v4

            manager.unload_model()
            sr, audio, info = generate_irodori_v4(text=text, ref_wavs=[ref], release_mode="idle")
        except Exception as exc:
            logger.exception("Irodori clone of Gemini voice failed")
            return None, f"エラー: {exc}"
        return (sr, audio), f"Irodori V4で生成しました（seed={info.get('used_seed')}）。Geminiのサンプルと聴き比べてください。"

    irodori_test.click(fn=on_irodori, inputs=[reference_path, irodori_text], outputs=[irodori_audio, status])

    def _reference_for(ref, name):
        """Reference built in this session, or a previously saved one by name."""
        if ref:
            return ref
        from modules.dataset_io import sanitize_segment
        from modules.voice_design import get_kept_voice_path_by_label

        return get_kept_voice_path_by_label(sanitize_segment(name or "", "gemini_voice"))

    def on_build_dataset(ref, name, corpus, count, folder, variant, progress=gr.Progress()):
        reference = _reference_for(ref, name)
        if not reference:
            yield "エラー: 先に「リファレンスを作成して保存」を実行するか、保存名に既存のリファレンス名を入力してください"
            return
        try:
            texts = load_corpora(corpus, "ja") if corpus else []
        except Exception as exc:
            yield f"エラー: コーパスを読み込めません ({exc})"
            return
        limit = int(count or 0)
        texts = texts[:limit] if limit > 0 else texts
        if not texts:
            yield "エラー: 生成するテキストがありません"
            return
        try:
            manager.unload_model()
            for pct, payload in batch_clone_irodori_v4(
                ref_audios=[reference],
                texts=texts,
                output_folder=folder or "gemini_voice_lora",
                target_sr=48000,
                model_variant=variant,
                release_mode="immediate",
            ):
                if isinstance(payload, dict):
                    yield (
                        format_batch_result(payload, format_duration)
                        + "\n\n次の手順: 一括クローンタブの「品質チェック」→ V4 LoRA学習タブで"
                        f"「{folder or 'gemini_voice_lora'}」を選んで学習"
                    )
                else:
                    progress(pct, desc=payload)
                    yield payload
        except Exception as exc:
            logger.exception("Gemini → Irodori dataset build failed")
            yield f"エラー: {exc}（生成済みの行は保存済みです。再実行すると続きから再開します）"

    dataset_event = lora_start.click(
        fn=on_build_dataset,
        inputs=[reference_path, ref_name, lora_corpus, lora_count, lora_folder, lora_model],
        outputs=[lora_result],
    )
    lora_stop.click(fn=None, cancels=[dataset_event])
    ref_name.change(fn=lambda name: f"{name}_lora" if name else "gemini_voice_lora",
                    inputs=[ref_name], outputs=[lora_folder])
