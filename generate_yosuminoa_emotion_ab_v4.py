"""Generate a controlled A/B emotion test for the Yosumi Noa V4 voice."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from generate_yosuminoa_downer_v4 import LORA_PATH, REFERENCE_FILES, atomic_write, audio_stats, sha256
from modules.irodori_bridge import get_bridge
from modules.gpu_gate import gpu_session


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "irodori_v4_emotion_ab_yosuminoa_24"
WAV_DIR = OUT_DIR / "wav"
TEXT_DIR = OUT_DIR / "text"
MANIFEST_PATH = OUT_DIR / "manifest.json"
ZIP_PATH = ROOT / "output" / "YosumiNoa_Emotion_AB_IrodoriV4_24.zip"

BASE_IDENTITY = (
    "日本語を話す若い女性。夜と月明かりが似合う、少しダウナーなAIシンガー。"
    "同じキャラクターの声を保ち、発音は明瞭にする。"
)

EMOTIONS = [
    {
        "id": 1,
        "slug": "joy",
        "label": "抑えきれない喜び",
        "text": "えっ、本当に来てくれたの？ うれしい……！",
        "styled_text": "えっ、本当に来てくれたの？😆 うれしい……！",
        "emoji": "😆",
        "caption": "嬉しさを抑えきれず、声が少し明るく弾む。ダウナーな質感は残しつつ、自然な笑顔と柔らかな抑揚。",
    },
    {
        "id": 2,
        "slug": "sadness",
        "label": "静かな寂しさ",
        "text": "……もう、行っちゃうんだね。少しだけ寂しい。",
        "styled_text": "……もう、行っちゃうんだね。😢 少しだけ寂しい。",
        "emoji": "😢",
        "caption": "寂しさをこらえ、語尾がわずかに揺れる。泣き崩れず、低く静かに悲しさをにじませる。",
    },
    {
        "id": 3,
        "slug": "anger",
        "label": "静かな怒り",
        "text": "それ以上、私の大切なものに触らないで。",
        "styled_text": "それ以上、私の大切なものに触らないで。😠",
        "emoji": "😠",
        "caption": "叫ばない静かな怒り。低く鋭い発音で不機嫌さと強い意志を伝え、冷静さの中に芯を持つ。",
    },
    {
        "id": 4,
        "slug": "relief",
        "label": "深い安堵",
        "text": "よかった……無事だったんだね。",
        "styled_text": "よかった……😌 無事だったんだね。",
        "emoji": "😌",
        "caption": "大きく息を吐いたあとの深い安堵。声が柔らかくほどけ、相手を包む温かさが自然に出る。",
    },
    {
        "id": 5,
        "slug": "anxiety",
        "label": "不安と緊張",
        "text": "待って、今の音……何か聞こえなかった？",
        "styled_text": "待って、今の音……😰 何か聞こえなかった？",
        "emoji": "😰",
        "caption": "不安と緊張で息が少し浅くなり、普段よりわずかに急ぐ。怯えすぎず、夜の気配を警戒する。",
    },
    {
        "id": 6,
        "slug": "determination",
        "label": "静かな決意",
        "text": "大丈夫。今度は私が、ちゃんと守るから。",
        "styled_text": "大丈夫。😤 今度は私が、ちゃんと守るから。",
        "emoji": "😤",
        "caption": "迷いを断ち切った静かな決意。低い声に芯と力を込め、騒がしくならず頼もしさを伝える。",
    },
]

CONDITIONS = [
    {
        "id": "base_design",
        "label": "V4ベース・Voice Design",
        "short": "ベースのみ",
        "mode": "design",
        "use_lora": False,
        "use_refs": False,
        "use_emoji": False,
        "cfg_scale_speaker": 0.0,
        "note": "LoRAなし・参照音声なし。V4本体のCaptionによる感情表現を確認。",
    },
    {
        "id": "base_ref",
        "label": "V4ベース＋参照音声",
        "short": "ベース＋参照",
        "mode": "clone",
        "use_lora": False,
        "use_refs": True,
        "use_emoji": False,
        "cfg_scale_speaker": 3.0,
        "note": "LoRAなし・ニュートラル参照6本。話者誘導を弱めて声質と感情の両立を確認。",
    },
    {
        "id": "lora_current",
        "label": "現在のLoRA構成",
        "short": "LoRA＋参照",
        "mode": "clone",
        "use_lora": True,
        "use_refs": True,
        "use_emoji": False,
        "cfg_scale_speaker": 5.0,
        "note": "前回20種と同じ。ニュートラルLoRA＋ニュートラル参照＋Caption。",
    },
    {
        "id": "lora_emoji_soft",
        "label": "LoRA＋絵文字＋弱い話者誘導",
        "short": "LoRA＋絵文字",
        "mode": "clone",
        "use_lora": True,
        "use_refs": True,
        "use_emoji": True,
        "cfg_scale_speaker": 3.0,
        "note": "LoRAと参照は維持し、本文に絵文字を追加。話者CFGを5.0から3.0へ下げる。",
    },
]


def make_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    record_id = 1
    for emotion in EMOTIONS:
        for condition in CONDITIONS:
            stem = f"{emotion['id']:02d}_{condition['id']}_{emotion['slug']}"
            records.append(
                {
                    "id": record_id,
                    "emotion_id": emotion["id"],
                    "emotion_slug": emotion["slug"],
                    "emotion_label": emotion["label"],
                    "condition_id": condition["id"],
                    "condition_label": condition["label"],
                    "condition_short": condition["short"],
                    "condition_note": condition["note"],
                    "emoji": emotion["emoji"],
                    "text": emotion["styled_text"] if condition["use_emoji"] else emotion["text"],
                    "plain_text": emotion["text"],
                    "caption": f"{BASE_IDENTITY}{emotion['caption']}",
                    "mode": condition["mode"],
                    "use_lora": condition["use_lora"],
                    "use_refs": condition["use_refs"],
                    "use_emoji": condition["use_emoji"],
                    "cfg_scale_text": 3.0,
                    "cfg_scale_caption": 3.5,
                    "cfg_scale_speaker": condition["cfg_scale_speaker"],
                    "seed": 93000 + int(emotion["id"]),
                    "wav": f"wav/{stem}.wav",
                    "text_file": f"text/{stem}.txt",
                }
            )
            record_id += 1
    return records


def load_existing_records() -> dict[int, dict[str, object]]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        int(record["id"]): record
        for record in payload.get("records", [])
        if isinstance(record, dict) and record.get("id") is not None
    }


def write_manifest(records: list[dict[str, object]], started_at: str, finished_at: str | None = None) -> None:
    payload = {
        "title": "夜澄ノア向け Irodori-TTS V4 感情表現 A/B 比較 24本",
        "created_at": started_at,
        "finished_at": finished_at,
        "purpose": "同じ6感情文を4条件で比較し、V4本体・参照音声・LoRA・絵文字制御の影響を切り分ける。",
        "model": {
            "profile": "v4",
            "model_variant": "full",
            "model_precision": "bf16",
            "target_sample_rate": 48000,
            "num_steps": 40,
            "cfg_scale_text": 3.0,
            "cfg_scale_caption": 3.5,
            "lora": str(LORA_PATH),
            "references": [str(path) for path in REFERENCE_FILES],
        },
        "conditions": CONDITIONS,
        "emotions": EMOTIONS,
        "records": records,
    }
    atomic_write(MANIFEST_PATH, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_text_files(records: list[dict[str, object]]) -> None:
    for record in records:
        content = (
            f"A/B比較 No. {int(record['id']):02d}\n"
            f"感情: {record['emotion_label']}\n"
            f"条件: {record['condition_label']}\n"
            f"シード: {int(record['seed'])}\n"
            f"話者CFG: {float(record['cfg_scale_speaker']):.1f}\n"
            f"絵文字制御: {'あり' if record['use_emoji'] else 'なし'}\n\n"
            f"読み上げテキスト:\n{record['text']}\n\n"
            f"Caption:\n{record['caption']}\n\n"
            f"条件メモ:\n{record['condition_note']}\n"
        )
        atomic_write(OUT_DIR / str(record["text_file"]), content)


def combine_audio(records: list[dict[str, object]]) -> Path:
    path = OUT_DIR / "all_emotion_ab_075s_gap.wav"
    temp = path.with_suffix(".wav.tmp")
    gap = np.zeros(int(48000 * 0.75), dtype=np.float32)
    with sf.SoundFile(
        str(temp),
        mode="w",
        samplerate=48000,
        channels=1,
        subtype="PCM_16",
        format="WAV",
    ) as output:
        for record in records:
            with sf.SoundFile(str(OUT_DIR / str(record["wav"])), mode="r") as source:
                for block in source.blocks(blocksize=65536, dtype="float32", always_2d=False):
                    output.write(block)
            output.write(gap)
    os.replace(temp, path)
    return path


def write_playlist(records: list[dict[str, object]]) -> None:
    lines = ["#EXTM3U", "# 夜澄ノア向け Irodori-TTS V4 感情表現 A/B 比較"]
    for record in records:
        lines.extend([f"# {int(record['id']):02d} {record['emotion_label']} / {record['condition_short']}", str(record["wav"])])
    atomic_write(OUT_DIR / "playlist.m3u", "\n".join(lines) + "\n")


def write_readme(combined: Path) -> None:
    lines = [
        "# 夜澄ノア向け Irodori-TTS V4 感情表現 A/B 比較 24本",
        "",
        "同じ6つの感情文を4条件で生成し、声質そのものと、参照音声・LoRA・絵文字制御の影響を比較します。",
        "",
        "- `index.html`: 感情ごとに4条件を横並びで試聴できます。",
        f"- `{combined.name}`: 24本を各0.75秒間隔で連結した比較用音声です。",
        "- `playlist.m3u`: 対応プレーヤー用プレイリストです。",
        "- `manifest.json`: 生成条件、Caption、音声統計です。",
        "",
        "## 条件",
        "",
        "| 条件 | 内容 |",
        "|---|---|",
    ]
    for condition in CONDITIONS:
        lines.append(f"| {condition['short']} | {condition['note']} |")
    lines.extend(
        [
            "",
            "## 判定の目安",
            "",
            "- ベースのみが最も感情豊かなら、ニュートラルLoRAまたは参照音声が抑揚を抑えています。",
            "- ベース＋参照で平坦化するなら、ニュートラル参照の話者誘導が主因です。",
            "- 絵文字条件だけ改善するなら、本文内スタイル制御が有効です。",
            "- 4条件すべて平坦なら、Captionを短く強くするか、感情データ入りLoRAを検討します。",
        ]
    )
    atomic_write(OUT_DIR / "README.md", "\n".join(lines) + "\n")


def write_html(records: list[dict[str, object]]) -> None:
    groups: dict[int, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(int(record["emotion_id"]), []).append(record)
    css = "body{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;background:#12131a;color:#f1f3f8}.lead{color:#bdc5d4}.emotion{margin:2rem 0}.emotion h2{border-bottom:1px solid #3b4354;padding-bottom:.4rem}.utterance{color:#c6cede}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:1rem}.card{background:#1c2130;border:1px solid #344057;border-radius:12px;padding:1rem}.card h3{font-size:1rem;margin:.1rem 0 .3rem}.meta{color:#9eaac0;font-size:.82rem}audio{width:100%;margin:.6rem 0}details{color:#cbd3df;font-size:.88rem}summary{cursor:pointer}"
    parts = [
        "<!doctype html>",
        "<html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>夜澄ノア — Irodori V4 Emotion A/B 24</title>",
        f"<style>{css}</style></head><body>",
        "<h1>夜澄ノア向け Irodori-TTS V4 感情表現 A/B 比較 24本</h1>",
        "<p class='lead'>同じ感情文を4条件で聴き比べて、V4本体・参照音声・LoRA・絵文字制御の影響を確認できます。</p>",
    ]
    for emotion in EMOTIONS:
        cards = []
        for record in groups[int(emotion["id"])]:
            cards.append(
                "<article class='card'>"
                f"<h3>{html.escape(str(record['condition_short']))}</h3>"
                f"<p class='meta'>No.{int(record['id']):02d} / seed {int(record['seed'])} / speaker CFG {float(record['cfg_scale_speaker']):.1f}</p>"
                f"<audio controls preload='none' src='{html.escape(str(record['wav']))}'></audio>"
                f"<details><summary>本文とCaption</summary><p><b>本文:</b> {html.escape(str(record['text']))}</p><p><b>Caption:</b> {html.escape(str(record['caption']))}</p></details>"
                "</article>"
            )
        parts.extend(
            [
                "<section class='emotion'>",
                f"<h2>{int(emotion['id']):02d} — {html.escape(str(emotion['label']))}</h2>",
                f"<p class='utterance'>{html.escape(str(emotion['text']))}</p>",
                f"<div class='grid'>{''.join(cards)}</div>",
                "</section>",
            ]
        )
    parts.append("</body></html>")
    atomic_write(OUT_DIR / "index.html", "\n".join(parts) + "\n")


def package_output() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    output_root = ROOT / "output"
    with zipfile.ZipFile(str(ZIP_PATH), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root))


def run(limit: int | None) -> None:
    if len(EMOTIONS) != 6 or len(CONDITIONS) != 4:
        raise RuntimeError("The controlled test must contain 6 emotions and 4 conditions")
    missing = [str(path) for path in [LORA_PATH, *REFERENCE_FILES] if not path.exists()]
    if missing:
        raise FileNotFoundError("V4 generation inputs are missing:\n" + "\n".join(missing))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    records = make_records()
    existing = load_existing_records()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    write_text_files(records)
    write_manifest(records, started_at)
    total = min(limit, len(records)) if limit is not None else len(records)
    bridge = get_bridge()
    started = time.monotonic()
    generated = 0
    with gpu_session():
        bridge.ensure_started()
        try:
            for position, record in enumerate(records[:total], start=1):
                wav_path = OUT_DIR / str(record["wav"])
                if wav_path.exists() and wav_path.stat().st_size > 1000:
                    old = existing.get(int(record["id"]), {})
                    record.update({"status": "generated", "audio": audio_stats(wav_path)})
                    if old.get("used_seed") is not None:
                        record["used_seed"] = old["used_seed"]
                    print(f"[{position:02d}/{total:02d}] 再利用: {record['emotion_label']} / {record['condition_short']}", flush=True)
                    continue

                part_path = wav_path.with_name("." + wav_path.name + ".part.wav")
                if part_path.exists():
                    part_path.unlink()
                print(f"[{position:02d}/{total:02d}] 生成中: {record['emotion_label']} / {record['condition_short']}", flush=True)
                response = bridge.synthesize(
                    profile="v4",
                    mode=str(record["mode"]),
                    model_variant="full",
                    model_precision="bf16",
                    text=str(record["text"]),
                    caption=str(record["caption"]),
                    ref_wavs=[str(path) for path in REFERENCE_FILES] if record["use_refs"] else None,
                    no_ref=not bool(record["use_refs"]),
                    out_path=part_path,
                    target_sr=48000,
                    lora_path=LORA_PATH if record["use_lora"] else None,
                    seed=int(record["seed"]),
                    num_steps=40,
                    duration_scale=1.0,
                    max_ref_seconds=120.0,
                    cfg_scale_text=3.0,
                    cfg_scale_caption=3.5,
                    cfg_scale_speaker=float(record["cfg_scale_speaker"]),
                    release_after_synthesis=False,
                )
                if not part_path.exists() or part_path.stat().st_size <= 1000:
                    raise RuntimeError(f"V4 worker reported success but did not create {part_path}")
                os.replace(part_path, wav_path)
                record.update({"status": "generated", "used_seed": response.get("used_seed"), "audio": audio_stats(wav_path)})
                generated += 1
                elapsed = time.monotonic() - started
                remaining = (elapsed / position) * (total - position)
                print(
                    f"[{position:02d}/{total:02d}] 完了: {record['emotion_label']} / {record['condition_short']} / "
                    f"{record['audio']['duration_sec']}s / 残り約{remaining / 60:.1f}分",
                    flush=True,
                )
                write_manifest(records, started_at)
        finally:
            bridge.shutdown()

    if total < len(records):
        write_manifest(records, started_at)
        print(f"パイロット完了: {total}本。残り{len(records) - total}本です。", flush=True)
        return

    for record in records:
        path = OUT_DIR / str(record["wav"])
        if not path.exists():
            raise RuntimeError(f"Missing A/B audio: {path}")
        if "audio" not in record:
            record.update({"status": "generated", "audio": audio_stats(path)})
    combined = combine_audio(records)
    write_playlist(records)
    write_readme(combined)
    write_html(records)
    write_manifest(records, started_at, dt.datetime.now(dt.timezone.utc).isoformat())
    package_output()
    print(f"完了: {len(records)}本 / 新規生成{generated}本 / 出力={OUT_DIR} / ZIP={ZIP_PATH} / ZIP SHA256={sha256(ZIP_PATH)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N records for a pilot run.")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 24:
        raise SystemExit("--limit must be between 1 and 24")
    run(args.limit)


if __name__ == "__main__":
    main()
