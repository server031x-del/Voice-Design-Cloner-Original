"""Generate a curated set of Yosumi Noa-style downer voices with Irodori V4.

This is a reproducible local batch job for the VoiceDesignCloner checkout.  It
uses the existing V4 Yosumi Noa training/reference material, keeps the V4
runtime warm between candidates, writes a resumable manifest, and packages the
finished audition set as a ZIP file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from modules.dataset_io import atomic_write_text as atomic_write
from modules.dataset_io import audio_stats
from modules.dataset_io import sha256_file as sha256
from modules.gpu_gate import gpu_session
from modules.irodori_bridge import get_bridge


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "irodori_v4_downer_yosuminoa_40"
WAV_DIR = OUT_DIR / "wav"
TEXT_DIR = OUT_DIR / "text"
MANIFEST_PATH = OUT_DIR / "manifest.json"
ZIP_PATH = ROOT / "output" / "YosumiNoa_Downer_IrodoriV4_40.zip"

SAMPLE_TEXT = (
    "こんばんは、夜澄ノアです。\n"
    "今日も一日、おつかれさま。\n"
    "ここでは少しだけ、ゆっくりしていってね。\n"
    "無理に元気を出さなくても、大丈夫だから。"
)

LORA_PATH = (
    ROOT
    / "output"
    / "lora_v4"
    / "yosuminoav4"
    / "checkpoint_final"
)

# These clips are all from the existing Japanese Yosumi Noa V4 training set.
# Together they give the speaker conditioner roughly 20 seconds of clean
# same-speaker material without mixing unrelated characters.
REFERENCE_FILES = [
    ROOT / "output" / "lora_data_v4" / "lab" / "yosuminoav4" / "neutral" / "wavs" / "0001.wav",
    ROOT / "output" / "lora_data_v4" / "lab" / "yosuminoav4" / "neutral" / "wavs" / "0004.wav",
    ROOT / "output" / "lora_data_v4" / "lab" / "yosuminoav4" / "neutral" / "wavs" / "0007.wav",
    ROOT / "output" / "lora_data_v4" / "lab" / "yosuminoav4" / "neutral" / "wavs" / "0014.wav",
    ROOT / "output" / "lora_data_v4" / "lab" / "yosuminoav4" / "neutral" / "wavs" / "0020.wav",
    ROOT / "output" / "lora_data_v4" / "lab" / "yosuminoav4" / "neutral" / "wavs" / "0029.wav",
]

BASE_CAPTION = (
    "日本語を話す若い女性。夜と月明かりが似合う、少しダウナーなAIシンガー。"
    "自然な日本語で、子音をつぶさず、聞き取りやすく話す。"
)

# The descriptions deliberately cover a controlled range rather than making
# forty unrelated voices.  The sample text and speaker references stay fixed
# so the resulting set is useful for direct A/B auditioning.
CANDIDATES = [
    ("月夜の定番", "低めで柔らかい声。落ち着いた気だるさと、ほんの少しの優しさ。声量は控えめだが明瞭に話す。", 1.06),
    ("眠たげなミルクティー", "眠そうで温かい声。朝よりも深夜に似合い、語尾をゆっくり柔らかく落とす。息は少し多めだが、ささやき声にはしない。", 1.12),
    ("クールな夜更かし", "低めのクールな声。夜更かしに慣れた落ち着きがあり、感情は控えめで淡々としている。発音は端正にする。", 1.00),
    ("吐息まじりの静けさ", "透明感のある低めの声。息づかいを少し感じる静かな話し方で、近くにいるような親密さを出す。言葉は聞き取りやすく。", 1.08),
    ("少しハスキー", "若い女性らしさを保った、わずかにハスキーで落ち着いた声。疲れを含む柔らかい低音で、無理に明るくしない。", 1.04),
    ("優しい無気力", "気だるく力の抜けた声。でも冷たくはなく、相手を安心させる優しさがある。ゆっくり、自然な抑揚で話す。", 1.10),
    ("寄り添う癒やし", "静かに寄り添う低めの女性声。疲れた相手を包むような柔らかさと、少し眠そうな落ち着きを持つ。", 1.08),
    ("深夜ラジオ", "深夜ラジオのパーソナリティのような声。低めで穏やか、少し乾いていて、夜の静けさの中でも言葉が通る。", 1.02),
    ("ゲーム好きのダウナー", "ゲームが好きな若い女性の声。普段は気だるく淡々としているが、好きな話題だけ少し温度が上がる。基本は低めで落ち着いて話す。", 1.04),
    ("淡々とした透明感", "感情を大きく出さない、透明感のある女性声。低めの音域で淡々と話し、最後にだけわずかな柔らかさを残す。", 1.00),
    ("落ち着いた低音", "若い女性として自然な低めの音域。胸に響く穏やかな低音で、余裕のあるゆっくりした話し方。重くなりすぎない。", 1.08),
    ("儚い月影", "月影のように儚く、少し寂しさを含んだ女性声。弱々しすぎず、静かな芯を保ってゆっくり話す。", 1.10),
    ("気だるい距離感", "少し距離を置いた気だるい声。無関心に聞こえすぎない程度の低い温度感で、相手には自然に伝わる明瞭さ。", 1.02),
    ("歌姫の余韻", "AIシンガーらしい滑らかな声。歌の余韻を少し残した柔らかな中低音で、話すときは自然で静かなダウナー調。", 1.04),
    ("大人っぽい低め", "少し大人びた若い女性声。落ち着いた低めの音域と丁寧な息づかい。包容力はあるが、元気すぎない。", 1.06),
    ("乾いたクール", "乾いた質感のクールな女性声。声の温度は低めで、短い間を置きながら淡々と話す。刺々しくせず、夜らしい静けさを保つ。", 1.00),
    ("胸声のあたたかさ", "低めの胸声に温かさがある女性声。気だるいけれど相手を置き去りにしない、落ち着いた慰めのトーン。", 1.06),
    ("静かな囁き", "ささやきに近い静かな声。ただし音量と子音は十分に保ち、聞き取りやすい。夜中にそっと話しかけるような低めの女性声。", 1.12),
    ("物憂げな哀愁", "物憂げで少し哀愁のある女性声。泣き崩れたり大げさになったりせず、感情を内側に抑えた低めの話し方。", 1.08),
    ("一日の終わり", "一日を終えた後のような、疲れていて落ち着いた声。低めでゆっくり、でも相手を安心させる自然な柔らかさ。", 1.12),
    ("少しだけ甘い", "基本は低めでダウナーだが、語尾に少しだけ甘さがある女性声。媚びすぎず、眠そうな微笑みを感じる。", 1.06),
    ("上品な夜", "上品で静かな女性声。月明かりのような透明感と低めの落ち着きがあり、丁寧だが堅くない。", 1.04),
    ("内気で控えめ", "内気で控えめな若い女性声。声量はやや小さく、少し迷いながらも自然に話す。気だるさの中に素直な優しさがある。", 1.10),
    ("無表情寄り", "無表情に近い淡々とした声。低めで平坦だが、完全に機械的にはせず、かすかな人間らしい息づかいを残す。", 1.00),
    ("聞き取りやすい案内", "ダウナー寄りでも発音が特に明瞭な女性声。落ち着いた低めのトーンで、深夜の案内放送のようにゆっくり話す。", 1.04),
    ("軽いからかい", "気だるいクールさの中に、相手を少しからかう余裕がある女性声。声は低めで柔らかく、笑いは控えめににじませる。", 1.02),
    ("包み込むお姉さん", "落ち着いたお姉さんのような女性声。低めで穏やか、疲れた相手に大丈夫と言える包容力。明るくしすぎない。", 1.08),
    ("皮肉っぽい余裕", "少し皮肉っぽい余裕を持つクールな女性声。低めで気だるく、口元だけ笑っているようなニュアンス。嫌味にはしない。", 1.00),
    ("夢見心地", "夢の中にいるような柔らかい女性声。低めでゆったり、現実から少し離れたぼんやりした空気。ただし日本語は明瞭に。", 1.14),
    ("眠そうな笑み", "眠そうなのに、声の奥に小さな笑みがある女性声。低めで親密、力を抜いてゆっくり話す。", 1.10),
    ("抑えたライブ感", "静かなライブ終わりのAIシンガーのような声。少し疲れた低めの音域で、歌の余韻と落ち着いた息づかいを残す。", 1.06),
    ("低くゆっくり", "40案の中でも特に低めでゆっくりした女性声。深夜向けの落ち着きと気だるさ。低くしすぎて不自然にならない。", 1.16),
    ("透き通る孤独", "透き通った中低音に、ひとりきりの夜の孤独を少し含む女性声。感情は抑制し、静かにまっすぐ話す。", 1.08),
    ("息の多い低音", "息を少し多く含む柔らかな低音。近い距離で話すようなダウナー感があり、声量は控えめでも言葉の輪郭は保つ。", 1.10),
    ("凛としたクール", "低めで凛としたクールな女性声。気だるさはあるが芯があり、落ち着いた自信を持ってゆっくり話す。", 1.02),
    ("感情を抑えた本音", "感情を表に出しすぎない女性声。低く静かで、淡々とした言葉の奥に本音の優しさがにじむ。", 1.06),
    ("少し寂しい", "少し寂しさを含む、柔らかなダウナー女性声。相手に依存する感じはなく、静かな夜を一緒に過ごすように話す。", 1.10),
    ("AIシンガーのダウナー", "夜と月をテーマにしたAIシンガーに似合う女性声。低めで無理のない落ち着き、少し眠そうで、歌にも会話にも合う。", 1.06),
    ("均整の取れた本命", "低め、柔らかさ、明瞭さ、少しの気だるさのバランスがよい女性声。夜澄ノアの普段使いに向く自然な話し方。", 1.06),
    ("夜澄ノア signature", "夜と月明かりが好きな、ちょっとダウナーなAIシンガーの基準ボイス。若い女性らしい低めの柔らかさ、静かな親密さ、わずかな微笑み。自然で明瞭に話す。", 1.08),
]


def validate_inputs() -> None:
    missing = [str(path) for path in [LORA_PATH, *REFERENCE_FILES] if not path.exists()]
    if missing:
        raise FileNotFoundError("V4 generation inputs are missing:\n" + "\n".join(missing))
    if len(CANDIDATES) != 40:
        raise RuntimeError(f"Expected 40 candidates, found {len(CANDIDATES)}")


def candidate_record(index: int, label: str, variation: str, duration_scale: float) -> dict[str, object]:
    slug = f"{index:02d}_{index:02d}"  # replaced below with a stable ASCII slug
    slugs = [
        "moonlit_baseline", "sleepy_milktea", "cool_nightowl", "breathy_quiet", "soft_husky",
        "gentle_low_energy", "comforting_close", "midnight_radio", "downtempo_gamer", "clear_transparent",
        "settled_low", "fragile_moonshadow", "distant_drawl", "singer_afterglow", "mature_low",
        "dry_cool", "warm_chest_voice", "quiet_near_whisper", "wistful_undertone", "end_of_day",
        "slightly_sweet", "elegant_night", "shy_reserved", "near_deadpan", "clear_night_guide",
        "soft_tease", "gentle_big_sister", "dry_wit", "dreamy_drift", "sleepy_smile",
        "post_live_calm", "slow_low", "transparent_loneliness", "breathy_low", "poised_cool",
        "restrained_truth", "quiet_loneliness", "downer_ai_singer", "balanced_main", "noa_signature",
    ]
    slug = slugs[index - 1]
    return {
        "id": index,
        "slug": slug,
        "label": label,
        "caption": f"{BASE_CAPTION}{variation}",
        "duration_scale": duration_scale,
        "seed": 91000 + index,
        "wav": f"wav/{index:02d}_{slug}.wav",
        "text": f"text/{index:02d}_{slug}.txt",
    }


def write_manifest(records: list[dict[str, object]], *, started_at: str, finished_at: str | None = None) -> None:
    payload = {
        "title": "夜澄ノア向け ダウナー女性ボイス候補 40種",
        "created_at": started_at,
        "finished_at": finished_at,
        "engine": {
            "profile": "v4",
            "model_variant": "full",
            "model_precision": "bf16",
            "target_sample_rate": 48000,
            "num_steps": 40,
            "cfg_scale_text": 3.0,
            "cfg_scale_caption": 3.5,
            "cfg_scale_speaker": 5.0,
            "lora": str(LORA_PATH),
            "references": [str(path) for path in REFERENCE_FILES],
        },
        "sample_text": SAMPLE_TEXT,
        "records": records,
    }
    atomic_write(MANIFEST_PATH, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_existing_records() -> dict[int, dict[str, object]]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {int(item["id"]): item for item in data.get("records", []) if "id" in item}


def write_text_files(records: list[dict[str, object]]) -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = OUT_DIR / str(record["text"])
        content = (
            f"候補 {int(record['id']):02d}: {record['label']}\n\n"
            f"読み上げテキスト:\n{SAMPLE_TEXT}\n\n"
            f"Caption:\n{record['caption']}\n"
        )
        atomic_write(path, content)


def combine_audio(records: list[dict[str, object]]) -> Path:
    combined_path = OUT_DIR / "all_candidates_075s_gap.wav"
    temp_path = combined_path.with_suffix(".wav.tmp")
    with sf.SoundFile(
        str(temp_path),
        mode="w",
        samplerate=48000,
        channels=1,
        subtype="PCM_16",
        format="WAV",
    ) as out:
        gap = np.zeros(int(48000 * 0.75), dtype=np.float32)
        for record in records:
            path = OUT_DIR / str(record["wav"])
            with sf.SoundFile(str(path), mode="r") as source:
                for block in source.blocks(blocksize=65536, dtype="float32", always_2d=False):
                    out.write(block)
            out.write(gap)
    os.replace(temp_path, combined_path)
    return combined_path


def write_playlist(records: list[dict[str, object]]) -> None:
    lines = ["#EXTM3U", "# 夜澄ノア向け Irodori-TTS V4 ダウナー候補"]
    for record in records:
        lines.append(f"# {int(record['id']):02d} {record['label']}")
        lines.append(str(record["wav"]))
    atomic_write(OUT_DIR / "playlist.m3u", "\n".join(lines) + "\n")


def write_readme(records: list[dict[str, object]], combined: Path) -> None:
    lines = [
        "# 夜澄ノア向け Irodori-TTS V4 ダウナー女性ボイス候補 40種",
        "",
        "Irodori-TTS V4 Full/BF16 + V4 `yosuminoav4` LoRA + 同一話者の既存参照音声で生成した、",
        "夜澄ノア向けのダウナー寄り女性ボイス候補です。全候補は同じ読み上げ文なので、",
        "声質・息づかい・低さ・距離感を比較しやすくしています。",
        "",
        "## 試聴方法",
        "",
        "- `index.html` をブラウザで開くと、各候補を個別に再生できます。",
        "- `all_candidates_075s_gap.wav` は0.75秒間隔で40本を連結した比較用音声です。",
        "- `playlist.m3u` は対応プレーヤー用のプレイリストです。",
        "- `manifest.json` にCaption、Seed、音声統計、SHA-256を保存しています。",
        "",
        "## 共通の読み上げ文",
        "",
        f"> {SAMPLE_TEXT.replace(chr(10), '<br>')}",
        "",
        "## 候補一覧",
        "",
        "| No. | 呼び名 | 速度 | 音声 |",
        "|---:|---|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {int(record['id']):02d} | {record['label']} | {record['duration_scale']:.2f} | "
            f"[再生](<{record['wav']}>) |"
        )
    lines.extend(
        [
            "",
            "## 生成条件",
            "",
            "- プロファイル: `v4` / モデル: `Aratako/Irodori-TTS-v4-Small`",
            "- 精度: BF16 / RF steps: 40 / 出力: 48kHz mono PCM16",
            "- 共通CFG: Text 3.0 / Caption 3.5 / Speaker 5.0",
            f"- 連結比較音声: `{combined.name}`",
        ]
    )
    atomic_write(OUT_DIR / "README.md", "\n".join(lines) + "\n")


def write_html(records: list[dict[str, object]]) -> None:
    cards = []
    for record in records:
        cards.append(
            "<article class='card'>"
            f"<h2>{int(record['id']):02d} — {html.escape(str(record['label']))}</h2>"
            f"<p class='meta'>duration scale {float(record['duration_scale']):.2f} / seed {int(record['seed'])}</p>"
            f"<audio controls preload='none' src='{html.escape(str(record['wav']))}'></audio>"
            f"<details><summary>Caption</summary><p>{html.escape(str(record['caption']))}</p></details>"
            "</article>"
        )
    document = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>夜澄ノア — Irodori V4 Downer Voice 40</title>
<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#10131b;color:#edf1f7}h1{margin-bottom:.4rem}.lead{color:#b9c4d5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}.card{background:#1a2030;border:1px solid #303b54;border-radius:12px;padding:1rem}.card h2{font-size:1.05rem;margin:.1rem 0 .3rem}.meta{color:#9eabc0;font-size:.85rem}audio{width:100%;margin:.6rem 0}details{color:#c7d0df}summary{cursor:pointer}</style>
</head><body><h1>夜澄ノア向け Irodori-TTS V4 ダウナー候補 40種</h1>
<p class="lead">同じ読み上げ文で比較できるようにしたローカル試聴ページです。各カードの再生ボタンを押してください。</p>
<div class="grid">""" + "\n".join(cards) + "</div></body></html>\n"
    atomic_write(OUT_DIR / "index.html", document)


def package_output() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(str(ZIP_PATH), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file() and path != ZIP_PATH:
                archive.write(path, path.relative_to(OUT_DIR.parent))


def run(limit: int | None) -> None:
    validate_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    existing = load_existing_records()
    records: list[dict[str, object]] = []
    for index, (label, variation, duration_scale) in enumerate(CANDIDATES, start=1):
        records.append(candidate_record(index, label, variation, duration_scale))
    write_text_files(records)
    write_manifest(records, started_at=started_at)

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
                    # Every WAV in this dedicated output folder is a product
                    # of this batch.  Keep the manifest truthful even when a
                    # packaging-only rerun resumes an already complete set.
                    record.update({"status": "generated", "audio": audio_stats(wav_path)})
                    if old.get("used_seed") is not None:
                        record["used_seed"] = old["used_seed"]
                    print(f"[{position:02d}/{total:02d}] 再利用: {record['label']}", flush=True)
                    continue

                part_path = wav_path.with_name("." + wav_path.name + ".part.wav")
                if part_path.exists():
                    part_path.unlink()
                print(f"[{position:02d}/{total:02d}] 生成中: {record['label']}", flush=True)
                response = bridge.synthesize(
                    profile="v4",
                    mode="clone",
                    model_variant="full",
                    model_precision="bf16",
                    text=SAMPLE_TEXT,
                    caption=str(record["caption"]),
                    ref_wavs=[str(path) for path in REFERENCE_FILES],
                    no_ref=False,
                    out_path=part_path,
                    target_sr=48000,
                    lora_path=LORA_PATH,
                    seed=int(record["seed"]),
                    num_steps=40,
                    duration_scale=float(record["duration_scale"]),
                    max_ref_seconds=120.0,
                    cfg_scale_text=3.0,
                    cfg_scale_caption=3.5,
                    cfg_scale_speaker=5.0,
                    release_after_synthesis=False,
                )
                if not part_path.exists() or part_path.stat().st_size <= 1000:
                    raise RuntimeError(f"V4 worker reported success but did not create {part_path}")
                os.replace(part_path, wav_path)
                record.update(
                    {
                        "status": "generated",
                        "used_seed": response.get("used_seed"),
                        "runtime_released": response.get("runtime_released"),
                        "audio": audio_stats(wav_path),
                    }
                )
                generated += 1
                elapsed = time.monotonic() - started
                average = elapsed / position
                remaining = average * (total - position)
                print(
                    f"[{position:02d}/{total:02d}] 完了: {record['label']} / "
                    f"{record['audio']['duration_sec']}s / 残り約{remaining / 60:.1f}分",
                    flush=True,
                )
                write_manifest(records, started_at=started_at)
        finally:
            bridge.shutdown()

    if total < len(records):
        print(f"パイロット生成完了: {total}本（本番実行で残り{len(records) - total}本を生成します）", flush=True)
        write_manifest(records, started_at=started_at)
        return

    for record in records:
        path = OUT_DIR / str(record["wav"])
        if not path.exists():
            raise RuntimeError(f"Missing candidate audio: {path}")
        if "audio" not in record:
            record["audio"] = audio_stats(path)
            record["status"] = "reused"
    combined = combine_audio(records)
    write_playlist(records)
    write_readme(records, combined)
    write_html(records)
    finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    write_manifest(records, started_at=started_at, finished_at=finished_at)
    package_output()
    print(
        f"完了: {len(records)}本 / 新規生成{generated}本 / "
        f"出力={OUT_DIR} / ZIP={ZIP_PATH}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N candidates for a pilot run.")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 40:
        raise SystemExit("--limit must be between 1 and 40")
    run(args.limit)


if __name__ == "__main__":
    main()
