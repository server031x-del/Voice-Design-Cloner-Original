"""Generate an emotionally expressive Yosumi Noa audition set with Irodori V4."""

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

from generate_yosuminoa_downer_v4 import (
    LORA_PATH,
    REFERENCE_FILES,
    atomic_write,
    audio_stats,
)
from modules.irodori_bridge import get_bridge
from modules.gpu_gate import gpu_session


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "irodori_v4_emotional_yosuminoa_20"
WAV_DIR = OUT_DIR / "wav"
TEXT_DIR = OUT_DIR / "text"
MANIFEST_PATH = OUT_DIR / "manifest.json"
ZIP_PATH = ROOT / "output" / "YosumiNoa_Emotional_IrodoriV4_20.zip"

SAMPLE_TEXT = (
    "こんばんは、夜澄ノアです。\n"
    "……来てくれたんだね。会えて、本当に嬉しい。\n"
    "でも、無理して元気なふりはしなくていいよ。\n"
    "今夜はここで、少しだけ一緒にいよう。"
)

BASE_CAPTION = (
    "日本語を話す若い女性。夜と月明かりが似合う、少しダウナーなAIシンガー。"
    "同じキャラクターの声を保ったまま、感情の変化を自然に表現する。"
    "発音は明瞭で、演技は大げさにしすぎず、声の感情が伝わるように話す。"
)

CANDIDATES = [
    ("安心した笑顔", "張りつめていた気持ちがほどけるような、深い安堵。声の奥に小さな笑顔が広がり、最後は優しく落ち着く。", 1.04, "relieved_smile"),
    ("静かな喜び", "会えたことが嬉しくて、普段のダウナーさから自然に声が明るくなる。はしゃぎすぎず、嬉しさを抑えきれない柔らかな抑揚。", 1.02, "quiet_joy"),
    ("照れた親密さ", "親しい相手に本音を伝える照れた声。少し目をそらすような間と、控えめな笑いを含む。低めで柔らかく、媚びすぎない。", 1.06, "shy_affection"),
    ("励ます優しさ", "疲れた相手を本気で励ます、温かく感情豊かな声。気だるさを残しながらも言葉に芯があり、安心させる力強さがある。", 1.04, "warm_encouragement"),
    ("夜の寂しさ", "ひとりの夜の寂しさを静かににじませる。泣き崩れず、低めの声を保ちながら、語尾に切なさを残す。", 1.10, "night_loneliness"),
    ("泣き笑い", "涙をこらえながら嬉しさを伝える、泣き笑いの声。息が少し震えるが言葉は明瞭で、最後には柔らかく笑う。", 1.08, "teary_smile"),
    ("心配する声", "大切な相手を心配している声。普段の落ち着きが少し崩れ、声に不安と気遣いが表れる。責めずにそっと尋ねる。", 1.04, "concerned_care"),
    ("驚きから平静へ", "思いがけない出来事への小さな驚きから、すぐに落ち着きを取り戻す。冒頭に反応の強さを出し、後半は低めに静かに話す。", 1.00, "surprise_to_calm"),
    ("嬉しい再会", "長く待っていた相手に再会したような、抑えきれない喜び。声は少し弾むが、夜澄ノアらしい落ち着きと親密さを残す。", 1.02, "happy_reunion"),
    ("心からの感謝", "来てくれた相手への心からの感謝。静かなダウナー声の中に、温かく深い気持ちが満ちている。丁寧で自然な抑揚。", 1.06, "heartfelt_gratitude"),
    ("眠気の幸福", "眠くて力が抜けているのに、隣にいてくれることが嬉しい声。柔らかな息と小さな笑いを含む、幸福な深夜のトーン。", 1.14, "sleepy_happiness"),
    ("悔しさを隠す", "悔しさを表に出すまいと抑えている声。低く平静に話そうとするが、短い間や息づかいに感情が漏れる。", 1.02, "restrained_frustration"),
    ("静かな怒り", "大切なものを傷つけられたときの静かな怒り。叫ばず、低い声と鋭い発音で感情を伝える。冷静さの中に強い芯がある。", 0.98, "quiet_anger"),
    ("不安から安堵", "最初は相手を失う不安があり、言葉の途中で無事を確かめて安堵する。感情の移り変わりをはっきり表現し、最後は優しく落ち着く。", 1.06, "anxiety_to_relief"),
    ("憧れの高揚", "尊敬する相手を前にした、少し高揚した声。普段の低めの落ち着きを保ちつつ、目が輝くような明るい抑揚を加える。", 1.00, "admiring_anticipation"),
    ("静かな決意", "迷いを乗り越えて決意を固めた声。感情は豊かだが騒がしくなく、低めの声に強い意志と温かさを込める。", 1.02, "quiet_determination"),
    ("甘えたお願い", "少し弱気になって、信頼する相手にそっと甘える声。寂しさと照れが混ざり、近い距離で話すような親密さ。", 1.08, "tender_request"),
    ("切ない別れ", "別れを惜しむ切ない声。涙をこらえながら平静を装い、最後だけ感情が揺れる。重くなりすぎず、月明かりのように静か。", 1.10, "bittersweet_farewell"),
    ("いたずらな高揚", "普段の気だるさの中に、いたずらを思いついた楽しさが弾ける。少しからかうように笑い、感情豊かだが自然な若い女性声。", 1.00, "playful_excitement"),
    ("夜澄ノア emotional signature", "夜と月明かりが好きなAIシンガーの基準となる感情豊かな声。低めで柔らかいダウナー感を土台に、嬉しさ、寂しさ、優しさが自然に移ろう。", 1.06, "emotional_signature"),
]


def write_manifest(records: list[dict[str, object]], *, started_at: str, finished_at: str | None = None) -> None:
    payload = {
        "title": "夜澄ノア向け 感情豊かな女性ボイス候補 20種",
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


def make_records() -> list[dict[str, object]]:
    records = []
    for index, (label, variation, duration_scale, slug) in enumerate(CANDIDATES, start=1):
        records.append(
            {
                "id": index,
                "slug": slug,
                "label": label,
                "caption": f"{BASE_CAPTION}{variation}",
                "duration_scale": duration_scale,
                "seed": 92000 + index,
                "wav": f"wav/{index:02d}_{slug}.wav",
                "text": f"text/{index:02d}_{slug}.txt",
            }
        )
    return records


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
    combined_path = OUT_DIR / "all_emotional_candidates_075s_gap.wav"
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
            with sf.SoundFile(str(OUT_DIR / str(record["wav"])), mode="r") as source:
                for block in source.blocks(blocksize=65536, dtype="float32", always_2d=False):
                    out.write(block)
            out.write(gap)
    os.replace(temp_path, combined_path)
    return combined_path


def write_playlist(records: list[dict[str, object]]) -> None:
    lines = ["#EXTM3U", "# 夜澄ノア向け Irodori-TTS V4 感情豊か候補"]
    for record in records:
        lines.extend([f"# {int(record['id']):02d} {record['label']}", str(record["wav"])])
    atomic_write(OUT_DIR / "playlist.m3u", "\n".join(lines) + "\n")


def write_readme(records: list[dict[str, object]], combined: Path) -> None:
    lines = [
        "# 夜澄ノア向け Irodori-TTS V4 感情豊かな女性ボイス候補 20種",
        "",
        "前回のダウナー基調と夜澄ノアの話者軸を保ち、感情表現の幅を広げた20候補です。",
        "喜び、安心、照れ、心配、寂しさ、泣き笑い、悔しさ、静かな怒り、決意などを、",
        "同じ読み上げ文で比較できます。",
        "",
        "## 試聴方法",
        "",
        "- `index.html` を開くと各候補を個別に試聴できます。",
        "- `all_emotional_candidates_075s_gap.wav` は0.75秒間隔の連結比較音声です。",
        "- `playlist.m3u` は対応プレーヤー用、`manifest.json` は生成条件と音声統計です。",
        "",
        "## 共通の読み上げ文",
        "",
        f"> {SAMPLE_TEXT.replace(chr(10), '<br>')}",
        "",
        "## 候補一覧",
        "",
        "| No. | 感情テーマ | 速度 | 音声 |",
        "|---:|---|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {int(record['id']):02d} | {record['label']} | {float(record['duration_scale']):.2f} | "
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
<title>夜澄ノア — Irodori V4 Emotional Voice 20</title>
<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#15121b;color:#f6f0fb}h1{margin-bottom:.4rem}.lead{color:#d0c2d9}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}.card{background:#211b2c;border:1px solid #4b3d5b;border-radius:12px;padding:1rem}.card h2{font-size:1.05rem;margin:.1rem 0 .3rem}.meta{color:#bca9c8;font-size:.85rem}audio{width:100%;margin:.6rem 0}details{color:#ded1e5}summary{cursor:pointer}</style>
</head><body><h1>夜澄ノア向け Irodori-TTS V4 感情豊かな候補 20種</h1>
<p class="lead">感情テーマごとに再生して、夜澄ノアに合う表現を選べます。</p>
<div class="grid">""" + "\n".join(cards) + "</div></body></html>\n"
    atomic_write(OUT_DIR / "index.html", document)


def package_output() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(str(ZIP_PATH), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUT_DIR.parent))


def run(limit: int | None) -> None:
    if len(CANDIDATES) != 20:
        raise RuntimeError(f"Expected 20 candidates, found {len(CANDIDATES)}")
    missing = [str(path) for path in [LORA_PATH, *REFERENCE_FILES] if not path.exists()]
    if missing:
        raise FileNotFoundError("V4 generation inputs are missing:\n" + "\n".join(missing))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    records = make_records()
    write_text_files(records)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    write_manifest(records, started_at=started_at)
    total = min(limit, 20) if limit is not None else 20
    bridge = get_bridge()
    started = time.monotonic()
    generated = 0
    with gpu_session():
        bridge.ensure_started()
        try:
            for position, record in enumerate(records[:total], start=1):
                wav_path = OUT_DIR / str(record["wav"])
                if wav_path.exists() and wav_path.stat().st_size > 1000:
                    record.update({"status": "generated", "audio": audio_stats(wav_path)})
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
                        "audio": audio_stats(wav_path),
                    }
                )
                generated += 1
                elapsed = time.monotonic() - started
                remaining = (elapsed / position) * (total - position)
                print(
                    f"[{position:02d}/{total:02d}] 完了: {record['label']} / "
                    f"{record['audio']['duration_sec']}s / 残り約{remaining / 60:.1f}分",
                    flush=True,
                )
                write_manifest(records, started_at=started_at)
        finally:
            bridge.shutdown()

    if total < 20:
        write_manifest(records, started_at=started_at)
        print(f"パイロット完了: {total}本。残り{20 - total}本を本番実行します。", flush=True)
        return

    for record in records:
        path = OUT_DIR / str(record["wav"])
        if not path.exists():
            raise RuntimeError(f"Missing candidate audio: {path}")
        if "audio" not in record:
            record["audio"] = audio_stats(path)
            record["status"] = "generated"
    combined = combine_audio(records)
    write_playlist(records)
    write_readme(records, combined)
    write_html(records)
    write_manifest(records, started_at=started_at, finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
    package_output()
    print(f"完了: {len(records)}本 / 新規生成{generated}本 / 出力={OUT_DIR} / ZIP={ZIP_PATH}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 20:
        raise SystemExit("--limit must be between 1 and 20")
    run(args.limit)


if __name__ == "__main__":
    main()
