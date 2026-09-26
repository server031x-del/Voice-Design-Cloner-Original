# VoiceDesignCloner on Google Colab

The notebook in `colab/voice-design-cloner-colab.ipynb` runs the current source snapshot with an Irodori-TTS GPU runtime. It asks Gradio to create a temporary public share URL only when the app is launched from the notebook. Normal local startup remains private.

## Prepare the source bundle on Windows

From the repository root, run:

```powershell
python .\colab\package_colab_source.py --force
```

This writes `output/colab/voice_design_cloner_colab_source.zip`. Upload that ZIP when prompted by the notebook. The packager includes the current working-tree files, including uncommitted V4 changes, while excluding `.git`, environments, logs, `config.json`, `output/`, `references/`, and LoRA/audio assets.

## Run in Colab

1. Open the notebook in Google Colab and select a GPU runtime.
2. Upload the generated source ZIP when prompted.
3. Run the cells in order. The setup cell installs the app and Irodori-TTS dependencies; the app cell creates and prints a Gradio share URL.
4. Keep the runtime running while using the URL. Stop the app/runtime when finished; the share URL is temporary.

The notebook mounts Drive for persistent `output/` files by default. Existing voice references and LoRA adapters are intentionally excluded from the source ZIP; place them in the Drive-backed output folders after the mount cell (`output/voice_design/` and `output/lora_v4/`). Model downloads and the Irodori virtual environment remain on the ephemeral Colab runtime and must be recreated after a reset.
