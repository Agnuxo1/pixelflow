# Deploying the pixelflow Space to HuggingFace

1. Log in to HuggingFace CLI:
   ```
   huggingface-cli login
   ```
   Paste your HF access token when prompted (create one at https://huggingface.co/settings/tokens).

2. Create the Space repository (skip if you already created it via the web UI):
   ```
   huggingface-cli repo create pixelflow --type=space --space_sdk=gradio
   ```

3. Clone the empty Space repo, copy the contents of `space/` into it, then push:
   ```
   git clone https://huggingface.co/spaces/Agnuxo1/pixelflow hf-pixelflow
   cp -r D:/PROJECTS/pixelflow/space/* hf-pixelflow/
   cd hf-pixelflow
   git add .
   git commit -m "initial Space deployment"
   git push
   ```

4. The first build takes roughly 5 minutes: HuggingFace installs the requirements
   (including pixelflow from GitHub) and then trains the MNIST classifier at startup.
   Subsequent cold starts re-train as well (no persistent cache between restarts on
   the free tier). Training on 2000 samples takes under 60 seconds on CPU.

5. Check build logs at:
   https://huggingface.co/spaces/Agnuxo1/pixelflow/logs

## Notes

- The Space uses `backend="cpu"` throughout. The `moderngl` GPU backend exists
  locally but requires OpenGL drivers that are not available on HF Spaces CPU tier.
- To use a GPU-tier Space, switch the hardware to T4 in the Space settings and
  change `backend="cpu"` to `backend="moderngl"` in `app.py`.
