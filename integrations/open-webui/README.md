# Pixelflow Reservoir — Open WebUI Tool

**Experimental research integration.** A single-file Open WebUI tool that
runs an uploaded image through a tiny [pixelflow](https://github.com/franciscoangulo/pixelflow)
reservoir (32x32x4, 4 steps, `wave` rule, CPU backend) and returns a
short description of the resulting feature vector.

It is a demo / exploration tool — not a real classifier. It shows what
the model sees after a few CA steps, which is useful for RC debugging
and education.

## Install

1. Install dependencies on the host running Open WebUI:

   ```bash
   pip install pixelflow-rc>=0.3.0 pillow numpy pydantic
   ```

2. In Open WebUI: **Workspace -> Tools -> +**, paste the contents of
   `pixelflow_tool.py`, give it a name, and save. See the official
   docs: <https://docs.openwebui.com/features/plugin/tools/>.

3. Enable the tool in a chat and ask the model to call
   `classify_with_reservoir` with a base64 image.

## Usage

The tool exposes one method:

```
classify_with_reservoir(image_base64: str) -> str
```

It accepts either a raw base64 string or a `data:image/png;base64,...`
data URL. It returns a string like:

```
Reservoir features: dim=4096, L2 norm=12.3421, sparsity (|v|<0.0001)=0.018.
Config: 32x32x4, rule='wave', steps=4.
```

## Valves

Admins can tweak the reservoir shape, rule, and seed from the tool's
valves panel without editing code.
