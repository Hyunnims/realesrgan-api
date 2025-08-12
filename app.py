from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps
import io, tempfile, numpy as np, torch, os, urllib.request

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI(title="Real-ESRGAN REST API (CPU Friendly)", version="1.1")

# ====== Pastikan weights tersedia (auto-download di runtime) ======
WEIGHTS_DIR = "weights"
WEIGHT_PATH = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")
WEIGHT_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

os.makedirs(WEIGHTS_DIR, exist_ok=True)
if not os.path.exists(WEIGHT_PATH):
    # Download sekali saat container start
    urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_PATH)

# ====== Model config (x4plus) ======
rrdbnet = RRDBNet(
    num_in_ch=3, num_out_ch=3,
    num_feat=64, num_block=23, num_grow_ch=32, scale=4
)

# Catatan: half=True hanya jika ada GPU CUDA (Railway free: CPU, jadi False)
upsampler = RealESRGANer(
    scale=4,
    model_path=WEIGHT_PATH,
    model=rrdbnet,
    tile=0,         # akan dioverride dinamis di endpoint
    tile_pad=10,
    pre_pad=0,
    half=torch.cuda.is_available()
)

def pick_tile_size(w, h):
    """Pilih tile otomatis biar hemat RAM/CPU."""
    mp = (w * h) / 1_000_000  # megapixels
    if mp <= 0.8:
        return 0
    elif mp <= 2.0:
        return 300
    elif mp <= 4.0:
        return 200
    else:
        return 128

@app.get("/")
def health():
    return {"ok": True, "msg": "Real-ESRGAN API is running (CPU-friendly defaults)"}

@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    outscale: float = 2.0,          # default 2x (lebih ringan)
    final_scale: float = 2.0        # mau 4x total: set final_scale=4 (ESRGAN 2x + Lanczos 2x)
):
    """
    outscale: faktor ESRGAN (default 2x supaya ringan)
    final_scale: faktor akhir setelah ESRGAN (Lanczos). Contoh:
      - 2x ESRGAN murni         -> outscale=2, final_scale=2
      - 4x hemat CPU (disarankan)-> outscale=2, final_scale=4
      - 4x maksimal (lebih berat)-> outscale=4, final_scale=4
    """
    try:
        data = await file.read()
        pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))

        # Simpan alpha kalau ada, dan normalkan ke RGB
        alpha = None
        if pil_img.mode == "RGBA":
            alpha = pil_img.split()[-1]
            pil_img = pil_img.convert("RGB")
        elif pil_img.mode in ("L", "P"):
            pil_img = pil_img.convert("RGB")
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Tentukan tile otomatis
        tile_choice = pick_tile_size(pil_img.width, pil_img.height)
        upsampler.tile = tile_choice

        # PIL (RGB) -> NumPy (BGR)
        img_rgb = np.array(pil_img)
        img_bgr = img_rgb[:, :, ::-1]

        # Jalankan ESRGAN
        output_bgr, _ = upsampler.enhance(img_bgr, outscale=float(outscale))

        # BGR -> RGB, kembali ke PIL
        out_rgb = output_bgr[:, :, ::-1]
        out_pil = Image.fromarray(out_rgb)

        # Kembalikan alpha (jika ada)
        if alpha is not None:
            new_w, new_h = out_pil.width, out_pil.height
            alpha_up = alpha.resize((new_w, new_h), resample=Image.BICUBIC)
            out_pil = out_pil.convert("RGBA")
            out_pil.putalpha(alpha_up)

        # Post-upscale Lanczos (hemat CPU) bila diminta
        if final_scale > outscale:
            factor = float(final_scale) / float(outscale)
            target_w = max(1, int(round(out_pil.width * factor)))
            target_h = max(1, int(round(out_pil.height * factor)))
            out_pil = out_pil.resize((target_w, target_h), resample=Image.LANCZOS)

        # Simpan sementara & kirim balik
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        out_pil.save(tmp_out.name)

        filename = file.filename or "image.png"
        return FileResponse(tmp_out.name, media_type="image/png", filename=f"upscaled_{filename}")

    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
