# app.py
import os
import uuid
import base64
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ML imports
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms

# ---------- Configuration ----------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODELS_FOLDER = "models"
MODEL_FILENAME = "mobilenetv2_deepfake.pth"   # put your .pth here
MODEL_PATH = os.path.join(MODELS_FOLDER, MODEL_FILENAME)

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv"}
FRAME_SKIP = 10      # sample every Nth frame
THRESHOLD = 0.5      # threshold for "fake" decisions
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# ---------- Preprocessing ----------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------- Model loading ----------
def load_model(path):
    """
    Loads MobileNetV2 with classifier adjusted for 2 classes and loads state_dict.
    Assumes state_dict was saved with torch.save(model.state_dict(), PATH)
    """
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Put your .pth as {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded on", DEVICE)

# ---------- Grad-CAM ----------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # register hooks
        target_layer.register_forward_hook(self._save_activation)
        # use register_full_backward_hook for modern PyTorch
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; take [0]
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        """
        x: torch tensor (1,3,224,224) on CPU or DEVICE
        returns: numpy array 224x224 normed 0..1
        """
        x = x.to(DEVICE)
        preds = self.model(x)
        # take softmax score for 'fake' class index 1
        probs = torch.softmax(preds, dim=1)
        score = probs[:, 1]    # shape [1]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        # global-average pool gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # shape [1,C,1,1]
        gradcam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))  # [1,1,H,W]
        gradcam = torch.nn.functional.interpolate(gradcam, size=(224,224), mode="bilinear", align_corners=False)
        gradcam = gradcam.squeeze().cpu().numpy()
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
        return gradcam

# Attach GradCAM to last conv feature layer (MobileNetV2 typical selection)
target_layer = model.features[-1]
gradcam_fn = GradCAM(model, target_layer)

# ---------- Utilities ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ---------- Video analysis ----------
def analyze_video(video_path, save_gradcam=False, frame_skip=FRAME_SKIP, threshold=THRESHOLD):
    """
    Analyze video by sampling frames every `frame_skip` frames.
    Returns:
      avg_score: float (0..1) or None if no frames processed
      gradcam_fname: str or None (if saved to disk)
      gradcam_base64: str or None (if NOT saved to disk, returned as base64 jpeg)
    Behavior:
      - compute per-sampled-frame fake-prob (softmax index 1)
      - avg_score = mean of frame scores
      - if avg_score > threshold and some frame had score > threshold, create grad-cam on first such frame
      - if save_gradcam True -> save to outputs/ and return filename
      - if save_gradcam False -> return base64 string (no disk save)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file: " + video_path)

    scores = []
    suspect_frame = None
    suspect_tensor = None

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            # ensure frame is BGR (OpenCV) — transform expects HWC uint8 (RGB conversion inside transform will handle)
            try:
                # transform expects numpy HxWxC in uint8 (RGB), but our transform uses ToPILImage so it's fine with BGR as well;
                # convert BGR -> RGB to be consistent with training.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = frame
            img_t = transform(rgb).unsqueeze(0)  # 1x3x224x224
            with torch.no_grad():
                out = model(img_t.to(DEVICE))
                prob_fake = torch.softmax(out, dim=1)[:, 1].item()
            scores.append(prob_fake)
            if prob_fake > threshold and suspect_frame is None:
                suspect_frame = frame.copy()
                suspect_tensor = img_t  # keep for gradcam
        idx += 1

    cap.release()

    if len(scores) == 0:
        # no frames processed
        return None, None, None

    avg_score = float(np.mean(scores))

    gradcam_fname = None
    gradcam_base64 = None

    if avg_score > threshold and suspect_tensor is not None:
        # compute grad-cam map
        cam = gradcam_fn(suspect_tensor)  # 224x224 float 0..1

        # build overlay image (RGB)
        frame_resized = cv2.resize(suspect_frame, (224,224))
        ori_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        heatmap = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET).astype(np.float32)/255.0
        overlay = (0.6 * ori_rgb + 0.4 * heatmap)
        overlay = (overlay / overlay.max() * 255).astype("uint8")
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        if save_gradcam:
            fname = f"gradcam_{uuid.uuid4().hex[:10]}.jpg"
            outpath = os.path.join(OUTPUT_FOLDER, fname)
            cv2.imwrite(outpath, overlay_bgr)
            gradcam_fname = fname
        else:
            # return base64 (no persistent save)
            _, buffer = cv2.imencode(".jpg", overlay_bgr)
            gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

    return avg_score, gradcam_fname, gradcam_base64

# ---------- Routes ----------
@app.route("/")
def index():
    # serve the static index.html (from static/index.html)
    return app.send_static_file("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No file part 'video' in request"}), 400
    file = request.files['video']
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        # remember last upload (simple method)
        with open("last_upload.txt", "w") as f:
            f.write(save_path)
        return jsonify({"success": True, "filename": filename}), 200
    else:
        return jsonify({"success": False, "error": "File type not allowed"}), 400

@app.route("/detect", methods=["POST"])
def detect():
    # parse form
    filename = request.form.get("filename")
    save_gradcam = request.form.get("save_gradcam", "false").lower() == "true"

    # resolve uploaded file
    if filename:
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(filename))
    else:
        if not os.path.exists("last_upload.txt"):
            return jsonify({"success": False, "error": "No filename provided and last_upload.txt missing"}), 400
        with open("last_upload.txt", "r") as f:
            upload_path = f.read().strip()

    if not os.path.exists(upload_path):
        return jsonify({"success": False, "error": f"Uploaded file not found: {upload_path}"}), 400

    try:
        avg_score, gradcam_fname, gradcam_base64 = analyze_video(upload_path, save_gradcam=save_gradcam)

        if avg_score is None:
            return jsonify({"success": False, "error": "No frames were processed from the video (maybe corrupted or empty)"}), 400

        response = {
            "success": True,
            "avg_score": avg_score,
            "is_deepfake": (avg_score > THRESHOLD),
            "gradcam_url": None,
            "gradcam_base64": None
        }

        # response = {
        #     "success": True,
        #     "avg_score": avg_score,
        #     "gradcam_url": None,
        #     "gradcam_base64": None
        # }

        if gradcam_fname:
            response["gradcam_url"] = f"/outputs/{gradcam_fname}"
        elif gradcam_base64:
            response["gradcam_base64"] = gradcam_base64

        return jsonify(response), 200

    except Exception as e:
        # log error
        print("Error in /detect:", str(e))
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

# ---------- Run ----------
if __name__ == "__main__":
    print("Starting Flask on 0.0.0.0:5000, using device:", DEVICE)
    print("Uploads folder:", os.path.abspath(UPLOAD_FOLDER))
    print("Outputs folder:", os.path.abspath(OUTPUT_FOLDER))
    print("Model path:", os.path.abspath(MODEL_PATH))
    app.run(debug=True, host="0.0.0.0", port=5000)
