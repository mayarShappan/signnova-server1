# ─────────────────────────────────────────
#  SignNova — Flask API Server
#  يشغل الموديل ويقبل landmarks من الويبسايت
#
#  تشغيل:
#    pip install flask flask-cors tensorflow mediapipe --break-system-packages
#    python server.py
#
#  بعدين افتح الويبسايت وهيتوصل أوتوماتيك
# ─────────────────────────────────────────

import os
import sys
import pickle
import base64

import numpy as np
import tensorflow as tf
from keras import layers
from flask import Flask, jsonify, request
from flask_cors import CORS

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions


# ─────────────────────────────────────────
#  Paths  (عدّل لو الـ model folder في مكان تاني)
# ─────────────────────────────────────────
MODEL_PATH   = "model/sign_model.h5"
ENCODER_PATH = "model/label_encoder.pkl"
CONFIG_PATH  = "model/config.pkl"
HAND_MODEL   = "hand_landmarker.task"

for f in [MODEL_PATH, ENCODER_PATH, HAND_MODEL]:
    if not os.path.exists(f):
        print(f"[ERROR] ملف مش موجود: {f}")
        print("        تأكد إن الـ server شغال في نفس folder الـ model")
        sys.exit(1)


# ─────────────────────────────────────────
#  Custom Layers (نفس الـ training code)
# ─────────────────────────────────────────
class FuzzyLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.mu    = self.add_weight(name="mu",    shape=(input_dim, self.output_dim), initializer="random_normal", trainable=True)
        self.sigma = self.add_weight(name="sigma", shape=(input_dim, self.output_dim), initializer="ones",          trainable=True)
        self.b     = self.add_weight(name="b",     shape=(input_dim, self.output_dim),
                                     initializer=tf.keras.initializers.Constant(2.0), trainable=True)
        super().build(input_shape)

    def call(self, x):
        x_exp   = tf.expand_dims(x, axis=-1)
        ratio   = tf.abs((x_exp - self.mu) / (self.sigma + 1e-6))
        two_b   = 2.0 * tf.abs(self.b) + 1e-6
        return tf.reduce_mean(1.0 / (1.0 + tf.pow(ratio + 1e-6, two_b)), axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg["output_dim"] = self.output_dim
        return cfg


class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        pe = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / 10000 ** (2 * i / d_model))
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(pos / 10000 ** (2 * i / d_model))
        self.pe = tf.constant(pe[np.newaxis], dtype=tf.float32)

    def call(self, x):
        return x + self.pe

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return cfg


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn  = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ff1   = layers.Dense(ff_dim, activation="gelu")
        self.ff2   = layers.Dense(d_model)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout_rate)
        self.drop2 = layers.Dropout(dropout_rate)
        self.d_model = d_model; self.num_heads = num_heads
        self.ff_dim  = ff_dim;  self.dropout_rate = dropout_rate

    def call(self, x, training=False):
        a = self.drop1(self.attn(x, x, training=training), training=training)
        x = self.norm1(x + a)
        f = self.drop2(self.ff2(self.ff1(x)), training=training)
        return self.norm2(x + f)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate})
        return cfg


# ─────────────────────────────────────────
#  Feature Engineering  (نفس الـ camera.py)
# ─────────────────────────────────────────
def normalize_landmarks(raw):
    lms   = np.array(raw).reshape(21, 3)
    lms  -= lms[0]
    scale = np.max(np.abs(lms))
    if scale > 0:
        lms /= scale
    return lms.flatten().tolist()


def compute_hand_features(lms_63):
    lms      = np.array(lms_63).reshape(21, 3)
    features = list(lms.flatten())
    TIPS = [4, 8, 12, 16, 20]
    MCP  = [2, 5, 9,  13, 17]
    PIP  = [3, 6, 10, 14, 18]

    for i in range(5):
        tip, pip, mcp = lms[TIPS[i]], lms[PIP[i]], lms[MCP[i]]
        v1, v2 = tip - pip, mcp - pip
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        features.append(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)) if n1 > 1e-6 and n2 > 1e-6 else 0.0)

    for i in range(5):
        for j in range(i + 1, 5):
            features.append(float(np.linalg.norm(lms[TIPS[i]] - lms[TIPS[j]])))

    for tip in TIPS:
        features.append(float(lms[tip][1]))

    palm = np.mean(lms[[0, 5, 9, 13, 17]], axis=0)
    for tip in TIPS:
        features.append(float(np.linalg.norm(lms[tip] - palm)))

    thumb, idx_mcp, mid_mcp = lms[4], lms[5], lms[9]
    features += [float(thumb[0] - idx_mcp[0]), float(thumb[1] - idx_mcp[1]), float(thumb[0] - mid_mcp[0])]

    i_tip, m_tip, r_tip, p_tip = lms[8], lms[12], lms[16], lms[20]
    features += [float(np.linalg.norm(i_tip[:2] - m_tip[:2])),
                 float(np.linalg.norm(m_tip[:2] - r_tip[:2])),
                 float(np.linalg.norm(r_tip[:2] - p_tip[:2]))]

    features.append(float(lms[4][0] - lms[2][0]))
    return np.array(features, dtype="float32")


# ─────────────────────────────────────────
#  Load Model
# ─────────────────────────────────────────
print("⏳ Loading ASL model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"FuzzyLayer": FuzzyLayer,
                    "PositionalEncoding": PositionalEncoding,
                    "TransformerBlock": TransformerBlock},
    compile=False,
)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

USE_FEATURES = False
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "rb") as f:
        cfg = pickle.load(f)
    USE_FEATURES = cfg.get("version", 1) >= 2

CLASSES = list(encoder.classes_)
print(f"✅ Model loaded! Classes ({len(CLASSES)}): {CLASSES}")
print(f"   Feature Engineering: {'ON' if USE_FEATURES else 'OFF'}")


# ─────────────────────────────────────────
#  MediaPipe HandLandmarker
# ─────────────────────────────────────────
print("⏳ Loading MediaPipe...")
mp_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
    running_mode=mp_vision.RunningMode.IMAGE,
)
landmarker = HandLandmarker.create_from_options(mp_options)
print("✅ MediaPipe ready!")


# ─────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # يسمح للويبسايت تتصل بالـ server


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": len(CLASSES), "model": "FuzzyTransformer_v2"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    يستقبل صورة base64 من الويبسايت،
    يستخرج hand landmarks بـ MediaPipe،
    يرجع predicted letter + top3 + confidence
    """
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "image field مش موجود"}), 400

        # ── Decode base64 image ──
        img_b64 = data["image"]
        if "," in img_b64:          # ازالة data URL prefix
            img_b64 = img_b64.split(",", 1)[1]

        import cv2
        img_bytes = base64.b64decode(img_b64)
        img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "مش قادر يفك تشفير الصورة"}), 400

        # ── MediaPipe detection ──
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result    = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return jsonify({"hand_detected": False, "letter": None, "confidence": 0, "top3": [], "landmarks": []})

        # ── Extract & normalize landmarks ──
        hand = result.hand_landmarks[0]
        raw  = []
        for lm in hand:
            raw.extend([lm.x, lm.y, lm.z])

        normalized = normalize_landmarks(raw)

        # ── Feature engineering ──
        inp_vec = compute_hand_features(normalized) if USE_FEATURES else np.array(normalized, dtype="float32")
        inp     = np.array([inp_vec], dtype="float32")

        # ── Predict ──
        preds      = model.predict(inp, verbose=0)[0]
        class_id   = int(np.argmax(preds))
        confidence = float(preds[class_id])
        letter     = CLASSES[class_id] if confidence >= 0.50 else None

        # ── Top 3 ──
        top3_idx = np.argsort(preds)[-3:][::-1]
        top3     = [{"letter": CLASSES[i], "confidence": float(preds[i])} for i in top3_idx]

        # ── Landmarks for drawing ──
        lm_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand]

        return jsonify({
            "hand_detected": True,
            "letter":        letter,
            "confidence":    confidence,
            "top3":          top3,
            "landmarks":     lm_list,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    """
    بديل أسرع: يستقبل landmarks مباشرة (21×3 = 63 قيمة)
    بدل ما يبعت صورة كاملة
    """
    try:
        data = request.get_json(force=True)
        raw  = data.get("landmarks")  # list of 63 floats [x,y,z, x,y,z, ...]

        if not raw or len(raw) != 63:
            return jsonify({"error": "landmarks لازم تكون 63 قيمة (21 نقطة × 3)"}), 400

        normalized = normalize_landmarks(raw)
        inp_vec    = compute_hand_features(normalized) if USE_FEATURES else np.array(normalized, dtype="float32")
        inp        = np.array([inp_vec], dtype="float32")

        preds      = model.predict(inp, verbose=0)[0]
        class_id   = int(np.argmax(preds))
        confidence = float(preds[class_id])
        letter     = CLASSES[class_id] if confidence >= 0.50 else None

        top3_idx = np.argsort(preds)[-3:][::-1]
        top3     = [{"letter": CLASSES[i], "confidence": float(preds[i])} for i in top3_idx]

        return jsonify({
            "hand_detected": True,
            "letter":        letter,
            "confidence":    confidence,
            "top3":          top3,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  SignNova API Server")
    print("  http://localhost:5000")
    print("=" * 55)
    print("  افتح الويبسايت وهتلاقي الـ demo بيتوصل بالموديل")
    print("  اضغط Ctrl+C عشان توقف الـ server\n")
    app.run(host="0.0.0.0", port=5000, debug=False)