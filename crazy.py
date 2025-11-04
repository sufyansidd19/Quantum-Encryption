from flask import Flask, render_template_string, request, send_file
import os
import time
import base64
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- Utility Functions ----------
def pad_key(key):
    """Ensure AES key is 16, 24, or 32 bytes."""
    key = key.encode("utf-8")
    if len(key) not in (16, 24, 32):
        key = key.ljust(32, b"0")[:32]
    return key

def classical_encrypt(data, key):
    key = pad_key(key)
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return cipher.nonce + tag + ciphertext

def classical_decrypt(data, key):
    key = pad_key(key)
    nonce, tag, ciphertext = data[:16], data[16:32], data[32:]
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)

def quantum_like_encrypt(data, chunk_size=8):
    """Simulate quantum-like reversible encryption using XOR."""
    np.random.seed(42)  # deterministic for decryption
    key_stream = np.random.randint(0, 256, len(data), dtype=np.uint8)
    encrypted = np.bitwise_xor(np.frombuffer(data, dtype=np.uint8), key_stream)
    return encrypted.tobytes()

def quantum_like_decrypt(data, chunk_size=8):
    """Same XOR operation decrypts exactly."""
    np.random.seed(42)
    key_stream = np.random.randint(0, 256, len(data), dtype=np.uint8)
    decrypted = np.bitwise_xor(np.frombuffer(data, dtype=np.uint8), key_stream)
    return decrypted.tobytes()

# ---------- Flask Routes ----------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Quantum vs Classical Encryption</title>
    <style>
        body { font-family: Arial; margin: 40px; background-color: #f9f9f9; }
        h2 { color: #333; }
        form { margin-bottom: 40px; padding: 20px; background: #fff; border-radius: 10px; width: 450px; }
        input, select, button { margin: 8px 0; padding: 8px; width: 100%; }
        .result { margin-top: 20px; background: #e9ffe9; padding: 15px; border-radius: 10px; }
    </style>
</head>
<body>
    <h2>🔐 Quantum vs Classical Encryption</h2>
    <form action="/encrypt" method="post" enctype="multipart/form-data">
        <label>Upload File:</label>
        <input type="file" name="file" required>

        <label>Encryption Type:</label>
        <select name="enc_type" required>
            <option value="classical">Classical (AES)</option>
            <option value="quantum">Quantum-like (Reversible XOR)</option>
        </select>

        <label>AES Key (for Classical only):</label>
        <input type="text" name="aes_key" placeholder="Enter AES key (16–32 chars)">

        <label>Quantum Chunk Size (for Quantum only):</label>
        <input type="number" name="chunk_size" value="8">

        <button type="submit">Encrypt 🔒</button>
    </form>

    <form action="/decrypt" method="post" enctype="multipart/form-data">
        <label>Upload Encrypted (.bin) File:</label>
        <input type="file" name="file" required>

        <label>Decryption Type:</label>
        <select name="dec_type" required>
            <option value="classical">Classical (AES)</option>
            <option value="quantum">Quantum-like (Reversible XOR)</option>
        </select>

        <label>AES Key (for Classical only):</label>
        <input type="text" name="aes_key" placeholder="Enter AES key (same as used in encryption)">

        <label>Quantum Chunk Size (for Quantum only):</label>
        <input type="number" name="chunk_size" value="8">

        <button type="submit">Decrypt 🔓</button>
    </form>

    {% if result %}
        <div class="result">
            <strong>{{ result }}</strong><br>
            {% if download_link %}
                <a href="{{ download_link }}">⬇️ Download Result</a>
            {% endif %}
        </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/encrypt", methods=["POST"])
def encrypt():
    file = request.files["file"]
    enc_type = request.form["enc_type"]
    aes_key = request.form.get("aes_key", "")
    chunk_size = int(request.form.get("chunk_size", 8))
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    with open(filepath, "rb") as f:
        data = f.read()

    start = time.time()
    try:
        if enc_type == "classical":
            encrypted = classical_encrypt(data, aes_key)
        else:
            encrypted = quantum_like_encrypt(data, chunk_size)
        duration = time.time() - start
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, result=f"Error: {e}")

    out_path = os.path.join(UPLOAD_FOLDER, file.filename + f"_{enc_type}.bin")
    with open(out_path, "wb") as f:
        f.write(encrypted)

    result = f"{enc_type.capitalize()} encryption done in {duration:.4f} sec"
    return render_template_string(HTML_TEMPLATE, result=result, download_link=f"/download/{os.path.basename(out_path)}")

@app.route("/decrypt", methods=["POST"])
def decrypt():
    file = request.files["file"]
    dec_type = request.form["dec_type"]
    aes_key = request.form.get("aes_key", "")
    chunk_size = int(request.form.get("chunk_size", 8))
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    with open(filepath, "rb") as f:
        data = f.read()

    start = time.time()
    try:
        if dec_type == "classical":
            decrypted = classical_decrypt(data, aes_key)
        else:
            decrypted = quantum_like_decrypt(data, chunk_size)
        duration = time.time() - start
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, result=f"Error during decryption: {e}")

    # Save decrypted file with _decrypted suffix
    original_name = os.path.splitext(file.filename)[0]
    out_path = os.path.join(UPLOAD_FOLDER, f"{original_name}_decrypted.txt")
    with open(out_path, "wb") as f:
        f.write(decrypted)

    result = f"{dec_type.capitalize()} decryption successful in {duration:.4f} sec"
    return render_template_string(HTML_TEMPLATE, result=result, download_link=f"/download/{os.path.basename(out_path)}")

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
