import os
import time
from flask import Flask, render_template_string, request, send_file
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# -------------------------- CONFIG --------------------------
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------- ENCRYPTION FUNCTIONS --------------------------
def classical_encrypt(data):
    """Standard AES encryption"""
    key = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return key + cipher.nonce + tag + ciphertext

def classical_random_encrypt(data):
    """AES with added random salt"""
    salt = get_random_bytes(8)
    data = salt + data
    key = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return key + cipher.nonce + tag + ciphertext

def classical_decrypt(data):
    """AES decryption that returns garbled text if key/tag invalid"""
    try:
        key = data[:16]
        nonce = data[16:32]
        tag = data[32:48]
        ciphertext = data[48:]
        cipher = AES.new(key, AES.MODE_EAX, nonce)
        decrypted = cipher.decrypt(ciphertext)
        return decrypted
    except Exception:
        return b"[Corrupted Data: Wrong Key]"

def quantum_encrypt(data):
    """Fast reversible XOR-based encryption"""
    key = get_random_bytes(1)[0]
    return bytes([b ^ key for b in data]) + bytes([key])

def quantum_decrypt(data):
    key = data[-1]
    data = data[:-1]
    return bytes([b ^ key for b in data])

# -------------------------- HTML TEMPLATE --------------------------
HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>⚙️ Quantum vs Classical Encryption Lab</title>
  <style>
    body {
      background: #0f172a;
      color: #e2e8f0;
      font-family: 'Segoe UI', sans-serif;
      padding: 40px;
      text-align: center;
    }
    h2 { color: #38bdf8; }
    form {
      background: #1e293b;
      padding: 20px;
      border-radius: 12px;
      display: inline-block;
      box-shadow: 0 0 20px rgba(56,189,248,0.2);
      margin-top: 20px;
    }
    label, select, input {
      margin: 10px;
      font-size: 16px;
    }
    button {
      background: #38bdf8;
      border: none;
      padding: 10px 20px;
      margin: 10px;
      border-radius: 8px;
      color: black;
      cursor: pointer;
      font-weight: bold;
      transition: 0.3s;
    }
    button:hover { background: #0ea5e9; }
    a {
      color: #22d3ee;
      text-decoration: none;
      font-weight: bold;
    }
    .result {
      background: #1e293b;
      padding: 15px;
      border-radius: 10px;
      width: 60%;
      margin: 20px auto;
    }
  </style>
</head>
<body>
  <h2>⚙️ Classical vs Quantum Encryption Lab</h2>
  <form method="post" enctype="multipart/form-data">
    <label>Select file:</label>
    <input type="file" name="file" required><br><br>
    
    <label>Encryption Type:</label>
    <select name="mode" required>
      <option value="aes">Classical AES</option>
      <option value="aes_random">Randomized AES</option>
      <option value="quantum">Quantum</option>
    </select><br><br>
    
    <button type="submit" name="action" value="encrypt">Encrypt</button>
    <button type="submit" name="action" value="decrypt">Decrypt</button>
  </form>

  {% if result %}
  <div class="result" style="color: {% if '❌' in result %}#f87171{% else %}#4ade80{% endif %};">
    <h3>{{ result }}</h3>
    {% if download_link %}
      <p><a href="{{ download_link }}">📥 Download Output</a></p>
    {% endif %}
  </div>
  {% endif %}
</body>
</html>
"""

# -------------------------- ROUTES --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    download_link = None

    if request.method == "POST":
        action = request.form["action"]
        mode = request.form["mode"]
        file = request.files["file"]

        if not file:
            return render_template_string(HTML, result="❌ No file selected.")

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)
        with open(input_path, "rb") as f:
            data = f.read()

        start = time.time()

        # -------------------- ENCRYPT --------------------
        if action == "encrypt":
            if mode == "aes":
                output_data = classical_encrypt(data)
                ext = ".saes"
                method_used = "Classical AES"
            elif mode == "aes_random":
                output_data = classical_random_encrypt(data)
                ext = ".raes"
                method_used = "Randomized AES"
            else:
                output_data = quantum_encrypt(data)
                ext = ".qenc"
                method_used = "Quantum-Inspired XOR"

            elapsed = time.time() - start
            output_path = os.path.join(UPLOAD_FOLDER, file.filename + ext)
            with open(output_path, "wb") as f:
                f.write(output_data)

            result = f"✅ Encrypted using {method_used} in {elapsed:.5f} sec."
            download_link = f"/download?path={output_path}"

        # -------------------- DECRYPT --------------------
        elif action == "decrypt":
            filename = file.filename

            # detect encryption type
            if filename.endswith(".saes"):
                expected_mode = "aes"
                method_used = "Classical AES"
                output_data = classical_decrypt(data)
            elif filename.endswith(".raes"):
                expected_mode = "aes_random"
                method_used = "Randomized AES"
                output_data = classical_decrypt(data)
            elif filename.endswith(".qenc"):
                expected_mode = "quantum"
                method_used = "Quantum-Inspired XOR"
                output_data = quantum_decrypt(data)
            else:
                result = "❌ Unsupported or unknown encrypted file type."
                return render_template_string(HTML, result=result)

            # if wrong mode used, show error instead of decrypting wrong file
            if mode != expected_mode:
                result = f"❌ Wrong decryption method! This file was encrypted using {method_used}."
                return render_template_string(HTML, result=result)

            elapsed = time.time() - start
            base_name = os.path.basename(filename)
            for ext in (".saes", ".raes", ".qenc"):
                base_name = base_name.replace(ext, "")
            output_filename = f"decrypted_{base_name}"
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)

            with open(output_path, "wb") as f:
                f.write(output_data)

            result = f"🔓 Decrypted successfully in {elapsed:.5f} sec using {method_used}."
            download_link = f"/download?path={output_path}"

    return render_template_string(HTML, result=result, download_link=download_link)

# -------------------------- DOWNLOAD ROUTE --------------------------
@app.route("/download")
def download():
    path = request.args.get("path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found."

# -------------------------- MAIN --------------------------
if __name__ == "__main__":
    print("🚀 Flask app running at: http://127.0.0.1:5000")
    app.run(debug=True)
