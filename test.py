# import os
# import time
# import uuid
# import json
# import math
# import numpy as np
# from pathlib import Path
# from flask import Flask, request, render_template_string, send_file, redirect, url_for
# from PyPDF2 import PdfReader
# from Crypto.Cipher import AES
# from Crypto.Util.Padding import pad, unpad

# # ---------------------- Flask App Setup ----------------------
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# # ---------------------- AES Setup ----------------------
# AES_KEY = b'ThisIsA16ByteKey'  # Must be 16 bytes for AES-128

# def aes_encrypt(plaintext_bytes):
#     cipher = AES.new(AES_KEY, AES.MODE_CBC)
#     ct_bytes = cipher.encrypt(pad(plaintext_bytes, AES.block_size))
#     return cipher.iv + ct_bytes  # Prepend IV for decryption

# def aes_decrypt(cipher_bytes):
#     iv = cipher_bytes[:16]
#     ct = cipher_bytes[16:]
#     cipher = AES.new(AES_KEY, AES.MODE_CBC, iv)
#     pt = unpad(cipher.decrypt(ct), AES.block_size)
#     return pt

# # ---------------------- Quantum-like Encryption Setup ----------------------
# H = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]], dtype=complex)
# def rx(theta):
#     return np.array([[math.cos(theta/2), -1j*math.sin(theta/2)],
#                      [-1j*math.sin(theta/2), math.cos(theta/2)]], dtype=complex)

# def apply_single_gate(state, gate, target, n):
#     shape = (2,)*n
#     psi = state.reshape(shape)
#     axes = [target] + [i for i in range(n) if i != target]
#     psi_t = np.transpose(psi, axes)
#     psi_mat = psi_t.reshape(2, -1)
#     out = gate @ psi_mat
#     out = out.reshape(psi_t.shape)
#     inv_axes = np.argsort(axes)
#     out = np.transpose(out, inv_axes)
#     return out.reshape(-1)

# def apply_cnot(state, control, target, n):
#     N = 1 << n
#     sv = state.copy()
#     for idx in range(N):
#         if ((idx >> (n-1-control)) & 1) == 1:
#             tgt_idx = idx ^ (1 << (n-1-target))
#             sv[idx], sv[tgt_idx] = sv[tgt_idx], sv[idx]
#     return sv

# def unitary_ops_for_chunk(n, rng):
#     ops = {'apply_h': [], 'rx_angles': [], 'cnot_pairs': []}
#     for q in range(n):
#         ops['apply_h'].append(bool(rng.random() > 0.5))
#         ops['rx_angles'].append(float(rng.random() * math.pi))
#     for q in range(n-1):
#         if rng.random() > 0.5:
#             ops['cnot_pairs'].append([q, q+1])
#     return ops

# def apply_ops_to_state(state, ops, n):
#     for q, do_h in enumerate(ops['apply_h']):
#         if do_h:
#             state = apply_single_gate(state, H, q, n)
#     for q, angle in enumerate(ops['rx_angles']):
#         if angle != 0.0:
#             state = apply_single_gate(state, rx(angle), q, n)
#     for c,t in ops['cnot_pairs']:
#         state = apply_cnot(state, c, t, n)
#     return state

# def apply_inverse_ops_to_state(state, ops, n):
#     for c,t in reversed(ops['cnot_pairs']):
#         state = apply_cnot(state, c, t, n)
#     for q, angle in reversed(list(enumerate(ops['rx_angles']))):
#         if angle != 0.0:
#             state = apply_single_gate(state, rx(-angle), q, n)
#     for q, do_h in reversed(list(enumerate(ops['apply_h']))):
#         if do_h:
#             state = apply_single_gate(state, H, q, n)
#     return state

# def reversible_quantum_encode(bitstring: str, chunk_size=8, rng_seed=None):
#     chunks = [bitstring[i:i+chunk_size] for i in range(0, len(bitstring), chunk_size)]
#     rng = np.random.default_rng(rng_seed)
#     statevectors = []
#     keys = []
#     t0 = time.perf_counter()
#     for ch in chunks:
#         n = len(ch)
#         if n < chunk_size:
#             ch += "0"*(chunk_size - n)
#             n = chunk_size
#         N = 1 << n
#         basis_index = int(ch, 2)
#         psi = np.zeros(N, dtype=complex)
#         psi[basis_index] = 1.0 + 0j
#         ops = unitary_ops_for_chunk(n, rng)
#         psi_final = apply_ops_to_state(psi, ops, n)
#         statevectors.append(psi_final)
#         keys.append({'n': n, 'ops': ops, 'orig_len': len(ch.rstrip('0')) if ch.rstrip('0') else chunk_size})
#     elapsed = time.perf_counter() - t0
#     return statevectors, keys, elapsed

# def reversible_quantum_decrypt(statevectors, keys):
#     recovered_chunks = []
#     t0 = time.perf_counter()
#     for sv, key in zip(statevectors, keys):
#         n = key['n']
#         ops = key['ops']
#         orig_len = key['orig_len']
#         psi_recovered = apply_inverse_ops_to_state(sv, ops, n)
#         idx = int(np.argmax(np.abs(psi_recovered)))
#         bits = format(idx, f"0{n}b")
#         bits = bits[:orig_len]
#         recovered_chunks.append(bits)
#     elapsed = time.perf_counter() - t0
#     full_bitstring = ''.join(recovered_chunks)
#     return full_bitstring, elapsed

# def bytes_to_bitstring(b: bytes) -> str:
#     return ''.join(f"{byte:08b}" for byte in b)

# def bitstring_to_bytes(bitstr: str) -> bytes:
#     pad = (-len(bitstr)) % 8
#     if pad:
#         bitstr += "0"*pad
#     out = bytearray()
#     for i in range(0, len(bitstr), 8):
#         out.append(int(bitstr[i:i+8],2))
#     return bytes(out)

# # ---------------------- PDF Text Extraction ----------------------
# def extract_text_from_pdf(path):
#     reader = PdfReader(path)
#     if reader.is_encrypted:
#         try: reader.decrypt("")
#         except: pass
#     parts = []
#     for p in reader.pages:
#         try: parts.append(p.extract_text() or "")
#         except: parts.append("")
#     return "\n".join(parts)

# # ---------------------- Flask HTML ----------------------
# HTML = '''
# <!doctype html>
# <title>PDF AES & Quantum Encoder/Decoder</title>
# <h2>Upload PDF to Encrypt/Decrypt (Classical AES & Quantum)</h2>

# <form method=post enctype=multipart/form-data action="/process">
#   <input type=file name=file accept="application/pdf" required>
#   <br><br>
#   <label>Quantum chunk size (default 8 qubits):</label>
#   <input type=number name=chunk_size value=8 min=2 max=12>
#   <br><br>
#   <button type=submit name=method value="aes">Classical AES Encrypt</button>
#   <button type=submit name=method value="quantum">Quantum Sim Encrypt</button>
# </form>

# {% if result %}
#   <hr>
#   <h3>Results for {{ result.filename }}</h3>
#   <ul>
#     <li>Original size: {{ result.filesize }} bytes</li>
#     <li>Encryption time: {{ result.enc_time }} s</li>
#     <li>Decryption time: {{ result.dec_time }} s</li>
#     <li>Output file: <a href="/download/{{ result.out_file }}">{{ result.out_file }}</a></li>
#   </ul>
# {% endif %}
# '''

# # ---------------------- Flask Routes ----------------------
# @app.route('/')
# def index(): return render_template_string(HTML, result=None)

# @app.route('/process', methods=['POST'])
# def process():
#     uploaded = request.files.get('file')
#     method = request.form.get('method')
#     chunk_size = int(request.form.get('chunk_size', 8))
#     if not uploaded or not method: return redirect(url_for('index'))

#     filename = f"{uuid.uuid4().hex}_{uploaded.filename}"
#     saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     uploaded.save(saved_path)

#     text = extract_text_from_pdf(saved_path)
#     data_bytes = text.encode('utf-8')
#     filesize = len(data_bytes)

#     if method == 'aes':
#         # AES classical encryption
#         t0_enc = time.perf_counter()
#         enc_bytes = aes_encrypt(data_bytes)
#         t_enc = time.perf_counter() - t0_enc
#         enc_name = f"{filename}.aes.bin"
#         enc_path = os.path.join(app.config['UPLOAD_FOLDER'], enc_name)
#         with open(enc_path, 'wb') as f: f.write(enc_bytes)

#         # Decrypt
#         t0_dec = time.perf_counter()
#         dec_bytes = aes_decrypt(enc_bytes)
#         t_dec = time.perf_counter() - t0_dec

#         out_file = enc_name

#     elif method == 'quantum':
#         # Quantum-like simulation
#         bitstr = bytes_to_bitstring(data_bytes)
#         t0_enc = time.perf_counter()
#         statevectors, keys, t_enc = reversible_quantum_encode(bitstr, chunk_size)
#         # Save quantum package
#         qname = f"{filename}.qbin.npz"
#         qpath = os.path.join(app.config['UPLOAD_FOLDER'], qname)
#         arrays = {f"v{i}": sv for i, sv in enumerate(statevectors)}
#         np.savez_compressed(qpath, __keys__=json.dumps(keys), **arrays)

#         # Decrypt
#         loaded = np.load(qpath, allow_pickle=True)
#         sv_list = [loaded[f'v{i}'] for i in range(len(loaded.files)-1)]
#         keys_loaded = json.loads(loaded['__keys__'].tolist())
#         recovered_bitstr, t_dec = reversible_quantum_decrypt(sv_list, keys_loaded)
#         dec_bytes = bitstring_to_bytes(recovered_bitstr)

#         out_file = qname

#     else:
#         return "Invalid method", 400

#     result = {
#         'filename': uploaded.filename,
#         'filesize': filesize,
#         'enc_time': round(t_enc, 6),
#         'dec_time': round(t_dec, 6),
#         'out_file': out_file
#     }

#     return render_template_string(HTML, result=result)

# @app.route('/download/<name>')
# def download(name):
#     path = os.path.join(app.config['UPLOAD_FOLDER'], name)
#     if not os.path.exists(path):
#         return 'File not found', 404
#     return send_file(path, as_attachment=True)

# # ---------------------- Run App ----------------------
# if __name__ == '__main__':
#     print("🚀 Starting Flask AES & Quantum Encoder/Decoder at http://127.0.0.1:5000")
#     app.run(debug=True)






import os
import time
import uuid
import json
import math
import numpy as np
from pathlib import Path
from flask import Flask, request, render_template_string, send_file, redirect, url_for
from PyPDF2 import PdfReader
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# ---------------------- Flask App Setup ----------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# ---------------------- AES Setup ----------------------
AES_KEY = b"thisis32byteslongkeyforaes256!!!"  # 32 bytes key for AES-256
AES_BLOCK_SIZE = 16

def aes_encrypt(plaintext_bytes):
    cipher = AES.new(AES_KEY, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plaintext_bytes, AES_BLOCK_SIZE))
    return cipher.iv + ct_bytes

def aes_decrypt(ciphertext_bytes):
    iv = ciphertext_bytes[:AES_BLOCK_SIZE]
    ct = ciphertext_bytes[AES_BLOCK_SIZE:]
    cipher = AES.new(AES_KEY, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ct), AES_BLOCK_SIZE)
    return pt

# ---------------------- Quantum Setup ----------------------
H = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]], dtype=complex)
def rx(theta):
    return np.array([[math.cos(theta/2), -1j*math.sin(theta/2)],
                     [-1j*math.sin(theta/2), math.cos(theta/2)]], dtype=complex)

def apply_single_gate(state, gate, target, n):
    shape = (2,)*n
    psi = state.reshape(shape)
    axes = [target] + [i for i in range(n) if i != target]
    psi_t = np.transpose(psi, axes)
    psi_mat = psi_t.reshape(2, -1)
    out = gate @ psi_mat
    out = out.reshape(psi_t.shape)
    inv_axes = np.argsort(axes)
    out = np.transpose(out, inv_axes)
    return out.reshape(-1)

def apply_cnot(state, control, target, n):
    N = 1<<n
    sv = state.copy()
    for idx in range(N):
        if ((idx >> (n-1-control)) & 1) == 1:
            tgt_idx = idx ^ (1 << (n-1-target))
            sv[idx], sv[tgt_idx] = sv[tgt_idx], sv[idx]
    return sv

def unitary_ops_for_chunk(n, rng):
    ops = {'apply_h': [], 'rx_angles': [], 'cnot_pairs': []}
    for q in range(n):
        ops['apply_h'].append(bool(rng.random() > 0.5))
        ops['rx_angles'].append(float(rng.random() * math.pi))
    for q in range(n-1):
        if rng.random() > 0.5:
            ops['cnot_pairs'].append([q, q+1])
    return ops

def apply_ops_to_state(state, ops, n):
    for q, do_h in enumerate(ops['apply_h']):
        if do_h:
            state = apply_single_gate(state, H, q, n)
    for q, angle in enumerate(ops['rx_angles']):
        if angle != 0.0:
            state = apply_single_gate(state, rx(angle), q, n)
    for c,t in ops['cnot_pairs']:
        state = apply_cnot(state, c, t, n)
    return state

def apply_inverse_ops_to_state(state, ops, n):
    for c,t in reversed(ops['cnot_pairs']):
        state = apply_cnot(state, c, t, n)
    for q, angle in reversed(list(enumerate(ops['rx_angles']))):
        if angle != 0.0:
            state = apply_single_gate(state, rx(-angle), q, n)
    for q, do_h in reversed(list(enumerate(ops['apply_h']))):
        if do_h:
            state = apply_single_gate(state, H, q, n)
    return state

def reversible_quantum_encode(bitstring: str, chunk_size=8, rng_seed=None):
    chunks = [bitstring[i:i+chunk_size] for i in range(0, len(bitstring), chunk_size)]
    rng = np.random.default_rng(rng_seed)
    statevectors = []
    keys = []
    t0 = time.perf_counter()
    for ch in chunks:
        n = len(ch)
        if n < chunk_size:
            ch += '0'*(chunk_size-n)
            n = chunk_size
        N = 1 << n
        psi = np.zeros(N, dtype=complex)
        basis_index = int(ch, 2)
        psi[basis_index] = 1.0 + 0j
        ops = unitary_ops_for_chunk(n, rng)
        psi_final = apply_ops_to_state(psi, ops, n)
        statevectors.append(psi_final)
        keys.append({'n': n, 'ops': ops, 'orig_len': len(ch.rstrip('0')) if ch.rstrip('0')!='' else chunk_size})
    elapsed = time.perf_counter() - t0
    return statevectors, keys, elapsed, len(chunks)

def reversible_quantum_decrypt(statevectors, keys):
    recovered_chunks = []
    t0 = time.perf_counter()
    for sv, key in zip(statevectors, keys):
        n = key['n']
        ops = key['ops']
        orig_len = key.get('orig_len', n)
        psi_recovered = apply_inverse_ops_to_state(sv, ops, n)
        idx = int(np.argmax(np.abs(psi_recovered)))
        bits = format(idx, f"0{n}b")[:orig_len]
        recovered_chunks.append(bits)
    elapsed = time.perf_counter() - t0
    full_bitstring = ''.join(recovered_chunks)
    return full_bitstring, elapsed

def save_quantum_package(statevectors, keys, outpath):
    arrays = {f"v{i}": sv for i, sv in enumerate(statevectors)}
    keys_json = json.dumps(keys)
    np.savez_compressed(outpath, __keys__=keys_json, **arrays)

def load_quantum_package(path):
    z = np.load(path, allow_pickle=True)
    keys = json.loads(z['__keys__'].tolist())
    statevectors = [z[f'v{i}'] for i in range(len(z.files)-1)]
    return statevectors, keys

# ---------------------- PDF & Bitstring Helpers ----------------------
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as e:
            raise RuntimeError("Encrypted PDF — cannot decrypt: "+str(e))
    parts = []
    for p in reader.pages:
        try:
            parts.append(p.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)

def text_to_bytes(text: str) -> bytes:
    return text.encode('utf-8')

def bytes_to_bitstring(b: bytes) -> str:
    return ''.join(f"{byte:08b}" for byte in b)

def bitstring_to_bytes(bitstr: str) -> bytes:
    pad = (-len(bitstr)) % 8
    if pad:
        bitstr += "0"*pad
    out = bytearray()
    for i in range(0, len(bitstr), 8):
        out.append(int(bitstr[i:i+8], 2))
    return bytes(out)

# ---------------------- HTML ----------------------
HTML = '''
<!doctype html>
<title>PDF AES / Quantum Encryptor</title>
<h2>Encrypt PDF</h2>
<form method=post enctype=multipart/form-data action="/process">
  <input type=file name=file accept="application/pdf" required><br><br>
  <label>Quantum chunk size (default 8 qubits):</label>
  <input type=number name=chunk_size value=8 min=2 max=12><br><br>
  <button type=submit name=method value="aes">Classical AES Encrypt</button>
  <button type=submit name=method value="quantum">Quantum Encrypt</button>
</form>

<hr>

<h2>Decrypt File</h2>
<form method=post enctype=multipart/form-data action="/decrypt">
  <input type=file name=enc_file required><br><br>
  <label>Method:</label>
  <select name="method">
    <option value="aes">AES</option>
    <option value="quantum">Quantum</option>
  </select><br><br>
  <button type=submit>Decrypt</button>
</form>

{% if result %}
<hr>
<h3>Encrypted File Info for {{ result.filename }}</h3>
<ul>
  <li>File size: {{ result.filesize }} bytes</li>
  <li>Encryption time: {{ result.time }} s</li>
  <li>Encrypted file: <a href="/download/{{ result.out_file }}">{{ result.out_file }}</a></li>
</ul>
{% endif %}

{% if decrypt_result %}
<hr>
<h3>Decrypted Text</h3>
<pre>{{ decrypt_result.text }}</pre>
{% endif %}
'''

# ---------------------- Flask Routes ----------------------
@app.route('/')
def index_page():
    return render_template_string(HTML, result=None, decrypt_result=None)

@app.route('/process', methods=['POST'])
def process_file():
    uploaded = request.files.get("file")
    method = request.form.get("method")
    chunk_size = int(request.form.get("chunk_size", 8))
    if not uploaded or method not in ['aes','quantum']:
        return redirect(url_for('index_page'))

    filename_key = f"{uuid.uuid4().hex}_{uploaded.filename}"
    saved_path = os.path.join(UPLOAD_FOLDER, filename_key)
    uploaded.save(saved_path)

    text = extract_text_from_pdf(saved_path)
    plaintext_bytes = text_to_bytes(text)
    t0 = time.perf_counter()
    if method=='aes':
        encrypted_bytes = aes_encrypt(plaintext_bytes)
        out_file = f"{filename_key}.aes.bin"
        with open(os.path.join(UPLOAD_FOLDER,out_file),'wb') as f:
            f.write(encrypted_bytes)
    else:
        bitstr = bytes_to_bitstring(plaintext_bytes)
        statevectors, keys, t_enc, _ = reversible_quantum_encode(bitstr, chunk_size=chunk_size)
        out_file = f"{filename_key}.qenc.npz"
        save_quantum_package(statevectors, keys, os.path.join(UPLOAD_FOLDER,out_file))
    elapsed = time.perf_counter() - t0

    return render_template_string(HTML, result={'filename':uploaded.filename,
                                                'filesize':os.path.getsize(saved_path),
                                                'time':round(elapsed,6),
                                                'out_file':out_file},
                                  decrypt_result=None)

@app.route('/decrypt', methods=['POST'])
def decrypt_file():
    enc_file = request.files.get("enc_file")
    method = request.form.get("method")
    if not enc_file or method not in ['aes','quantum']:
        return redirect(url_for('index_page'))

    filename_key = f"{uuid.uuid4().hex}_{enc_file.filename}"
    saved_path = os.path.join(UPLOAD_FOLDER, filename_key)
    enc_file.save(saved_path)

    try:
        if method=='aes':
            with open(saved_path,'rb') as f:
                enc_bytes = f.read()
            decrypted_bytes = aes_decrypt(enc_bytes)
        else:
            statevectors, keys = load_quantum_package(saved_path)
            recovered_bitstr,_ = reversible_quantum_decrypt(statevectors, keys)
            decrypted_bytes = bitstring_to_bytes(recovered_bitstr)

        decrypted_text = decrypted_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        return f"<h3>Error during decryption: {e}</h3><a href='/'>Back</a>"

    return render_template_string(HTML, result=None, decrypt_result={'text':decrypted_text})

@app.route('/download/<name>')
def download_file(name):
    path = os.path.join(UPLOAD_FOLDER, name)
    if not os.path.exists(path):
        return "File not found",404
    return send_file(path, as_attachment=True)

# ---------------------- Run App ----------------------
if __name__ == '__main__':
    print("🚀 Starting Flask app at http://127.0.0.1:5000")
    app.run(debug=True)
