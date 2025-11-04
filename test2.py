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

# # ---------------- App Config ----------------
# UPLOAD_FOLDER = "uploads"
# Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
# MAX_CHUNK_QUBITS = 12
# DEFAULT_CHUNK_QUBITS = 8

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # ---------------- HTML Template ----------------
# HTML = """
# <!doctype html>
# <title>PDF Classical & Quantum Encoder/Decoder</title>
# <h2>Upload PDF — Classical AES or Quantum Reversible</h2>

# <form method=post enctype=multipart/form-data action="/classical_encode">
#   <h3>Classical AES Encode</h3>
#   <input type=file name=file accept="application/pdf" required>
#   <br><br>
#   <input type=text name=key placeholder="AES Key (16/24/32 chars)" required>
#   <br><br>
#   <input type=submit value="Encrypt Classical">
# </form>

# <form method=post enctype=multipart/form-data action="/classical_decrypt">
#   <h3>Classical AES Decrypt</h3>
#   <input type=file name=encfile accept=".bin" required>
#   <br><br>
#   <input type=text name=key placeholder="AES Key (16/24/32 chars)" required>
#   <br><br>
#   <input type=submit value="Decrypt Classical">
# </form>

# <hr>

# <form method=post enctype=multipart/form-data action="/quantum_encode">
#   <h3>Quantum Reversible Encode</h3>
#   <input type=file name=file accept="application/pdf" required>
#   <br><br>
#   Chunk size (qubits per chunk, 2..{{max}}): 
#   <input type=number name=chunk_size value="{{default}}" min=2 max="{{max}}">
#   <br><br>
#   <input type=submit value="Encrypt Quantum">
# </form>

# <form method=post enctype=multipart/form-data action="/quantum_decrypt">
#   <h3>Quantum Reversible Decrypt</h3>
#   <input type=file name=qenc accept=".qenc.npz" required>
#   <br><br>
#   <input type=submit value="Decrypt Quantum">
# </form>

# {% if result %}
#   <hr>
#   <h3>Results for {{ result.filename }}</h3>
#   <ul>
#     {% if result.classical %}
#       <li>Classical file: <a href="/download/{{ result.classical_out }}">{{ result.classical_out }}</a></li>
#       <li>Time: {{ result.time }} s</li>
#     {% endif %}
#     {% if result.quantum %}
#       <li>Quantum reversible package: <a href="/download/{{ result.quantum_out }}">{{ result.quantum_out }}</a></li>
#       <li>Time: {{ result.time }} s | Chunks: {{ result.chunks }} | Chunk size: {{ result.chunk_size }}</li>
#     {% endif %}
#   </ul>
# {% endif %}
# """

# # ---------------- Helpers: PDF / Classical AES ----------------
# def extract_text_from_pdf(path):
#     reader = PdfReader(path)
#     if reader.is_encrypted:
#         try:
#             reader.decrypt("")
#         except Exception as e:
#             raise RuntimeError("Encrypted PDF — cannot decrypt: " + str(e))
#     parts = []
#     for p in reader.pages:
#         try:
#             parts.append(p.extract_text() or "")
#         except Exception:
#             parts.append("")
#     return "\n".join(parts)

# def aes_encrypt_bytes(data: bytes, key: bytes) -> bytes:
#     cipher = AES.new(key, AES.MODE_CBC)
#     ct_bytes = cipher.encrypt(pad(data, AES.block_size))
#     return cipher.iv + ct_bytes  # prepend IV

# def aes_decrypt_bytes(enc_data: bytes, key: bytes) -> bytes:
#     iv = enc_data[:AES.block_size]
#     ct = enc_data[AES.block_size:]
#     cipher = AES.new(key, AES.MODE_CBC, iv)
#     pt = unpad(cipher.decrypt(ct), AES.block_size)
#     return pt

# # ---------------- Helpers: Quantum Reversible ----------------
# H = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]], dtype=complex)
# def rx(theta):
#     return np.array([[math.cos(theta/2), -1j*math.sin(theta/2)],
#                      [-1j*math.sin(theta/2), math.cos(theta/2)]], dtype=complex)

# def apply_single_gate(state, gate, target, n):
#     shape = (2,)*n
#     psi = state.reshape(shape)
#     axes = [target]+[i for i in range(n) if i!=target]
#     psi_t = np.transpose(psi, axes)
#     psi_mat = psi_t.reshape(2,-1)
#     out = gate @ psi_mat
#     out = out.reshape(psi_t.shape)
#     inv_axes = np.argsort(axes)
#     out = np.transpose(out, inv_axes)
#     return out.reshape(-1)

# def apply_cnot(state, control, target, n):
#     N = 1<<n
#     sv = state.copy()
#     for idx in range(N):
#         if ((idx>>(n-1-control))&1)==1:
#             tgt_idx = idx^(1<<(n-1-target))
#             sv[idx], sv[tgt_idx] = sv[tgt_idx], sv[idx]
#     return sv

# def unitary_ops_for_chunk(n,rng):
#     ops={'apply_h':[],'rx_angles':[],'cnot_pairs':[]}
#     for q in range(n):
#         ops['apply_h'].append(bool(rng.random()>0.5))
#         ops['rx_angles'].append(float(rng.random()*math.pi))
#     for q in range(n-1):
#         if rng.random()>0.5:
#             ops['cnot_pairs'].append([q,q+1])
#     return ops

# def apply_ops_to_state(state, ops, n):
#     for q,do_h in enumerate(ops['apply_h']):
#         if do_h:
#             state = apply_single_gate(state,H,q,n)
#     for q,angle in enumerate(ops['rx_angles']):
#         if angle!=0.0:
#             state = apply_single_gate(state,rx(angle),q,n)
#     for c,t in ops['cnot_pairs']:
#         state = apply_cnot(state,c,t,n)
#     return state

# def apply_inverse_ops_to_state(state, ops, n):
#     for c,t in reversed(ops['cnot_pairs']):
#         state = apply_cnot(state,c,t,n)
#     for q,angle in reversed(list(enumerate(ops['rx_angles']))):
#         if angle!=0.0:
#             state = apply_single_gate(state,rx(-angle),q,n)
#     for q,do_h in reversed(list(enumerate(ops['apply_h']))):
#         if do_h:
#             state = apply_single_gate(state,H,q,n)
#     return state

# def reversible_quantum_encode(bitstring:str, chunk_size=DEFAULT_CHUNK_QUBITS, rng_seed=None):
#     if chunk_size<1 or chunk_size>MAX_CHUNK_QUBITS:
#         raise RuntimeError("chunk_size out of range")
#     chunks = [bitstring[i:i+chunk_size] for i in range(0,len(bitstring),chunk_size)]
#     rng=np.random.default_rng(rng_seed)
#     statevectors=[]
#     keys=[]
#     t0=time.perf_counter()
#     for ch in chunks:
#         n=len(ch)
#         if n<chunk_size:
#             ch+='0'*(chunk_size-n)
#             n=chunk_size
#         N=1<<n
#         psi=np.zeros(N,dtype=complex)
#         psi[int(ch,2)]=1.0+0j
#         ops=unitary_ops_for_chunk(n,rng)
#         psi_final=apply_ops_to_state(psi,ops,n)
#         statevectors.append(psi_final)
#         keys.append({'n':n,'ops':ops,'orig_len':len(ch.rstrip('0')) if ch.rstrip('0')!='' else chunk_size})
#     elapsed=time.perf_counter()-t0
#     return statevectors, keys, elapsed, len(chunks)

# def save_quantum_package(statevectors, keys, outpath):
#     arrays={f'v{i}':sv for i,sv in enumerate(statevectors)}
#     keys_json=json.dumps(keys)
#     np.savez_compressed(outpath,__keys_json__=keys_json,**arrays)

# def load_quantum_package(path):
#     z=np.load(path, allow_pickle=True)
#     keys=json.loads(z['__keys_json__'].tolist())
#     statevectors=[]
#     i=0
#     while f'v{i}' in z:
#         statevectors.append(z[f'v{i}'])
#         i+=1
#     return statevectors, keys

# def reversible_quantum_decrypt(statevectors, keys):
#     recovered=[]
#     t0=time.perf_counter()
#     for sv,key in zip(statevectors, keys):
#         n=key['n']
#         ops=key['ops']
#         orig_len=key.get('orig_len',n)
#         psi_recovered=apply_inverse_ops_to_state(sv,ops,n)
#         idx=int(np.argmax(np.abs(psi_recovered)))
#         bits=format(idx,f"0{n}b")[:orig_len]
#         recovered.append(bits)
#     elapsed=time.perf_counter()-t0
#     return ''.join(recovered), elapsed

# # ---------------- Flask Routes ----------------
# @app.route('/')
# def index():
#     return render_template_string(HTML, default=DEFAULT_CHUNK_QUBITS, max=MAX_CHUNK_QUBITS, result=None)

# # --- Classical AES Encode ---
# @app.route('/classical_encode', methods=['POST'])
# def classical_encode():
#     uploaded=request.files.get('file')
#     key_str=request.form.get('key')
#     if not uploaded or not key_str:
#         return redirect(url_for('index'))
#     if len(key_str) not in [16,24,32]:
#         return "<h3>Key must be 16/24/32 bytes!</h3>"
#     key=key_str.encode()
#     filename=f"{uuid.uuid4().hex}_{uploaded.filename}"
#     saved_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     uploaded.save(saved_path)
#     data=extract_text_from_pdf(saved_path).encode()
#     t0=time.perf_counter()
#     encrypted=aes_encrypt_bytes(data,key)
#     elapsed=time.perf_counter()-t0
#     out_name=filename+".bin"
#     out_path=os.path.join(app.config['UPLOAD_FOLDER'],out_name)
#     with open(out_path,"wb") as f:
#         f.write(encrypted)
#     result={'filename':uploaded.filename,'classical':True,'time':round(elapsed,6),'classical_out':out_name}
#     return render_template_string(HTML,result=result,default=DEFAULT_CHUNK_QUBITS,max=MAX_CHUNK_QUBITS)

# # --- Classical AES Decrypt ---
# @app.route('/classical_decrypt', methods=['POST'])
# def classical_decrypt():
#     uploaded=request.files.get('encfile')
#     key_str=request.form.get('key')
#     if not uploaded or not key_str:
#         return redirect(url_for('index'))
#     if len(key_str) not in [16,24,32]:
#         return "<h3>Key must be 16/24/32 bytes!</h3>"
#     key=key_str.encode()
#     filename=f"{uuid.uuid4().hex}_{uploaded.filename}"
#     saved_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
#     uploaded.save(saved_path)
#     try:
#         with open(saved_path,"rb") as f:
#             encrypted=f.read()
#         t0=time.perf_counter()
#         decrypted=aes_decrypt_bytes(encrypted,key)
#         elapsed=time.perf_counter()-t0
#         out_name=filename+".recovered.txt"
#         out_path=os.path.join(app.config['UPLOAD_FOLDER'],out_name)
#         with open(out_path,"wb") as f:
#             f.write(decrypted)
#     except Exception as e:
#         return f"<h3>Error during decryption: {e}</h3>"
#     result={'filename':uploaded.filename,'classical':True,'time':round(elapsed,6),'classical_out':out_name}
#     return render_template_string(HTML,result=result,default=DEFAULT_CHUNK_QUBITS,max=MAX_CHUNK_QUBITS)

# # --- Quantum Encode ---
# @app.route('/quantum_encode', methods=['POST'])
# def quantum_encode():
#     uploaded=request.files.get('file')
#     chunk_size=int(request.form.get('chunk_size',DEFAULT_CHUNK_QUBITS))
#     if not uploaded:
#         return redirect(url_for('index'))
#     filename=f"{uuid.uuid4().hex}_{uploaded.filename}"
#     saved_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
#     uploaded.save(saved_path)
#     text=extract_text_from_pdf(saved_path)
#     b=bytes_to_bitstring(text.encode())
#     t0=time.perf_counter()
#     sv, keys, elapsed, chunks = reversible_quantum_encode(b,chunk_size)
#     out_name=filename+".qenc.npz"
#     save_quantum_package(sv,keys,os.path.join(app.config['UPLOAD_FOLDER'],out_name))
#     result={'filename':uploaded.filename,'quantum':True,'time':round(elapsed,6),'quantum_out':out_name,'chunks':chunks,'chunk_size':chunk_size}
#     return render_template_string(HTML,result=result,default=DEFAULT_CHUNK_QUBITS,max=MAX_CHUNK_QUBITS)

# # --- Quantum Decrypt ---
# @app.route('/quantum_decrypt', methods=['POST'])
# def quantum_decrypt():
#     uploaded=request.files.get('qenc')
#     if not uploaded:
#         return redirect(url_for('index'))
#     filename=f"{uuid.uuid4().hex}_{uploaded.filename}"
#     saved_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
#     uploaded.save(saved_path)
#     try:
#         sv, keys = load_quantum_package(saved_path)
#         recovered_bits, elapsed = reversible_quantum_decrypt(sv,keys)
#         recovered_bytes = bitstring_to_bytes(recovered_bits)
#         out_name=filename+".recovered.txt"
#         out_path=os.path.join(app.config['UPLOAD_FOLDER'],out_name)
#         with open(out_path,"wb") as f:
#             f.write(recovered_bytes)
#     except Exception as e:
#         return f"<h3>Quantum decryption error: {e}</h3>"
#     result={'filename':uploaded.filename,'quantum':True,'time':round(elapsed,6),'quantum_out':out_name}
#     return render_template_string(HTML,result=result,default=DEFAULT_CHUNK_QUBITS,max=MAX_CHUNK_QUBITS)

# # --- Download endpoint ---
# @app.route('/download/<name>')
# def download(name):
#     path=os.path.join(app.config['UPLOAD_FOLDER'],name)
#     if not os.path.exists(path):
#         return "File not found",404
#     return send_file(path,as_attachment=True)

# # ---------------- Run App ----------------
# if __name__=="__main__":
#     print("🚀 Starting Flask app at http://127.0.0.1:5000")
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

# ---------------- App Config ----------------
UPLOAD_FOLDER = "uploads"
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
MAX_CHUNK_QUBITS = 12
DEFAULT_CHUNK_QUBITS = 8

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- HTML Template ----------------
HTML = """
<!doctype html>
<title>PDF Classical & Quantum Encoder/Decoder</title>
<h2>Upload PDF — Classical AES or Quantum Reversible</h2>

<form method=post enctype=multipart/form-data action="/classical_encode">
  <h3>Classical AES Encode</h3>
  <input type=file name=file accept="application/pdf" required>
  <br><br>
  <input type=text name=key placeholder="AES Key (16/24/32 chars)" required>
  <br><br>
  <input type=submit value="Encrypt Classical">
</form>

<form method=post enctype=multipart/form-data action="/classical_decrypt">
  <h3>Classical AES Decrypt</h3>
  <input type=file name=encfile accept=".bin" required>
  <br><br>
  <input type=text name=key placeholder="AES Key (16/24/32 chars)" required>
  <br><br>
  <input type=submit value="Decrypt Classical">
</form>

<hr>

<form method=post enctype=multipart/form-data action="/quantum_encode">
  <h3>Quantum Reversible Encode</h3>
  <input type=file name=file accept="application/pdf" required>
  <br><br>
  Chunk size (qubits per chunk, 2..{{max}}): 
  <input type=number name=chunk_size value="{{default}}" min=2 max="{{max}}">
  <br><br>
  <input type=submit value="Encrypt Quantum">
</form>

<form method=post enctype=multipart/form-data action="/quantum_decrypt">
  <h3>Quantum Reversible Decrypt</h3>
  <input type=file name=qenc accept=".qenc.npz" required>
  <br><br>
  <input type=submit value="Decrypt Quantum">
</form>

{% if result %}
  <hr>
  <h3>Results for {{ result.filename }}</h3>
  <ul>
    {% if result.classical %}
      <li>Classical file: <a href="/download/{{ result.classical_out }}">{{ result.classical_out }}</a></li>
      <li>Time: {{ result.time }} s</li>
    {% endif %}
    {% if result.quantum %}
      <li>Quantum reversible package: <a href="/download/{{ result.quantum_out }}">{{ result.quantum_out }}</a></li>
      <li>Time: {{ result.time }} s | Chunks: {{ result.chunks }} | Chunk size: {{ result.chunk_size }}</li>
    {% endif %}
  </ul>
{% endif %}
"""

# ---------------- Helpers ----------------
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    if reader.is_encrypted:
        try: reader.decrypt("")
        except: pass
    parts = []
    for p in reader.pages:
        try: parts.append(p.extract_text() or "")
        except: parts.append("")
    return "\n".join(parts)

# ---- Classical AES ----
def aes_encrypt_bytes(data: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(data, AES.block_size))
    return cipher.iv + ct_bytes

def aes_decrypt_bytes(enc_data: bytes, key: bytes) -> bytes:
    iv = enc_data[:AES.block_size]
    ct = enc_data[AES.block_size:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size)

# ---- Bitstring helpers ----
def bytes_to_bitstring(b: bytes) -> str:
    return ''.join(f"{byte:08b}" for byte in b)

def bitstring_to_bytes(bitstr: str) -> bytes:
    pad_len = (-len(bitstr)) % 8
    if pad_len: bitstr += "0"*pad_len
    return bytes(int(bitstr[i:i+8],2) for i in range(0,len(bitstr),8))

# ---- Quantum reversible helpers ----
H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]],dtype=complex)
def rx(theta): return np.array([[math.cos(theta/2),-1j*math.sin(theta/2)],[-1j*math.sin(theta/2),math.cos(theta/2)]],dtype=complex)
def apply_single_gate(state,gate,target,n):
    shape=(2,)*n; psi=state.reshape(shape)
    axes=[target]+[i for i in range(n) if i!=target]
    psi_t=np.transpose(psi,axes)
    psi_mat=psi_t.reshape(2,-1)
    out=gate@psi_mat
    out=out.reshape(psi_t.shape)
    inv_axes=np.argsort(axes)
    return np.transpose(out,inv_axes).reshape(-1)
def apply_cnot(state,control,target,n):
    N=1<<n; sv=state.copy()
    for idx in range(N):
        if ((idx>>(n-1-control))&1)==1:
            tgt_idx=idx^(1<<(n-1-target))
            sv[idx],sv[tgt_idx]=sv[tgt_idx],sv[idx]
    return sv
def unitary_ops_for_chunk(n,rng):
    ops={'apply_h':[],'rx_angles':[],'cnot_pairs':[]}
    for q in range(n): ops['apply_h'].append(bool(rng.random()>0.5)); ops['rx_angles'].append(float(rng.random()*math.pi))
    for q in range(n-1):
        if rng.random()>0.5: ops['cnot_pairs'].append([q,q+1])
    return ops
def apply_ops_to_state(state,ops,n):
    for q,do_h in enumerate(ops['apply_h']): 
        if do_h: state=apply_single_gate(state,H,q,n)
    for q,angle in enumerate(ops['rx_angles']):
        if angle!=0: state=apply_single_gate(state,rx(angle),q,n)
    for c,t in ops['cnot_pairs']: state=apply_cnot(state,c,t,n)
    return state
def apply_inverse_ops_to_state(state,ops,n):
    for c,t in reversed(ops['cnot_pairs']): state=apply_cnot(state,c,t,n)
    for q,angle in reversed(list(enumerate(ops['rx_angles']))):
        if angle!=0: state=apply_single_gate(state,rx(-angle),q,n)
    for q,do_h in reversed(list(enumerate(ops['apply_h']))):
        if do_h: state=apply_single_gate(state,H,q,n)
    return state
def reversible_quantum_encode(bitstring:str, chunk_size=DEFAULT_CHUNK_QUBITS, rng_seed=None):
    if chunk_size<1 or chunk_size>MAX_CHUNK_QUBITS: raise RuntimeError("chunk_size out of range")
    chunks=[bitstring[i:i+chunk_size] for i in range(0,len(bitstring),chunk_size)]
    rng=np.random.default_rng(rng_seed)
    statevectors=[]; keys=[]
    t0=time.perf_counter()
    for ch in chunks:
        n=len(ch)
        if n<chunk_size: ch+='0'*(chunk_size-n); n=chunk_size
        N=1<<n; psi=np.zeros(N,dtype=complex); psi[int(ch,2)]=1+0j
        ops=unitary_ops_for_chunk(n,rng)
        psi_final=apply_ops_to_state(psi,ops,n)
        statevectors.append(psi_final)
        keys.append({'n':n,'ops':ops,'orig_len':len(ch.rstrip('0')) if ch.rstrip('0')!='' else chunk_size})
    elapsed=time.perf_counter()-t0
    return statevectors, keys, elapsed, len(chunks)
def save_quantum_package(statevectors,keys,outpath):
    arrays={f'v{i}':sv for i,sv in enumerate(statevectors)}
    np.savez_compressed(outpath,__keys_json__=json.dumps(keys),**arrays)
def load_quantum_package(path):
    z=np.load(path,allow_pickle=True)
    keys=json.loads(z['__keys_json__'].tolist())
    svs=[]
    i=0
    while f'v{i}' in z: svs.append(z[f'v{i}']); i+=1
    return svs,keys
def reversible_quantum_decrypt(statevectors,keys):
    recovered=[]; t0=time.perf_counter()
    for sv,key in zip(statevectors,keys):
        n=key['n']; ops=key['ops']; orig_len=key.get('orig_len',n)
        psi=apply_inverse_ops_to_state(sv,ops,n)
        idx=int(np.argmax(np.abs(psi))); bits=format(idx,f"0{n}b")[:orig_len]
        recovered.append(bits)
    elapsed=time.perf_counter()-t0
    return ''.join(recovered), elapsed

# ---------------- Routes ----------------
@app.route('/')
def index(): return render_template_string(HTML, default=DEFAULT_CHUNK_QUBITS, max=MAX_CHUNK_QUBITS, result=None)

@app.route('/classical_encode',methods=['POST'])
def classical_encode():
    uploaded=request.files.get('file'); key_str=request.form.get('key')
    if not uploaded or not key_str: return redirect(url_for('index'))
    if len(key_str) not in [16,24,32]: return "<h3>Key must be 16/24/32 chars!</h3>"
    key=key_str.encode()
    filename=f"{uuid.uuid4().hex}_{uploaded.filename}"; saved_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
    uploaded.save(saved_path)
    data=extract_text_from_pdf(saved_path).encode()
    t0=time.perf_counter(); enc=aes_encrypt_bytes(data,key); elapsed=time.perf_counter()-t0
    out_name=filename+".bin"; out_path=os.path.join(app.config['UPLOAD_FOLDER'],out_name)
    with open(out_path,"wb") as f: f.write(enc)
    result={'filename':uploaded.filename,'classical':True,'time':round(elapsed,6),'classical_out':out_name}
    return render_template_string(HTML,result=result,default=DEFAULT_CHUNK_QUBITS,max=MAX_CHUNK_QUBITS)

@app.route('/classical_decrypt',methods=['POST'])
def classical_decrypt():
    uploaded=request.files.get('encfile'); key_str=request.form.get('key')
    if not uploaded or not key_str: return redirect(url_for('index'))
    if len(key_str) not in [16,24,32]: return "<h3>Key must be 16/24/32 chars!</h3>"
    key=key_str.encode(); filename=f"{uuid.uuid4().hex}_{uploaded.filename}"; saved_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
    uploaded.save(saved_path)
    try:
        with open(saved_path,"rb") as f: enc=f.read()
        t0=time.perf_counter(); dec=aes_decrypt_bytes(enc,key); elapsed=time.perf_counter()-t0
        out_name=filename+".recovered.txt"; out_path=os.path.join(app.config['UPLOAD_FOLDER'],out_name)
        with open(out_path,"wb") as f: f.write(dec)
    except Exception as e: return f"<h3>Decryption error: {e}</h3>"
    result={'filename':uploaded.filename,'classical':True,'time':round(elapsed,6),'classical_out':out_name}
    return render_template_string(HTML,result=result,default=DEFAULT_CHUNK_QUBITS,max=MAX_CHUNK_QUBITS)

@app.route('/quantum_encode',methods=['POST'])
def quantum_encode():
    uploaded=request.files.get('file'); chunk_size=int(request.form.get('chunk_size',DEFAULT_CHUNK_QUBITS))
    if not uploaded: return redirect(url_for('index'))
    filename=f"{uuid.uuid4().hex}_{uploaded.filename}"; saved_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
    uploaded.save(saved_path)
    text=extract_text_from_pdf(saved_path); b=bytes_to_bitstring(text.encode())
    t0=time.perf_counter(); sv,keys,elapsed,chunks=reversible_quantum_encode(b,chunk_size)
    out_name=filename+".qenc.npz"; save_quantum_package(sv,keys,os.path.join(app.config['UPLOAD_FOLDER'],out_name))
    result={'filename':uploaded.filename,'quantum':True,'time':round(elapsed,6),'quantum_out':out_name,'chunks':chunks,'chunk_size':chunk_size}
    return render_template_string(HTML,result=result,default=DEFAULT_CHUNK_QUBITS,max=MAX_CHUNK_QUBITS)

@app.route('/quantum_decrypt',methods=['POST'])
def quantum_decrypt():
    uploaded=request.files.get('qenc'); 
    if not uploaded: return redirect(url_for('index'))
    filename=f"{uuid.uuid4().hex}_{uploaded.filename}"; saved_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
    uploaded.save(saved_path)
    try:
        sv,keys=load_quantum_package(saved_path)
        bits,elapsed=reversible_quantum_decrypt(sv,keys)
        recovered_bytes=bitstring_to_bytes(bits)
        out_name=filename+".recovered.txt"; out_path=os.path.join(app.config['UPLOAD_FOLDER'],out_name)
        with open(out_path,"wb") as f: f.write(recovered_bytes)
    except Exception as e: return f"<h3>Quantum decryption error: {e}</h3>"
    result={'filename':uploaded.filename,'quantum':True,'time':round(elapsed,6),'quantum_out':out_name}
    return render_template_string(HTML,result=result,default=DEFAULT_CHUNK_QUBITS,max=MAX_CHUNK_QUBITS)

@app.route('/download/<name>')
def download(name):
    path=os.path.join(app.config['UPLOAD_FOLDER'],name)
    if not os.path.exists(path): return "File not found",404
    return send_file(path,as_attachment=True)

if __name__=="__main__":
    print("🚀 Starting Flask app at http://127.0.0.1:5000")
    app.run(debug=True)
