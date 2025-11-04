"""
PDF Classical vs Quantum Conversion Web App (Flask)

Features:
- Upload a PDF via web form
- Classical conversion: extract text (PyPDF2) -> bytes -> bitstring
- Quantum conversion (simulated, using Qiskit Aer)
- Shows exact time taken for both conversions
- Handles encrypted PDFs safely

Run:
1. pip install flask pypdf2 pycryptodome qiskit
2. python pdf_quantum_classical_converter.py
3. Open http://127.0.0.1:5000
"""

# Classical Start 



# import os
# import time
# import uuid
# from pathlib import Path
# from flask import Flask, request, render_template_string, send_file, redirect, url_for

# # --- PDF parsing ---
# try:
#     from PyPDF2 import PdfReader
# except Exception:
#     raise RuntimeError("Please install PyPDF2: pip install pypdf2")

# # --- Quantum simulation (Qiskit) ---
# use_qiskit = True
# try:
#     from qiskit import QuantumCircuit, transpile, assemble
#     from qiskit.providers.aer import AerSimulator
#     simulator = AerSimulator()
# except Exception:
#     use_qiskit = False

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# # --- HTML Interface ---
# HTML_PAGE = '''
# <!doctype html>
# <title>PDF Classical vs Quantum Converter</title>
# <h2>Upload PDF to Convert (Classical & Simulated Quantum)</h2>
# <form method=post enctype=multipart/form-data action="/convert">
#   <input type=file name=file accept="application/pdf" required>
#   <br><br>
#   <label>Quantum chunk size (qubits per simulation chunk, default 12):</label>
#   <input type=number name=chunk_size value=12 min=2 max=20>
#   <br><br>
#   <input type=submit value="Upload & Convert">
# </form>

# {% if result %}
#   <h3>Results for {{ result.filename }}</h3>
#   <ul>
#     <li>File size: {{ result.filesize }} bytes</li>
#     <li>Classical conversion time: {{ result.classical_time }} s</li>
#     <li>Classical output bytes: {{ result.classical_bytes }} bytes</li>
#     <li>Quantum simulation measured chunks: {{ result.measured_chunks }} / {{ result.total_chunks }}</li>
#     <li>Quantum simulated time (measured): {{ result.quantum_measured_time }} s</li>
#     <li>Quantum estimated total time: {{ result.quantum_estimated_time }} s</li>
#   </ul>
#   <a href="/download/{{ result.classical_out }}">Download classical bitstring (.bin)</a>
# {% endif %}
# '''

# # --------------------------- Core Functions ---------------------------

# def extract_text_from_pdf(filepath):
#     """Safely extract text, even from encrypted PDFs."""
#     reader = PdfReader(filepath)
#     if reader.is_encrypted:
#         try:
#             reader.decrypt("")  # try empty password
#         except Exception as e:
#             raise RuntimeError(f"Cannot read encrypted PDF: {e}")

#     text_parts = []
#     for page in reader.pages:
#         try:
#             text_parts.append(page.extract_text() or "")
#         except Exception:
#             text_parts.append("")
#     return "\n".join(text_parts)


# def classical_conversion(filepath):
#     """Extract text, encode to bytes, return (data_bytes, size, elapsed_time)."""
#     t0 = time.perf_counter()
#     text = extract_text_from_pdf(filepath)
#     data_bytes = text.encode('utf-8')
#     elapsed = time.perf_counter() - t0
#     return data_bytes, len(data_bytes), elapsed


# def bytes_to_bitstring(b: bytes) -> str:
#     return ''.join(f'{byte:08b}' for byte in b)


# def quantum_conversion_simulated(b: bytes, chunk_size=12, max_direct_chunks=500):
#     """
#     Simulate a 'quantum encoding' of file bits.
#     Splits bits into chunks and encodes each chunk into a quantum circuit.
#     Measures performance and estimates total time for large files.
#     """
#     if not use_qiskit:
#         raise RuntimeError("Qiskit Aer not available. Install qiskit and qiskit-aer to use quantum simulation.")

#     bitstring = bytes_to_bitstring(b)
#     total_chunks = (len(bitstring) + chunk_size - 1) // chunk_size
#     if total_chunks == 0:
#         return 0.0, 0.0, 0, 0

#     def simulate_chunk(chunk_bits):
#         n = len(chunk_bits)
#         qc = QuantumCircuit(n, n)
#         for i, bit in enumerate(reversed(chunk_bits)):  # LSB → qubit 0
#             if bit == '1':
#                 qc.x(i)
#         qc.measure(range(n), range(n))
#         t0 = time.perf_counter()
#         qc2 = transpile(qc, simulator)
#         qobj = assemble(qc2, simulator, shots=1)
#         job = simulator.run(qobj)
#         job.result()
#         return time.perf_counter() - t0

#     chunks = [bitstring[i*chunk_size:(i+1)*chunk_size] for i in range(total_chunks)]

#     # Sample if file is too large
#     if total_chunks <= max_direct_chunks:
#         measured_time = sum(simulate_chunk(chunk) for chunk in chunks)
#         estimated_total = measured_time
#         measured_chunks = total_chunks
#     else:
#         step = max(1, total_chunks // max_direct_chunks)
#         sampled_indices = list(range(0, total_chunks, step))[:max_direct_chunks]
#         measured_time = sum(simulate_chunk(chunks[i]) for i in sampled_indices)
#         avg_time = measured_time / len(sampled_indices)
#         estimated_total = avg_time * total_chunks
#         measured_chunks = len(sampled_indices)

#     return measured_time, estimated_total, measured_chunks, total_chunks

# # --------------------------- Flask Routes ---------------------------

# @app.route('/')
# def index():
#     return render_template_string(HTML_PAGE, result=None)


# @app.route('/convert', methods=['POST'])
# def convert():
#     uploaded = request.files.get('file')
#     if not uploaded:
#         return redirect(url_for('index'))

#     filename = f"{uuid.uuid4().hex}_{uploaded.filename}"
#     saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     uploaded.save(saved_path)

#     try:
#         # Classical conversion
#         classical_bytes, classical_size, classical_time = classical_conversion(saved_path)
#         classical_out_name = f"{filename}.bin"
#         classical_out_path = os.path.join(app.config['UPLOAD_FOLDER'], classical_out_name)
#         with open(classical_out_path, 'wb') as f:
#             f.write(classical_bytes)

#         # Quantum simulation
#         chunk_size = int(request.form.get('chunk_size', 12))
#         quantum_measured_time, quantum_estimated_time, measured_chunks, total_chunks = quantum_conversion_simulated(
#             classical_bytes, chunk_size=chunk_size
#         )

#     except Exception as e:
#         return f"<h3>Error: {e}</h3><a href='/'>Back</a>"

#     result = {
#         'filename': uploaded.filename,
#         'filesize': os.path.getsize(saved_path),
#         'classical_time': round(classical_time, 6),
#         'classical_bytes': classical_size,
#         'classical_out': classical_out_name,
#         'quantum_measured_time': round(quantum_measured_time, 6),
#         'quantum_estimated_time': round(quantum_estimated_time, 6),
#         'measured_chunks': measured_chunks,
#         'total_chunks': total_chunks,
#     }

#     return render_template_string(HTML_PAGE, result=result)


# @app.route('/download/<name>')
# def download(name):
#     path = os.path.join(app.config['UPLOAD_FOLDER'], name)
#     if not os.path.exists(path):
#         return 'File not found', 404
#     return send_file(path, as_attachment=True)


# if __name__ == '__main__':
#     print("Starting app at http://127.0.0.1:5000")
#     app.run(debug=True)



# Classical END 



# Quantum System 

"""
PDF Classical vs Quantum Conversion Web App (Flask)
===================================================
Features:
- Upload a PDF via web form
- Classical conversion: extract text (PyPDF2) → bytes → bitstring
- Quantum conversion: simulated encryption using Qiskit Aer
- Measures both times and allows file downloads

Run:
1. pip install flask pypdf2 pycryptodome qiskit qiskit-aer
2. python pdf_quantum_converter.py
3. Open http://127.0.0.1:5000
"""

import os
import time
import uuid
from pathlib import Path
from flask import Flask, request, render_template_string, send_file, redirect, url_for

# ---------------------- PDF Parsing ----------------------
try:
    from PyPDF2 import PdfReader
except Exception:
    raise RuntimeError("Please install PyPDF2: pip install pypdf2")

# ---------------------- Quantum Setup ----------------------
use_qiskit = True
try:
    from qiskit import QuantumCircuit, transpile, assemble
    # from qiskit_aer import AerSimulator
    from qiskit_aer.aerprovider import AerSimulator
    from qiskit.visualization import circuit_drawer
    import numpy as np
    simulator = AerSimulator()
except Exception:
    use_qiskit = False

# ---------------------- Flask App Setup ----------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# ---------------------- HTML Template ----------------------
HTML_PAGE = '''
<!doctype html>
<title>PDF Classical vs Quantum Converter</title>
<h2>Upload PDF to Convert (Classical & Quantum Simulation)</h2>
<form method=post enctype=multipart/form-data action="/convert">
  <input type=file name=file accept="application/pdf" required>
  <br><br>
  <label>Quantum chunk size (default 8 qubits):</label>
  <input type=number name=chunk_size value=8 min=2 max=20>
  <br><br>
  <input type=submit value="Upload & Convert">
</form>

{% if result %}
  <hr>
  <h3>Results for {{ result.filename }}</h3>
  <ul>
    <li>File size: {{ result.filesize }} bytes</li>
    <li>Classical conversion time: {{ result.classical_time }} s</li>
    <li>Quantum simulation time: {{ result.quantum_time }} s</li>
    <li>Quantum output bits: {{ result.quantum_bits }} bits</li>
  </ul>
  <a href="/download/{{ result.classical_out }}">Download Classical Bitstring (.bin)</a><br>
  <a href="/download/{{ result.quantum_out }}">Download Quantum Bitstring (.qbin)</a>
{% endif %}
'''

# ---------------------- Core Functions ----------------------
def extract_text_from_pdf(filepath):
    """Safely extract text from PDF (handles encryption)."""
    reader = PdfReader(filepath)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as e:
            raise RuntimeError(f"Cannot read encrypted PDF: {e}")
    text_parts = []
    for page in reader.pages:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:
            text_parts.append("")
    return "\n".join(text_parts)

def classical_conversion(filepath):
    """Extracts PDF text, converts to bytes, and returns timing."""
    t0 = time.perf_counter()
    text = extract_text_from_pdf(filepath)
    data_bytes = text.encode('utf-8')
    elapsed = time.perf_counter() - t0
    return data_bytes, len(data_bytes), elapsed

def bytes_to_bitstring(b: bytes) -> str:
    return ''.join(f'{byte:08b}' for byte in b)

# ---------------------- Quantum Conversion ----------------------
def quantum_encrypt_bits(bitstring: str, chunk_size: int = 8):
    """Simulates quantum encryption for a bitstring using Qiskit."""
    if not use_qiskit:
        raise RuntimeError("Qiskit Aer not available. Install qiskit and qiskit-aer.")

    chunks = [bitstring[i:i+chunk_size] for i in range(0, len(bitstring), chunk_size)]
    encrypted_bits = []
    t0 = time.perf_counter()

    for chunk in chunks:
        n = len(chunk)
        qc = QuantumCircuit(n, n)

        # Initialize qubits based on bits
        for i, bit in enumerate(chunk):
            if bit == '1':
                qc.x(i)

        # Apply random quantum gates as "encryption"
        rng = np.random.default_rng()
        for i in range(n):
            if rng.random() > 0.5:
                qc.h(i)
            if rng.random() > 0.5:
                qc.rx(rng.random() * 3.14, i)
            if rng.random() > 0.5:
                qc.z(i)

        # Measure
        qc.measure(range(n), range(n))
        qc2 = transpile(qc, simulator)
        qobj = assemble(qc2, simulator, shots=1)
        job = simulator.run(qobj)
        result = job.result().get_counts()

        measured_bitstring = list(result.keys())[0]
        encrypted_bits.append(measured_bitstring)

    quantum_time = time.perf_counter() - t0
    final_qbits = ''.join(encrypted_bits)
    return final_qbits, len(final_qbits), quantum_time

# ---------------------- Flask Routes ----------------------
@app.route('/')
def index():
    return render_template_string(HTML_PAGE, result=None)

@app.route('/convert', methods=['POST'])
def convert():
    uploaded = request.files.get('file')
    if not uploaded:
        return redirect(url_for('index'))

    filename = f"{uuid.uuid4().hex}_{uploaded.filename}"
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded.save(saved_path)

    try:
        # Classical conversion
        classical_bytes, classical_size, classical_time = classical_conversion(saved_path)
        classical_out_name = f"{filename}.bin"
        classical_out_path = os.path.join(app.config['UPLOAD_FOLDER'], classical_out_name)
        with open(classical_out_path, 'wb') as f:
            f.write(classical_bytes)

        # Quantum conversion
        chunk_size = int(request.form.get('chunk_size', 8))
        bitstring = bytes_to_bitstring(classical_bytes)
        quantum_bits, qlen, quantum_time = quantum_encrypt_bits(bitstring, chunk_size)

        quantum_out_name = f"{filename}.qbin"
        quantum_out_path = os.path.join(app.config['UPLOAD_FOLDER'], quantum_out_name)
        with open(quantum_out_path, 'w') as f:
            f.write(quantum_bits)

    except Exception as e:
        return f"<h3>Error: {e}</h3><a href='/'>Back</a>"

    result = {
        'filename': uploaded.filename,
        'filesize': os.path.getsize(saved_path),
        'classical_time': round(classical_time, 6),
        'quantum_time': round(quantum_time, 6),
        'quantum_bits': qlen,
        'classical_out': classical_out_name,
        'quantum_out': quantum_out_name
    }
    return render_template_string(HTML_PAGE, result=result)

@app.route('/download/<name>')
def download(name):
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    if not os.path.exists(path):
        return 'File not found', 404
    return send_file(path, as_attachment=True)

# ---------------------- Run App ----------------------
if __name__ == '__main__':
    print("🚀 Starting Flask app at http://127.0.0.1:5000")
    app.run(debug=True)
 
 
 
 
 
#Very Important   
 
 
 
"""
PDF → Classical vs Quantum Encoding Web App
===========================================
Performs two *independent* conversions from the original PDF:
1. Classical encoding (text → binary bits)
2. Quantum encoding (text → quantum superposition amplitudes)

Then compares both conversions in time and output size.

Run:
    pip install flask PyPDF2 qiskit numpy
    python pdf_classical_quantum_compare.py
    Open http://127.0.0.1:5000
"""

# import os
# import time
# import uuid
# import numpy as np
# from pathlib import Path
# from flask import Flask, request, render_template_string, send_file, redirect, url_for
# from PyPDF2 import PdfReader
# from qiskit import QuantumCircuit
# from qiskit.quantum_info import Statevector

# # ----------------------- Flask Setup -----------------------
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# # ----------------------- HTML Template -----------------------
# HTML_PAGE = '''
# <!doctype html>
# <html>
# <head>
# <title>PDF → Classical & Quantum Encoder</title>
# <style>
#   body { font-family: Arial; margin: 40px; background: #fafafa; }
#   h2 { color: #333; }
#   table { border-collapse: collapse; margin-top: 15px; width: 80%; }
#   th, td { padding: 10px; border: 1px solid #ccc; text-align: left; }
#   th { background-color: #f2f2f2; }
#   a { text-decoration: none; color: #007bff; }
#   a:hover { text-decoration: underline; }
# </style>
# </head>
# <body>

# <h2>Upload PDF for Classical & Quantum Encoding Comparison</h2>
# <form method=post enctype=multipart/form-data action="/convert">
#   <input type=file name=file accept="application/pdf" required>
#   <br><br>
#   <input type=submit value="Upload & Encode">
# </form>

# {% if result %}
#   <h3>Results for: {{ result.filename }}</h3>
#   <table>
#     <tr><th>Metric</th><th>Classical Encoding</th><th>Quantum Encoding</th></tr>
#     <tr><td>Output file</td>
#         <td><a href="/download/{{ result.classical_out }}">Download</a></td>
#         <td><a href="/download/{{ result.quantum_out }}">Download</a></td></tr>
#     <tr><td>Output size</td><td>{{ result.classical_size }} bytes</td><td>{{ result.quantum_size }} amplitudes</td></tr>
#     <tr><td>Processing time (seconds)</td>
#         <td>{{ result.classical_time }}</td>
#         <td>{{ result.quantum_time }}</td></tr>
#     <tr><td>Speed comparison</td>
#         <td colspan=2>Quantum took {{ result.speed_ratio }}× the time of classical</td></tr>
#   </table>
# {% endif %}
# </body></html>
# '''

# # ----------------------- Helper Functions -----------------------

# def extract_text_from_pdf(filepath):
#     """Extract text safely from PDF."""
#     reader = PdfReader(filepath)
#     if reader.is_encrypted:
#         try:
#             reader.decrypt("")
#         except Exception:
#             raise RuntimeError("Cannot read encrypted PDF.")

#     text = ""
#     for page in reader.pages:
#         try:
#             text += page.extract_text() or ""
#         except Exception:
#             continue
#     return text


# def classical_encoding(text: str):
#     """Classical binary encoding of text."""
#     t0 = time.perf_counter()
#     data_bytes = text.encode("utf-8")
#     bitstring = ''.join(f"{byte:08b}" for byte in data_bytes)
#     elapsed = time.perf_counter() - t0
#     return bitstring, len(data_bytes), elapsed


# def quantum_superposition_encoding(text: str):
#     """
#     Simulate quantum superposition encoding of text.
#     Each character → ASCII amplitude → normalized → statevector.
#     """
#     t0 = time.perf_counter()
#     if not text.strip():
#         raise ValueError("No text extracted from PDF.")

#     # Convert first 32 chars for simplicity (longer texts get large)
#     ascii_vals = np.array([ord(c) for c in text[:32]], dtype=float)
#     normalized = ascii_vals / np.linalg.norm(ascii_vals)

#     # Number of qubits required
#     num_qubits = int(np.ceil(np.log2(len(normalized))))
#     padded = np.zeros(2**num_qubits)
#     padded[:len(normalized)] = normalized

#     state = Statevector(padded)
#     qc = QuantumCircuit(num_qubits)
#     qc.initialize(state.data, range(num_qubits))

#     elapsed = time.perf_counter() - t0
#     return state.data, len(state.data), elapsed, qc


# # ----------------------- Flask Routes -----------------------

# @app.route('/')
# def index():
#     return render_template_string(HTML_PAGE, result=None)


# @app.route('/convert', methods=['POST'])
# def convert():
#     uploaded = request.files.get('file')
#     if not uploaded:
#         return redirect(url_for('index'))

#     filename = f"{uuid.uuid4().hex}_{uploaded.filename}"
#     saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     uploaded.save(saved_path)

#     try:
#         # Extract text
#         text = extract_text_from_pdf(saved_path)

#         # Classical encoding
#         classical_bits, classical_size, classical_time = classical_encoding(text)
#         classical_out = f"{filename}_classical.bin"
#         classical_path = os.path.join(app.config['UPLOAD_FOLDER'], classical_out)
#         with open(classical_path, "w") as f:
#             f.write(classical_bits)

#         # Quantum encoding
#         quantum_state, quantum_size, quantum_time, qc = quantum_superposition_encoding(text)
#         quantum_out = f"{filename}_quantum.npy"
#         np.save(os.path.join(app.config['UPLOAD_FOLDER'], quantum_out), quantum_state)

#         # Comparison
#         speed_ratio = round(quantum_time / classical_time, 3) if classical_time > 0 else "∞"

#     except Exception as e:
#         return f"<h3>Error: {e}</h3><a href='/'>Back</a>"

#     result = {
#         'filename': uploaded.filename,
#         'classical_out': classical_out,
#         'quantum_out': quantum_out,
#         'classical_size': classical_size,
#         'quantum_size': quantum_size,
#         'classical_time': round(classical_time, 6),
#         'quantum_time': round(quantum_time, 6),
#         'speed_ratio': speed_ratio,
#     }

#     return render_template_string(HTML_PAGE, result=result)


# @app.route('/download/<name>')
# def download(name):
#     path = os.path.join(app.config['UPLOAD_FOLDER'], name)
#     if not os.path.exists(path):
#         return 'File not found', 404
#     return send_file(path, as_attachment=True)


# # ----------------------- Main -----------------------

# if __name__ == '__main__':
#     print("🚀 App running at http://127.0.0.1:5000")
#     app.run(debug=True)



# End of very important 











"""
pdf_quantum_reversible.py

Reversible quantum-style encoding & decoding for PDF-extracted text.

- Classical: extracts text → UTF-8 bytes → .bin
- Quantum (reversible): chunks bits, apply reversible unitary per chunk (no measurement),
  store final statevectors + keys (.qenc.npz). Decrypt by applying inverse unitaries
  to recover original basis states and reconstruct original bytes.

Run:
    pip install flask pypdf2 numpy
    python pdf_quantum_reversible.py
Open http://127.0.0.1:5000
"""

import os
import time
import uuid
import json
import math
import numpy as np
from pathlib import Path
from flask import Flask, request, render_template_string, send_file, redirect, url_for
from PyPDF2 import PdfReader

# ---------------- app config ----------------
UPLOAD_FOLDER = "uploads"
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
MAX_CHUNK_QUBITS = 12
DEFAULT_CHUNK_QUBITS = 8

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- html ----------------
HTML = """
<!doctype html>
<title>PDF Reversible Quantum Encoder/Decoder</title>
<h2>Upload PDF — Encode (classical & reversible-quantum) or Decrypt</h2>

<form method=post enctype=multipart/form-data action="/encode">
  <h3>Encode (normal → classical & reversible quantum)</h3>
  <input type=file name=file accept="application/pdf" required>
  <br><br>
  Chunk size (qubits per chunk, 2..{{max}}): 
  <input type=number name=chunk_size value="{{default}}" min=2 max="{{max}}">
  <br><br>
  <button type=submit>Encode</button>
</form>

<hr>

<form method=post enctype=multipart/form-data action="/decrypt">
  <h3>Decrypt (reconstruct original from quantum encoding)</h3>
  <label>Upload .qenc.npz (quantum encoding package):</label><br>
  <input type=file name=qenc accept=".npz" required>
  <br><br>
  <button type=submit>Decrypt</button>
</form>

{% if encode_result %}
  <hr>
  <h3>Encode Results for {{ encode_result.filename }}</h3>
  <ul>
    <li>Classical .bin: <a href="/download/{{ encode_result.classical_name }}">{{ encode_result.classical_name }}</a> ({{ encode_result.classical_size }} bytes)</li>
    <li>Quantum reversible package: <a href="/download/{{ encode_result.qenc_name }}">{{ encode_result.qenc_name }}</a> (statevectors + key)</li>
    <li>Classical encode time: {{ encode_result.classical_time }} s</li>
    <li>Quantum (unitary) encode time: {{ encode_result.quantum_time }} s</li>
    <li>Chunks: {{ encode_result.chunks }} (chunk size {{ encode_result.chunk_size }})</li>
  </ul>
{% endif %}

{% if decrypt_result %}
  <hr>
  <h3>Decrypt Results</h3>
  <ul>
    <li>Recovered file: <a href="/download/{{ decrypt_result.recovered_name }}">{{ decrypt_result.recovered_name }}</a></li>
    <li>Reconstruction time: {{ decrypt_result.time }} s</li>
    <li>Recovered bytes: {{ decrypt_result.bytes }} bytes</li>
  </ul>
{% endif %}
"""

# ---------------- helpers: pdf/classical ----------------

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as e:
            raise RuntimeError("Encrypted PDF — cannot decrypt: " + str(e))
    parts = []
    for p in reader.pages:
        try:
            parts.append(p.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)

def text_to_bytes(text: str) -> bytes:
    return text.encode("utf-8")

def bytes_to_bitstring(b: bytes) -> str:
    return ''.join(f"{byte:08b}" for byte in b)

def bitstring_to_bytes(bitstr: str) -> bytes:
    pad = (-len(bitstr)) % 8
    if pad:
        bitstr = bitstr + "0"*pad
    out = bytearray()
    for i in range(0, len(bitstr), 8):
        out.append(int(bitstr[i:i+8], 2))
    return bytes(out)

# ---------------- numpy statevector utils (reversible) ----------------

H = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]], dtype=complex)
def rx(theta):
    return np.array([[math.cos(theta/2), -1j*math.sin(theta/2)],
                     [-1j*math.sin(theta/2), math.cos(theta/2)]], dtype=complex)

def apply_single_gate(state, gate, target, n):
    """Apply single-qubit gate matrix to target qubit (0 = MSB)"""
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
    """Apply CNOT (control,target) to statevector (control,target 0=MSB)."""
    N = 1<<n
    sv = state.copy()
    for idx in range(N):
        if ((idx >> (n-1-control)) & 1) == 1:
            tgt_idx = idx ^ (1 << (n-1-target))
            sv[idx], sv[tgt_idx] = sv[tgt_idx], sv[idx]
    return sv

# ---------------- reversible quantum encoding (no measurement) ----------------

def unitary_ops_for_chunk(n, rng):
    """
    Create a deterministic list of operations for a chunk:
    - per-qubit flags: apply_h (bool), rx_angle (float)
    - cnot_pairs: list of (control,target) applied in order
    Returns ops dict serializable to JSON.
    """
    ops = {'apply_h': [], 'rx_angles': [], 'cnot_pairs': []}
    for q in range(n):
        ops['apply_h'].append(bool(rng.random() > 0.5))
        # choose angle in [0, pi)
        ops['rx_angles'].append(float(rng.random() * math.pi))
    for q in range(n-1):
        if rng.random() > 0.5:
            ops['cnot_pairs'].append([q, q+1])
    return ops

def apply_ops_to_state(state, ops, n):
    """Apply ops (in forward order) to statevector."""
    # H gates
    for q, do_h in enumerate(ops['apply_h']):
        if do_h:
            state = apply_single_gate(state, H, q, n)
    # RX gates
    for q, angle in enumerate(ops['rx_angles']):
        if angle != 0.0:
            state = apply_single_gate(state, rx(angle), q, n)
    # CNOTs in listed order
    for c,t in ops['cnot_pairs']:
        state = apply_cnot(state, c, t, n)
    return state

def apply_inverse_ops_to_state(state, ops, n):
    """Apply inverse ops in reverse order to statevector."""
    # inverse of CNOT is itself; reverse order
    for c,t in reversed(ops['cnot_pairs']):
        state = apply_cnot(state, c, t, n)
    # inverse RX is RX(-angle)
    for q, angle in reversed(list(enumerate(ops['rx_angles']))):
        if angle != 0.0:
            state = apply_single_gate(state, rx(-angle), q, n)
    # inverse H is H itself (apply in reverse order)
    for q, do_h in reversed(list(enumerate(ops['apply_h']))):
        if do_h:
            state = apply_single_gate(state, H, q, n)
    return state

def reversible_quantum_encode(bitstring: str, chunk_size=DEFAULT_CHUNK_QUBITS, rng_seed=None):
    """
    For each chunk (length chunk_size), produce:
      - final statevector after applying a random unitary (reversible)
      - store ops/key to invert later
    Returns:
      statevectors: list of complex arrays (dtype complex128) as numpy arrays
      keys: list of ops per chunk
      elapsed_time, chunks_count
    """
    if chunk_size < 1 or chunk_size > MAX_CHUNK_QUBITS:
        raise RuntimeError("chunk_size out of range")
    chunks = [bitstring[i:i+chunk_size] for i in range(0, len(bitstring), chunk_size)]
    rng = np.random.default_rng(rng_seed)
    statevectors = []
    keys = []
    t0 = time.perf_counter()
    for ch in chunks:
        n = len(ch)
        # pad last chunk to chunk_size if needed (we will remember actual length)
        if n < chunk_size:
            ch = ch + "0" * (chunk_size - n)
            n = chunk_size
        N = 1 << n
        basis_index = int(ch, 2)
        psi = np.zeros(N, dtype=complex)
        psi[basis_index] = 1.0 + 0j
        ops = unitary_ops_for_chunk(n, rng)
        psi_final = apply_ops_to_state(psi, ops, n)
        statevectors.append(psi_final)
        keys.append({'n': n, 'ops': ops, 'orig_len': len(ch.rstrip('0')) if ch.rstrip('0') != '' else chunk_size})
    elapsed = time.perf_counter() - t0
    return statevectors, keys, elapsed, len(chunks)

def save_quantum_package(statevectors, keys, outpath):
    """
    Save statevectors and keys to a single .npz file.
    We'll store statevectors as arrays v0, v1, ... and keys as JSON string.
    """
    arrays = {f"v{i}": sv for i, sv in enumerate(statevectors)}
    # keys must be JSON serializable; angles already floats
    keys_json = json.dumps(keys)
    np.savez_compressed(outpath, __keys_json__=keys_json, **arrays)

def load_quantum_package(path):
    z = np.load(path, allow_pickle=True)
    keys_json = z['__keys_json__'].tolist()
    keys = json.loads(keys_json)
    statevectors = []
    i = 0
    while f"v{i}" in z:
        statevectors.append(z[f"v{i}"])
        i += 1
    return statevectors, keys

def reversible_quantum_decrypt(statevectors, keys):
    """
    Given statevectors and keys, apply inverse ops to recover basis bitstrings,
    concatenate and return recovered bitstring.
    """
    recovered_chunks = []
    t0 = time.perf_counter()
    for sv, key in zip(statevectors, keys):
        n = key['n']
        ops = key['ops']
        orig_len = key.get('orig_len', n)
        # apply inverse ops
        psi_recovered = apply_inverse_ops_to_state(sv, ops, n)
        # psi_recovered should be basis vector; find index of max amplitude
        idx = int(np.argmax(np.abs(psi_recovered)))
        bits = format(idx, f"0{n}b")
        # trim to original length
        bits = bits[:orig_len]
        recovered_chunks.append(bits)
    elapsed = time.perf_counter() - t0
    full_bitstring = ''.join(recovered_chunks)
    return full_bitstring, elapsed

# ---------------- flask endpoints ----------------

@app.route("/", methods=["GET"])
def index_page():
    return render_template_string(HTML, default=DEFAULT_CHUNK_QUBITS, max=MAX_CHUNK_QUBITS, encode_result=None, decrypt_result=None)

@app.route("/encode", methods=["POST"])
def encode_route():
    uploaded = request.files.get("file")
    if not uploaded:
        return redirect(url_for("index_page"))
    filename_key = f"{uuid.uuid4().hex}_{uploaded.filename}"
    saved_path = os.path.join(UPLOAD_FOLDER, filename_key)
    uploaded.save(saved_path)

    # extract text
    try:
        text = extract_text_from_pdf(saved_path)
    except Exception as e:
        return f"<h3>Error reading PDF: {e}</h3><a href='/'>Back</a>"

    # classical
    t0c = time.perf_counter()
    b = text_to_bytes(text)
    classical_bitstr = bytes_to_bitstring(b)
    t_classical = time.perf_counter() - t0c
    classical_name = f"{filename_key}.bin"
    classical_path = os.path.join(UPLOAD_FOLDER, classical_name)
    with open(classical_path, "wb") as f:
        f.write(b)

    # quantum reversible encode
    chunk_size = int(request.form.get("chunk_size", DEFAULT_CHUNK_QUBITS))
    try:
        statevectors, keys, t_quantum, chunks = reversible_quantum_encode(classical_bitstr, chunk_size=chunk_size)
    except Exception as e:
        return f"<h3>Quantum encode error: {e}</h3><a href='/'>Back</a>"

    qenc_name = f"{filename_key}.qenc.npz"
    qenc_path = os.path.join(UPLOAD_FOLDER, qenc_name)
    save_quantum_package(statevectors, keys, qenc_path)

    result = {
        "filename": uploaded.filename,
        "classical_name": classical_name,
        "classical_size": len(b),
        "qenc_name": qenc_name,
        "classical_time": round(t_classical, 6),
        "quantum_time": round(t_quantum, 6),
        "chunks": chunks,
        "chunk_size": chunk_size
    }
    return render_template_string(HTML, default=DEFAULT_CHUNK_QUBITS, max=MAX_CHUNK_QUBITS, encode_result=result, decrypt_result=None)

@app.route("/decrypt", methods=["POST"])
def decrypt_route():
    qenc_file = request.files.get("qenc")
    if not qenc_file:
        return redirect(url_for("index_page"))
    keyname = f"{uuid.uuid4().hex}_{qenc_file.filename}"
    saved_path = os.path.join(UPLOAD_FOLDER, keyname)
    qenc_file.save(saved_path)

    try:
        statevectors, keys = load_quantum_package(saved_path)
    except Exception as e:
        return f"<h3>Error loading quantum package: {e}</h3><a href='/'>Back</a>"

    try:
        recovered_bitstr, t_decrypt = reversible_quantum_decrypt(statevectors, keys)
    except Exception as e:
        return f"<h3>Decrypt error: {e}</h3><a href='/'>Back</a>"

    recovered_bytes = bitstring_to_bytes(recovered_bitstr)
    recovered_name = f"{keyname}.recovered.txt"
    recovered_path = os.path.join(UPLOAD_FOLDER, recovered_name)
    with open(recovered_path, "wb") as f:
        f.write(recovered_bytes)

    result = {
        "recovered_name": recovered_name,
        "time": round(t_decrypt, 6),
        "bytes": len(recovered_bytes)
    }
    return render_template_string(HTML, default=DEFAULT_CHUNK_QUBITS, max=MAX_CHUNK_QUBITS, encode_result=None, decrypt_result=result)

@app.route("/download/<name>", methods=["GET"])
def download(name):
    path = os.path.join(UPLOAD_FOLDER, name)
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    print("Starting reversible quantum encoder/decrypter at http://127.0.0.1:5000")
    app.run(debug=True)



