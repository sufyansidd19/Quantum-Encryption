import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

db_path = "file_comparisons.db"  # make sure the file is in the same folder
conn = sqlite3.connect(db_path)


files_query = "SELECT DISTINCT original_file_name FROM file_comparisons"
files = pd.read_sql_query(files_query, conn)['original_file_name'].tolist()

print("Available files for analysis:\n")
for i, f in enumerate(files, 1):
    print(f"{i}. {f}")


# Select file from user

choice = int(input("\nEnter the file number you want to analyze: "))
selected_file = files[choice - 1]

print(f"\nAnalyzing: {selected_file}")


# Fetch records for the selected file

query = f"""
SELECT method, encrypt_time, decrypt_time, original_file_name, file_name
FROM file_comparisons
WHERE original_file_name LIKE '%{selected_file}%'
   OR file_name LIKE '%{selected_file}%'
"""
df = pd.read_sql_query(query, conn)
conn.close()


# Clean and prepare data

df = df.fillna(0)
grouped = df.groupby('method')[['encrypt_time', 'decrypt_time']].sum()

# Ensure consistent order
grouped = grouped.reindex(['Classical', 'Quantum'])


# Prepare data for plotting

labels = ['Encryption', 'Decryption']
classical_values = [grouped.loc['Classical', 'encrypt_time'], grouped.loc['Classical', 'decrypt_time']]
quantum_values = [grouped.loc['Quantum', 'encrypt_time'], grouped.loc['Quantum', 'decrypt_time']]

x = range(len(labels))
width = 0.35


# Plotting

plt.figure(figsize=(8,5))
plt.bar([p - width/2 for p in x], classical_values, width=width, label='Classical', color='skyblue')
plt.bar([p + width/2 for p in x], quantum_values, width=width, label='Quantum', color='orange')

plt.xticks(x, labels)
plt.ylabel('Time (seconds)')
plt.title(f'Encryption vs Decryption Time Comparison\nFile: {selected_file}')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Annotate values on top of bars
for i, v in enumerate(classical_values):
    plt.text(i - width/2, v + 0.0005, f"{v:.4f}", ha='center', va='bottom', fontsize=9)
for i, v in enumerate(quantum_values):
    plt.text(i + width/2, v + 0.0005, f"{v:.4f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
