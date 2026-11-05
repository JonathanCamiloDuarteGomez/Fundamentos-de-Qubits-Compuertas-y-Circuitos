# Fundamentos-de-Qubits-Compuertas-y-Circuitos
Solución de taller cuántica
python.exe .\src\p1_qubit.py

# configuracion de ambiente
python --version
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
pip install qiskit qiskit-aer qutip matplotlib pylatexenc
python -c "import qiskit, qutip, qiskit_aer; print(qiskit.__version__, qutip.__version__)"