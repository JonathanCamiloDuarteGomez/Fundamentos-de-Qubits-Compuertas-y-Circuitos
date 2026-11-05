#correr el repo: python.exe .\src\p1_qubit.py 

# comandos para crear el .env
# python --version
# python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

#instalar dependecias: 
# pip install qiskit qiskit-aer qutip matplotlib pylatexenc
#comprobar:
python -c "import qiskit, qutip, qiskit_aer; print(qiskit.__version__, qutip.__version__)"
