"""Problema 2 - Registros multi-qubit

Aplicación interactiva para estudiar el estado
    |psi> = |+> tensor |0> tensor |->

Convención de orden:
    primer factor  -> q2 (bit más significativo)
    segundo factor -> q1
    tercer factor  -> q0 (bit menos significativo)

Requisitos:
    pip install numpy matplotlib qiskit

Tkinter normalmente viene incluido con Python. En Linux puede requerir
instalar el paquete python3-tk.

Qiskit es opcional para la parte matemática y las gráficas. Si está instalado,
la aplicación también construye el circuito y verifica el estado mediante
Statevector y partial_trace.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, partial_trace
    QISKIT_AVAILABLE = True
except ImportError:
    QuantumCircuit = None
    Statevector = None
    partial_trace = None
    QISKIT_AVAILABLE = False


# ============================================================
# 1. PALETA VISUAL (alto contraste)
# ============================================================

BG = "#0B1220"          # fondo general
PANEL = "#14243A"       # paneles
PANEL_2 = "#1B304A"     # panel secundario
FG = "#F4F7FB"          # texto claro
MUTED = "#A9B8C9"        # texto secundario
CIAN = "#00E5FF"         # amplitudes reales / resaltados
NARANJA = "#FFB000"      # probabilidades / amplitudes imaginarias
MAGENTA = "#FF4FA3"      # resultados solicitados
VERDE = "#8BFF70"        # validaciones
CERO = "#52657A"         # amplitud cero

SQRT2_INV = 1 / np.sqrt(2)
TOL = 1e-10

# Vectores en el orden [amplitud de |0>, amplitud de |1>].
# Los estados |+i> y |-i> se incluyen para explorar fases complejas.
QUBIT_STATES: Dict[str, np.ndarray] = {
    "|0⟩": np.array([1, 0], dtype=complex),
    "|1⟩": np.array([0, 1], dtype=complex),
    "|+⟩": np.array([SQRT2_INV, SQRT2_INV], dtype=complex),
    "|−⟩": np.array([SQRT2_INV, -SQRT2_INV], dtype=complex),
    "|+i⟩": np.array([SQRT2_INV, 1j * SQRT2_INV], dtype=complex),
    "|−i⟩": np.array([SQRT2_INV, -1j * SQRT2_INV], dtype=complex),
}

# Preparación de cada estado desde |0> mediante puertas elementales.
# La lista se aplica en orden temporal, de izquierda a derecha.
PREPARATION_GATES: Dict[str, List[str]] = {
    "|0⟩": [],
    "|1⟩": ["x"],
    "|+⟩": ["h"],
    "|−⟩": ["x", "h"],
    "|+i⟩": ["h", "s"],
    "|−i⟩": ["h", "sdg"],
}


# ============================================================
# 2. FUNCIONES MATEMÁTICAS
# ============================================================

def format_complex(value: complex, digits: int = 4) -> str:
    """Formatea un número complejo de forma legible."""
    z = complex(value)
    if abs(z.imag) < TOL:
        return f"{z.real:+.{digits}f}"
    if abs(z.real) < TOL:
        return f"{z.imag:+.{digits}f}i"
    return f"({z.real:+.{digits}f}{z.imag:+.{digits}f}i)"


def tensor_three(v2: np.ndarray, v1: np.ndarray, v0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula primero q2⊗q1 y después (q2⊗q1)⊗q0."""
    pair = np.kron(v2, v1)
    full = np.kron(pair, v0)
    return pair, full


def basis_label(index: int) -> str:
    """Etiqueta computacional en el orden |q2 q1 q0>."""
    return f"|{index:03b}⟩"


def state_expression(amplitudes: np.ndarray) -> str:
    """Construye la expresión del estado omitiendo amplitudes nulas."""
    terms = []
    for index, amplitude in enumerate(amplitudes):
        if abs(amplitude) > TOL:
            terms.append(f"({format_complex(amplitude)}){basis_label(index)}")
    return "|ψ⟩ = " + " + ".join(terms) if terms else "|ψ⟩ = 0"


def reduced_purity_from_vector(vector: np.ndarray) -> float:
    """Pureza Tr(rho²) de un qubit puro, calculada desde su vector."""
    rho = np.outer(vector, np.conjugate(vector))
    return float(np.real(np.trace(rho @ rho)))


def build_qiskit_circuit(labels: Tuple[str, str, str]):
    """Prepara q2, q1 y q0 en un circuito Qiskit de tres qubits."""
    if not QISKIT_AVAILABLE:
        return None, None

    circuit = QuantumCircuit(3)
    # El estado escrito |q2 q1 q0> corresponde a los qubits [2, 1, 0].
    for qubit, label in zip((2, 1, 0), labels):
        for gate_name in PREPARATION_GATES[label]:
            getattr(circuit, gate_name)(qubit)

    qiskit_state = Statevector.from_instruction(circuit)
    return circuit, qiskit_state


def calculate_experiment(labels: Tuple[str, str, str]) -> dict:
    """Calcula expansión, probabilidades, separabilidad y verificaciones."""
    v2, v1, v0 = (QUBIT_STATES[label] for label in labels)
    pair, amplitudes = tensor_three(v2, v1, v0)
    probabilities = np.abs(amplitudes) ** 2

    # Validaciones matemáticas básicas.
    local_norms = [np.vdot(v, v).real for v in (v2, v1, v0)]
    global_norm = float(np.vdot(amplitudes, amplitudes).real)
    assert np.allclose(local_norms, [1, 1, 1], atol=TOL)
    assert np.isclose(global_norm, 1.0, atol=TOL)
    assert np.isclose(np.sum(probabilities), 1.0, atol=TOL)

    nonzero = [i for i, amp in enumerate(amplitudes) if abs(amp) > TOL]

    # Las purezas locales valen 1 porque cada factor es un estado puro
    # y el estado global fue construido como producto tensorial.
    local_purities = {
        "q2": reduced_purity_from_vector(v2),
        "q1": reduced_purity_from_vector(v1),
        "q0": reduced_purity_from_vector(v0),
    }

    circuit, qiskit_state = build_qiskit_circuit(labels)
    qiskit_match = None
    qiskit_purities = None

    if QISKIT_AVAILABLE:
        # La comparación considera equivalentes los estados que solo difieren
        # en una fase global, aunque aquí se espera coincidencia directa.
        qiskit_match = bool(
            np.allclose(amplitudes, qiskit_state.data, atol=1e-9)
            or qiskit_state.equiv(Statevector(amplitudes))
        )
        qiskit_purities = {}
        # partial_trace recibe los índices de los qubits que se eliminan.
        for qubit, name in ((2, "q2"), (1, "q1"), (0, "q0")):
            traced_out = [i for i in range(3) if i != qubit]
            reduced_state = partial_trace(qiskit_state, traced_out)
            qiskit_purities[name] = float(np.real(reduced_state.purity()))

    # Para un estado global puro, que cada reducción de un qubit tenga
    # pureza 1 confirma que no está entrelazado con los otros qubits.
    separable = all(abs(value - 1.0) < 1e-8 for value in local_purities.values())

    return {
        "labels": labels,
        "vectors": (v2, v1, v0),
        "pair": pair,
        "amplitudes": amplitudes,
        "probabilities": probabilities,
        "nonzero": nonzero,
        "local_norms": local_norms,
        "global_norm": global_norm,
        "local_purities": local_purities,
        "separable": separable,
        "circuit": circuit,
        "qiskit_state": qiskit_state,
        "qiskit_match": qiskit_match,
        "qiskit_purities": qiskit_purities,
    }


# ============================================================
# 3. DESARROLLO PASO A PASO PARA LA SUSTENTACIÓN
# ============================================================

def create_explanation(result: dict) -> str:
    labels = result["labels"]
    v2, v1, v0 = result["vectors"]
    pair = result["pair"]
    amplitudes = result["amplitudes"]
    probabilities = result["probabilities"]
    nonzero = result["nonzero"]

    lines: List[str] = []

    def heading(text: str):
        lines.append("\n" + "═" * 76)
        lines.append(text)
        lines.append("═" * 76)

    heading("PROBLEMA 2 · REGISTROS MULTI-QUBIT")
    lines.append("Estado solicitado: |ψ⟩ = |+⟩ ⊗ |0⟩ ⊗ |−⟩")
    lines.append("Convención: |q2 q1 q0⟩; q2 es el primer factor y q0 el tercero.")
    lines.append("En la base computacional, el orden es |000⟩, |001⟩, ..., |111⟩.\n")

    heading("PASO 1 · REPRESENTAR CADA QUBIT COMO UN VECTOR")
    for name, label, vector in zip(("q2", "q1", "q0"), labels, (v2, v1, v0)):
        lines.append(
            f"{name} = {label} = [{format_complex(vector[0])}, "
            f"{format_complex(vector[1])}]ᵀ"
        )
    lines.append("Recordatorio: |+⟩ = (|0⟩ + |1⟩)/√2")
    lines.append("             |−⟩ = (|0⟩ − |1⟩)/√2")

    heading("PASO 2 · PRODUCTO TENSORIAL DE q2 Y q1")
    lines.append("Para dos vectores [a0, a1]ᵀ y [b0, b1]ᵀ:")
    lines.append("[a0, a1]ᵀ ⊗ [b0, b1]ᵀ = [a0b0, a0b1, a1b0, a1b1]ᵀ")
    lines.append("En este ejercicio:")
    lines.append("|+⟩ ⊗ |0⟩ = (1/√2)[1, 0, 1, 0]ᵀ")
    lines.append("Vector calculado para q2⊗q1:")
    lines.append("[" + ", ".join(format_complex(x) for x in pair) + "]ᵀ")

    heading("PASO 3 · INCORPORAR EL TERCER QUBIT q0")
    lines.append("Ahora calculamos (q2⊗q1)⊗q0. El resultado tiene 2³ = 8 amplitudes.")
    for index, amplitude in enumerate(amplitudes):
        q2 = (index >> 2) & 1
        q1 = (index >> 1) & 1
        q0 = index & 1
        pair_index = 2 * q2 + q1
        third_index = q0
        lines.append(
            f"{basis_label(index)}: ({format_complex(pair[pair_index])}) "
            f"× ({format_complex(v0[third_index])}) = {format_complex(amplitude)}"
        )

    heading("INCISO a) · EXPRESIÓN EN LA BASE COMPUTACIONAL")
    lines.append(state_expression(amplitudes))
    lines.append("Vector de estado, en el orden |000⟩ hasta |111⟩:")
    lines.append("[" + ", ".join(format_complex(x) for x in amplitudes) + "]ᵀ")

    heading("INCISO b) · ¿CUÁNTOS ESTADOS BASE TIENEN AMPLITUD NO NULA?")
    lines.append(f"Cantidad: {len(nonzero)} de 8 estados base.")
    lines.append("Estados con amplitud no nula: " + ", ".join(basis_label(i) for i in nonzero))
    lines.append("Los otros cuatro estados tienen amplitud cero.")

    heading("INCISO c) · PROBABILIDADES CON LA REGLA DE BORN")
    lines.append("La probabilidad de medir un estado base es el módulo al cuadrado de su amplitud:")
    lines.append("P(|abc⟩) = |amplitud de |abc⟩|²")
    for index, amplitude in enumerate(amplitudes):
        lines.append(
            f"P({basis_label(index)}) = |{format_complex(amplitude)}|² "
            f"= {probabilities[index]:.4f} ({probabilities[index] * 100:.2f} %)"
        )
    lines.append(f"P(|000⟩) = |{format_complex(amplitudes[0])}|² = {probabilities[0]:.4f}")
    lines.append(f"P(|101⟩) = |{format_complex(amplitudes[5])}|² = {probabilities[5]:.4f}")
    lines.append(f"Comprobación: suma de las 8 probabilidades = {probabilities.sum():.4f} (debe ser 1).")

    heading("INCISO d) · ¿EL ESTADO ES SEPARABLE O ENTRELAZADO?")
    lines.append("Es SEPARABLE porque está escrito como producto tensorial de tres estados:")
    lines.append("|ψ⟩ = |+⟩ ⊗ |0⟩ ⊗ |−⟩")
    lines.append("Esto significa que no hay correlaciones de entrelazamiento entre los qubits.")
    lines.append("Comprobación mediante pureza de cada qubit reducido, Tr(ρ²):")
    for name, purity in result["local_purities"].items():
        lines.append(f"  {name}: {purity:.6f} (igual a 1: estado puro)")
    lines.append("Para un estado global puro, pureza local 1 en cada qubit confirma separabilidad.")

    heading("VERIFICACIONES")
    lines.append("Norma de q2: {:.6f}".format(result["local_norms"][0]))
    lines.append("Norma de q1: {:.6f}".format(result["local_norms"][1]))
    lines.append("Norma de q0: {:.6f}".format(result["local_norms"][2]))
    lines.append("Norma del estado global: {:.6f}".format(result["global_norm"]))
    if QISKIT_AVAILABLE:
        lines.append(f"Coincidencia entre producto de Kronecker y Qiskit: {result['qiskit_match']}")
        lines.append("Purezas calculadas con partial_trace de Qiskit:")
        for name, purity in result["qiskit_purities"].items():
            lines.append(f"  {name}: {purity:.6f}")
    else:
        lines.append("Qiskit no está instalado: se ejecutó la verificación matemática con NumPy.")
        lines.append("Instala Qiskit para mostrar el circuito y comprobar partial_trace.")

    heading("IDEA CENTRAL PARA LA SUSTENTACIÓN")
    lines.append("El producto tensorial combina las tres posibilidades de los qubits.")
    lines.append("La regla de Born convierte cada amplitud en una probabilidad.")
    lines.append("El estado es separable porque se puede escribir como producto de estados individuales.")

    return "\n".join(lines)


# ============================================================
# 4. INTERFAZ GRÁFICA INTERACTIVA
# ============================================================

class MultiQubitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Problema 2 · Registros multi-qubit")
        self.geometry("1250x850")
        self.minsize(1050, 720)
        self.configure(bg=BG)

        self.current_result = None
        self._configure_style()
        self._build_header()
        self._build_notebook()
        self.update_experiment()

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL_2, foreground=FG,
                        padding=(14, 8), font=("Arial", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", CIAN)],
                  foreground=[("selected", BG)])
        style.configure("TCombobox", fieldbackground=PANEL_2, background=PANEL_2,
                        foreground=FG, arrowcolor=FG)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL_2)],
                  foreground=[("readonly", FG)])
        style.configure("TButton", background=PANEL_2, foreground=FG,
                        padding=7, font=("Arial", 10, "bold"))
        style.map("TButton", background=[("active", CIAN)],
                  foreground=[("active", BG)])

    def _build_header(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=16, pady=(12, 8))

        tk.Label(
            header, text="PROBLEMA 2 · REGISTROS MULTI-QUBIT",
            bg=BG, fg=CIAN, font=("Arial", 19, "bold")
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Estado: |ψ⟩ = |+⟩ ⊗ |0⟩ ⊗ |−⟩     ·     Orden: |q2 q1 q0⟩",
            bg=BG, fg=FG, font=("Arial", 12)
        ).pack(anchor="w", pady=(3, 8))

        controls = tk.Frame(header, bg=PANEL, padx=12, pady=10)
        controls.pack(fill="x")

        self.combos = {}
        defaults = (("q2", "|+⟩"), ("q1", "|0⟩"), ("q0", "|−⟩"))
        for name, default in defaults:
            tk.Label(controls, text=f"{name}:", bg=PANEL, fg=FG,
                     font=("Arial", 11, "bold")).pack(side="left", padx=(8, 5))
            combo = ttk.Combobox(
                controls, values=list(QUBIT_STATES.keys()), width=7,
                state="readonly", font=("Arial", 11)
            )
            combo.set(default)
            combo.pack(side="left", padx=(0, 16))
            combo.bind("<<ComboboxSelected>>", self.update_experiment)
            self.combos[name] = combo

        ttk.Button(controls, text="Restaurar ejercicio", command=self.restore_default).pack(side="right", padx=8)

        tk.Label(
            header,
            text=("Los selectores permiten explorar otros estados individuales. "
                  "Todas esas combinaciones siguen siendo separables porque no se aplica ninguna puerta de entrelazamiento."),
            bg=BG, fg=MUTED, font=("Arial", 9), wraplength=1180, justify="left"
        ).pack(anchor="w", pady=(7, 0))

    def _build_notebook(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        self.tab_graphs = tk.Frame(notebook, bg=BG)
        self.tab_math = tk.Frame(notebook, bg=BG)
        self.tab_circuit = tk.Frame(notebook, bg=BG)

        notebook.add(self.tab_graphs, text="  Resultados y gráficas  ")
        notebook.add(self.tab_math, text="  Desarrollo matemático  ")
        notebook.add(self.tab_circuit, text="  Circuito y verificaciones  ")

        self._build_graph_tab()
        self._build_math_tab()
        self._build_circuit_tab()

    def _build_graph_tab(self):
        self.summary = tk.Label(
            self.tab_graphs, text="", bg=PANEL, fg=FG,
            font=("Arial", 11, "bold"), justify="left", anchor="w",
            padx=14, pady=10
        )
        self.summary.pack(fill="x", padx=12, pady=12)

        self.figure = Figure(figsize=(11, 6.0), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.tab_graphs)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _build_math_tab(self):
        self.math_text = ScrolledText(
            self.tab_math, wrap="none", bg=BG, fg=FG,
            insertbackground=FG, font=("Consolas", 10),
            padx=14, pady=12, borderwidth=0
        )
        self.math_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.math_text.tag_configure("heading", foreground=CIAN,
                                     font=("Consolas", 11, "bold"))

    def _build_circuit_tab(self):
        self.circuit_text = ScrolledText(
            self.tab_circuit, wrap="none", bg=BG, fg=FG,
            insertbackground=FG, font=("Consolas", 11),
            padx=14, pady=12, borderwidth=0
        )
        self.circuit_text.pack(fill="both", expand=True, padx=8, pady=8)

    def restore_default(self):
        for name, value in (("q2", "|+⟩"), ("q1", "|0⟩"), ("q0", "|−⟩")):
            self.combos[name].set(value)
        self.update_experiment()

    def update_experiment(self, _event=None):
        try:
            labels = tuple(self.combos[name].get() for name in ("q2", "q1", "q0"))
            result = calculate_experiment(labels)
            self.current_result = result
            self._update_summary(result)
            self._update_graphs(result)
            self._update_math(result)
            self._update_circuit(result)
        except Exception as exc:
            messagebox.showerror("Error en el experimento", str(exc))

    def _update_summary(self, result: dict):
        amp = result["amplitudes"]
        probs = result["probabilities"]
        self.summary.config(
            text=(
                f"Estados con amplitud no nula: {len(result['nonzero'])}/8     |     "
                f"P(|000⟩): {probs[0]:.2%}     |     P(|101⟩): {probs[5]:.2%}     |     "
                f"Estado: {'SEPARABLE' if result['separable'] else 'ENTRELAZADO'}\n"
                f"Amplitudes: {state_expression(amp)}"
            )
        )

    def _style_axis(self, ax, title: str, ylabel: str):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=FG, fontsize=11, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, color=FG)
        ax.tick_params(colors=FG, labelsize=8)
        ax.grid(axis="y", color="#53677D", alpha=0.35, linestyle="--")
        for spine in ax.spines.values():
            spine.set_color(MUTED)
        ax.set_axisbelow(True)

    def _update_graphs(self, result: dict):
        self.figure.clear()
        ax_prob, ax_amp = self.figure.subplots(1, 2)

        labels = [basis_label(i) for i in range(8)]
        probs = result["probabilities"]
        amps = result["amplitudes"]

        # Gráfica 1: distribución de probabilidad en la base computacional.
        colors = []
        for i, probability in enumerate(probs):
            if probability < TOL:
                colors.append(CERO)
            elif i in (0, 5):
                colors.append(MAGENTA)
            else:
                colors.append(CIAN)

        bars = ax_prob.bar(labels, probs, color=colors, edgecolor=FG, linewidth=0.8)
        for bar, value in zip(bars, probs):
            if value > TOL:
                ax_prob.text(bar.get_x() + bar.get_width() / 2, value + 0.018,
                             f"{value:.2f}", ha="center", va="bottom",
                             color=FG, fontsize=8, fontweight="bold")
        ax_prob.set_ylim(0, max(0.25, float(np.max(probs)) * 1.25))
        ax_prob.tick_params(axis="x", rotation=45)
        self._style_axis(ax_prob, "Probabilidad de cada estado base", "Probabilidad")
        ax_prob.set_xlabel("Base computacional |q2 q1 q0⟩", color=FG)

        # Gráfica 2: amplitud compleja separada en parte real e imaginaria.
        x = np.arange(8)
        width = 0.36
        real_values = np.real(amps)
        imag_values = np.imag(amps)
        ax_amp.bar(x - width / 2, real_values, width, color=CIAN,
                   edgecolor=FG, linewidth=0.6, label="Parte real")
        ax_amp.bar(x + width / 2, imag_values, width, color=NARANJA,
                   edgecolor=FG, linewidth=0.6, label="Parte imaginaria")
        ax_amp.axhline(0, color=FG, linewidth=0.8)
        ax_amp.set_xticks(x)
        ax_amp.set_xticklabels(labels, rotation=45)
        self._style_axis(ax_amp, "Amplitudes: parte real e imaginaria", "Amplitud")
        ax_amp.set_xlabel("Base computacional |q2 q1 q0⟩", color=FG)
        legend = ax_amp.legend(facecolor=PANEL_2, edgecolor=MUTED, fontsize=8)
        for text in legend.get_texts():
            text.set_color(FG)

        self.figure.tight_layout(pad=2.2)
        self.canvas.draw_idle()

    def _update_math(self, result: dict):
        self.math_text.config(state="normal")
        self.math_text.delete("1.0", "end")
        content = create_explanation(result)
        for line in content.splitlines(True):
            if line.startswith("PASO ") or line.startswith("INCISO ") or line.startswith("PROBLEMA ") or line.startswith("VERIFICACIONES") or line.startswith("IDEA CENTRAL"):
                self.math_text.insert("end", line, "heading")
            else:
                self.math_text.insert("end", line)
        self.math_text.config(state="disabled")

    def _update_circuit(self, result: dict):
        self.circuit_text.config(state="normal")
        self.circuit_text.delete("1.0", "end")

        if QISKIT_AVAILABLE:
            circuit = result["circuit"]
            circuit_text = str(circuit.draw(output="text", reverse_bits=True))
            self.circuit_text.insert("end", "CIRCUITO PREPARADO CON QISKIT\n")
            self.circuit_text.insert("end", "Orden visual: q2, q1, q0.\n\n")
            self.circuit_text.insert("end", circuit_text + "\n\n")
            self.circuit_text.insert("end", f"Coincidencia con producto tensorial: {result['qiskit_match']}\n")
            self.circuit_text.insert("end", "Pureza de los qubits reducidos calculada con partial_trace:\n")
            for name, purity in result["qiskit_purities"].items():
                self.circuit_text.insert("end", f"  {name}: {purity:.6f}\n")
        else:
            self.circuit_text.insert("end", "QISKIT NO ESTÁ INSTALADO\n\n")
            self.circuit_text.insert("end", "La aplicación puede calcular el ejercicio con NumPy, pero para ver el circuito\n")
            self.circuit_text.insert("end", "y verificar las trazas parciales instala las dependencias:\n\n")
            self.circuit_text.insert("end", "python -m pip install qiskit\n\n")
            self.circuit_text.insert("end", "Preparación equivalente desde |000⟩ para el estado original:\n")
            self.circuit_text.insert("end", "  q2: H       prepara |+⟩\n")
            self.circuit_text.insert("end", "  q1: ninguna puerta; permanece en |0⟩\n")
            self.circuit_text.insert("end", "  q0: X y después H; prepara |−⟩\n")

        self.circuit_text.insert("end", "\nNOTA DE ORDEN DE BITS\n")
        self.circuit_text.insert("end", "El estado se escribe |q2 q1 q0⟩. Qiskit numera q0 como el bit menos significativo;\n")
        self.circuit_text.insert("end", "por eso el vector de Statevector se interpreta como |000⟩, |001⟩, ..., |111⟩.\n")
        self.circuit_text.config(state="disabled")


# ============================================================
# 5. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    app = MultiQubitApp()
    app.mainloop()
