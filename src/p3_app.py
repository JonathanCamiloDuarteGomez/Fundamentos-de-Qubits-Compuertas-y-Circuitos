"""Problema 3 - Compuertas de un qubit: HZH = X

Aplicación didáctica con Tkinter y Matplotlib.
Qiskit es opcional: si está instalado, también verifica el circuito.

Instalación básica:
    pip install numpy matplotlib

Verificación adicional con Qiskit:
    pip install qiskit

Ejecución:
    python problema3_compuertas_hzh_corregido.py
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Qiskit es opcional. La aplicación sigue funcionando sin él.
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    QISKIT_AVAILABLE = True
except ImportError:
    QuantumCircuit = None
    Operator = None
    QISKIT_AVAILABLE = False


# ---------------------------------------------------------------------
# Colores: definidos aquí para que el programa NO dependa de p1_app.py
# ---------------------------------------------------------------------
BG = "#101820"
PANEL = "#1B2836"
FG = "#F2F5F7"
CIAN = "#42D9E8"
NARANJA = "#FFAA55"
MAGENTA = "#E58BFF"


# ---------------------------------------------------------------------
# Matrices y estados
# ---------------------------------------------------------------------
S = 1 / np.sqrt(2)

H = S * np.array([[1, 1],
                  [1, -1]], dtype=complex)

Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

ESTADOS = {
    "|0⟩": np.array([1, 0], dtype=complex),
    "|1⟩": np.array([0, 1], dtype=complex),
    "|+⟩": S * np.array([1, 1], dtype=complex),
    "|−⟩": S * np.array([1, -1], dtype=complex),
}


def fmt(valor):
    """Formatea números reales o complejos sin mostrar -0.0000."""
    valor = complex(valor)
    real = 0.0 if abs(valor.real) < 1e-10 else valor.real
    imag = 0.0 if abs(valor.imag) < 1e-10 else valor.imag

    if imag == 0:
        return f"{real:+.4f}"
    return f"({real:+.4f}{imag:+.4f}i)"


def nombre(vector):
    """Reconoce |0>, |1>, |+> o |->, ignorando la fase global."""
    vector = np.asarray(vector, dtype=complex)
    for etiqueta, estado in ESTADOS.items():
        producto = np.vdot(estado, vector)
        if abs(abs(producto) - 1) < 1e-9:
            return etiqueta
    return f"[{fmt(vector[0])}, {fmt(vector[1])}]ᵀ"


def matriz_a_texto(M, prefijo="    "):
    return (
        f"{prefijo}[ {fmt(M[0, 0])}  {fmt(M[0, 1])} ]\n"
        f"{prefijo}[ {fmt(M[1, 0])}  {fmt(M[1, 1])} ]"
    )


def construir_circuito_qiskit():
    """Construye H -> Z -> H en Qiskit, si está disponible."""
    if not QISKIT_AVAILABLE:
        return None

    circuito = QuantumCircuit(1)
    circuito.h(0)
    circuito.z(0)
    circuito.h(0)
    return circuito


def procedimiento(etiqueta_entrada):
    """Calcula las tres etapas y prepara el procedimiento explicado."""
    v0 = ESTADOS[etiqueta_entrada]

    # El símbolo @ representa multiplicación matricial.
    v1 = H @ v0       # Primera compuerta: H
    v2 = Z @ v1       # Segunda compuerta: Z
    v3 = H @ v2       # Tercera compuerta: H

    # La secuencia H -> Z -> H se representa como H @ Z @ H.
    ZH = Z @ H
    U = H @ ZH

    # Comprobación numérica independiente.
    assert np.allclose(U, X), "La identidad HZH = X no se verificó."
    assert np.allclose(v3, X @ v0), "El resultado sobre el estado no coincide con X."

    qiskit_texto = "Qiskit no está instalado; se omitió esta comprobación."
    qiskit_ok = None
    circuito = construir_circuito_qiskit()
    if circuito is not None:
        # Se comparan operadores para comprobar que representan la misma puerta.
        qiskit_ok = Operator(circuito).equiv(Operator(X))
        qiskit_texto = f"Operator(circuito).equiv(Operator(X)) = {qiskit_ok}"
        assert qiskit_ok, "La comprobación de Qiskit no coincidió con X."

    pasos = []

    def encabezado(texto):
        pasos.append((f"\n{texto}\n", "h"))

    def linea(texto):
        pasos.append((texto + "\n", "f"))

    def resultado(texto):
        pasos.append((texto + "\n", "r"))

    encabezado("PASO 1 · ¿Qué secuencia vamos a aplicar?")
    linea("El orden físico es: primero H, después Z y finalmente H.")
    linea("En matrices, la operación total se escribe U = H · Z · H.")
    linea("La compuerta que actúa primero queda a la derecha.")

    encabezado("PASO 2 · Matrices de las compuertas")
    linea("Matriz Hadamard:")
    linea("H = (1/√2) · [[1, 1], [1, −1]]")
    linea(matriz_a_texto(H))
    linea("Matriz Pauli-Z:")
    linea("Z = [[1, 0], [0, −1]]")
    linea(matriz_a_texto(Z))

    encabezado(f"PASO 3 · Aplicar HZH a {etiqueta_entrada}")
    linea(f"Estado inicial: {etiqueta_entrada} = [{fmt(v0[0])}, {fmt(v0[1])}]ᵀ")
    linea("\n(3.1) Primera compuerta: H")
    linea("H · [α, β]ᵀ = [(α + β)/√2, (α − β)/√2]ᵀ")
    linea(f"Resultado: {nombre(v1)} = [{fmt(v1[0])}, {fmt(v1[1])}]ᵀ")
    linea("\n(3.2) Segunda compuerta: Z")
    linea("Z · [α, β]ᵀ = [α, −β]ᵀ")
    linea("Z mantiene la amplitud de |0⟩ y cambia el signo de |1⟩.")
    linea(f"Resultado: {nombre(v2)} = [{fmt(v2[0])}, {fmt(v2[1])}]ᵀ")
    linea("\n(3.3) Tercera compuerta: H")
    linea("Volvemos a aplicar H al vector obtenido después de Z.")
    linea(f"Resultado final: {nombre(v3)} = [{fmt(v3[0])}, {fmt(v3[1])}]ᵀ")
    resultado(
        f"Cadena: {etiqueta_entrada} → H: {nombre(v1)} → Z: {nombre(v2)} → H: {nombre(v3)}"
    )

    encabezado("PASO 4 · Verificación matricial (apartado C)")
    linea("Primero multiplicamos Z · H:")
    linea("ZH = (1/√2) · [[1, 1], [−1, 1]]")
    linea(matriz_a_texto(ZH))
    linea("Ahora multiplicamos H · (ZH):")
    linea("HZH = (1/2) · [[1−1, 1+1], [1+1, 1−1]]")
    linea(matriz_a_texto(U))
    resultado("HZH = [[0, 1], [1, 0]] = X (compuerta Pauli-X)")
    linea("X intercambia |0⟩ y |1⟩, igual que una compuerta NOT sobre esos estados.")

    encabezado("PASO 5 · Verificación con Qiskit")
    resultado(qiskit_texto)

    encabezado("PASO 6 · Idea para recordar")
    linea("H · Z · H = X")
    linea("Las dos compuertas H convierten la acción de Z en una inversión de bit.")

    return pasos, [v0, v1, v2, v3], circuito


# ---------------------------------------------------------------------
# Interfaz gráfica
# ---------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Problema 3 · Compuertas de un qubit: HZH = X")
        self.geometry("1180x790")
        self.minsize(900, 650)
        self.configure(bg=BG)

        estilo = ttk.Style(self)
        try:
            estilo.theme_use("clam")
        except tk.TclError:
            pass
        estilo.configure("TNotebook", background=BG, borderwidth=0)
        estilo.configure("TNotebook.Tab", padding=(12, 7))
        estilo.configure("TCombobox", padding=4)

        superior = tk.Frame(self, bg=BG)
        superior.pack(fill="x", padx=14, pady=10)

        tk.Label(
            superior,
            text="Estado de entrada:",
            bg=BG,
            fg=FG,
            font=("Arial", 12, "bold"),
        ).pack(side="left")

        self.selector = ttk.Combobox(
            superior,
            values=list(ESTADOS.keys()),
            width=7,
            state="readonly",
        )
        self.selector.set("|0⟩")
        self.selector.pack(side="left", padx=10)
        self.selector.bind("<<ComboboxSelected>>", self.actualizar)

        tk.Label(
            superior,
            text="Secuencia: H → Z → H = X",
            bg=BG,
            fg=CIAN,
            font=("Arial", 12, "bold"),
        ).pack(side="left", padx=20)

        cuaderno = ttk.Notebook(self)
        cuaderno.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Pestaña 1: gráficas
        pestaña_graficas = tk.Frame(cuaderno, bg=BG)
        cuaderno.add(pestaña_graficas, text="  Gráficas  ")

        self.figura = Figure(figsize=(11, 5), facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.figura, master=pestaña_graficas)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Pestaña 2: procedimiento matemático
        pestaña_texto = tk.Frame(cuaderno, bg=BG)
        cuaderno.add(pestaña_texto, text="  Procedimiento matemático  ")

        self.texto = tk.Text(
            pestaña_texto,
            bg=PANEL,
            fg=FG,
            font=("Consolas", 11),
            wrap="word",
            padx=16,
            pady=12,
            borderwidth=0,
            insertbackground=FG,
        )
        barra = ttk.Scrollbar(pestaña_texto, command=self.texto.yview)
        self.texto.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        self.texto.pack(side="left", fill="both", expand=True)
        self.texto.tag_configure("h", foreground=CIAN, font=("Consolas", 12, "bold"))
        self.texto.tag_configure("f", foreground=FG)
        self.texto.tag_configure("r", foreground=NARANJA, font=("Consolas", 11, "bold"))

        # Pestaña 3: circuito Qiskit / esquema
        pestaña_circuito = tk.Frame(cuaderno, bg=BG)
        cuaderno.add(pestaña_circuito, text="  Circuito  ")

        self.texto_circuito = tk.Text(
            pestaña_circuito,
            bg=PANEL,
            fg=CIAN,
            font=("Consolas", 13),
            wrap="none",
            borderwidth=0,
            padx=20,
            pady=20,
        )
        self.texto_circuito.pack(fill="both", expand=True)
        self.texto_circuito.config(state="disabled")

        self.actualizar()

    def actualizar(self, _evento=None):
        pasos, vectores, circuito = procedimiento(self.selector.get())

        # Actualizar gráficas: amplitudes con signo y probabilidad en porcentaje.
        self.figura.clear()
        ejes = self.figura.subplots(1, 4)
        etapas = ["Entrada", "Después de H", "Después de Z", "Después de H"]

        for eje, vector, etapa in zip(ejes, vectores, etapas):
            eje.set_facecolor(PANEL)
            amplitudes = np.real_if_close(vector).real
            posiciones = np.arange(2)
            eje.bar(
                posiciones,
                amplitudes,
                color=[CIAN, NARANJA],
                edgecolor="white",
                linewidth=0.7,
            )

            for i, amplitud in enumerate(amplitudes):
                probabilidad = abs(vector[i]) ** 2
                desplazamiento = 0.08 if amplitud >= 0 else -0.16
                eje.text(
                    i,
                    amplitud + desplazamiento,
                    f"P={probabilidad * 100:.0f}%",
                    ha="center",
                    va="center",
                    color=FG,
                    fontsize=9,
                    fontweight="bold",
                )

            eje.axhline(0, color="white", linewidth=0.8)
            eje.set_xticks(posiciones, ["|0⟩", "|1⟩"])
            eje.set_ylim(-1.25, 1.25)
            eje.set_title(f"{etapa}\n{nombre(vector)}", color=FG, fontsize=10)
            eje.tick_params(colors=FG)
            eje.spines["bottom"].set_color(FG)
            eje.spines["left"].set_color(FG)
            eje.spines["top"].set_visible(False)
            eje.spines["right"].set_visible(False)

        ejes[0].set_ylabel("Amplitud", color=FG)
        self.figura.suptitle(
            f"Evolución del estado inicial {self.selector.get()} mediante H → Z → H",
            color=FG,
            fontsize=12,
        )
        self.figura.tight_layout(rect=(0, 0, 1, 0.92))
        self.canvas.draw()

        # Actualizar explicación
        self.texto.config(state="normal")
        self.texto.delete("1.0", "end")
        for contenido, etiqueta in pasos:
            self.texto.insert("end", contenido, etiqueta)
        self.texto.config(state="disabled")

        # Actualizar dibujo del circuito
        if circuito is not None:
            dibujo = str(circuito.draw(output="text"))
            estado_qiskit = "Qiskit está disponible y verificó el circuito."
        else:
            dibujo = "Qiskit no está instalado."
            estado_qiskit = "La aplicación funciona igualmente con NumPy."

        texto_circ = (
            "CIRCUITO: H → Z → H\n\n"
            f"{dibujo}\n\n"
            "Operador total: U = H · Z · H = X\n"
            "La secuencia equivale a una sola compuerta Pauli-X.\n\n"
            f"{estado_qiskit}"
        )
        self.texto_circuito.config(state="normal")
        self.texto_circuito.delete("1.0", "end")
        self.texto_circuito.insert("end", texto_circ)
        self.texto_circuito.config(state="disabled")


if __name__ == "__main__":
    App().mainloop()
