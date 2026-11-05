import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from qiskit.circuit.library import HGate

from p1_qubit import construir_estado, calcular_probabilidades

BG, PANEL, FG = "#101820", "#192936", "#FFFFFF"
CIAN, NARANJA, MAGENTA = "#00E5FF", "#FFB000", "#FF4FA3"


def c(z):
    """Formatea un complejo."""
    return f"({z.real:+.4f} {z.imag:+.4f}i)"


def procedimiento(fase):
    """Devuelve el desarrollo matemático como lista de (texto, etiqueta)."""
    phi = np.deg2rad(fase)
    a, b, sv = construir_estado(fase)
    p0, p1 = calcular_probabilidades(sv)
    s = 1 / np.sqrt(2)
    a2, b2 = (a + b) * s, (a - b) * s
    assert np.allclose([a2, b2], sv.evolve(HGate()).data)  # verifica con Qiskit
    k = np.sqrt(2) / 3
    inter = (a * np.conj(b)).real          # Re(alpha * conj(beta))
    p0h, p1h = abs(a2) ** 2, abs(b2) ** 2

    T = []
    h = lambda t: T.append((f"\n{t}\n", "h"))
    f = lambda t: T.append((t + "\n", "f"))
    r = lambda t: T.append((t + "\n", "r"))

    h("PASO 1 · Estado inicial")
    f("|ψ⟩ = α|0⟩ + β|1⟩")
    f(f"α = 1/√3 = {a.real:.4f}")
    f(f"β = √(2/3)·e^(iφ),   φ = {fase}° = {phi:.4f} rad")
    f(f"β = {np.sqrt(2/3):.4f}·(cos φ + i·sen φ) = {np.sqrt(2/3):.4f}·({np.cos(phi):.4f} + {np.sin(phi):.4f}i) = {c(b)}")
    f(f"Normalización: |α|² + |β|² = {abs(a)**2:.4f} + {abs(b)**2:.4f} = {abs(a)**2 + abs(b)**2:.4f}")
    r("✓ Estado válido (la fase φ no altera los módulos)")

    h("PASO 2 · Probabilidades ANTES de H (regla de Born)")
    f(f"P(0) = |α|² = {p0:.4f}   ({p0*100:.2f} %)")
    f(f"P(1) = |β|² = {p1:.4f}   ({p1*100:.2f} %)")

    h("PASO 3 · La compuerta Hadamard")
    f("        1   ⎡ 1   1 ⎤")
    f("H  =  ───  ⎢       ⎥        H† H = I   (unitaria, H² = I)")
    f("       √2   ⎣ 1  −1 ⎦")

    h("PASO 4 · Producto matriz × vector:  |ψ'⟩ = H|ψ⟩")
    f("α' = (α + β)/√2        β' = (α − β)/√2")
    f(f"α + β = {c(a + b)}")
    f(f"α − β = {c(a - b)}")
    f(f"α' = {c(a + b)} · 0.7071 = {c(a2)}")
    f(f"β' = {c(a - b)} · 0.7071 = {c(b2)}")

    h("PASO 5 · Probabilidades DESPUÉS de H")
    f("P'(0) = |α'|² = |α + β|²/2 = (|α|² + |β|² + 2·Re(αβ*)) / 2")
    f("       = ½ + Re(αβ*)          (porque |α|² + |β|² = 1)")
    f(f"αβ* = (1/√3)·√(2/3)·e^(−iφ) = (√2/3)·e^(−iφ) = {k:.4f}·e^(−iφ)")
    f(f"Re(αβ*) = {k:.4f}·cos({fase}°) = {k:.4f}·{np.cos(phi):.4f} = {inter:+.4f}")
    r(f"P'(0) = 0.5 {inter:+.4f} = {p0h:.4f}   ({p0h*100:.2f} %)")
    r(f"P'(1) = 0.5 {-inter:+.4f} = {p1h:.4f}   ({p1h*100:.2f} %)")
    f(f"Comprobación directa: |α'|² = {p0h:.4f},  |β'|² = {p1h:.4f},  suma = {p0h + p1h:.4f}")

    h("PASO 6 · Interpretación: interferencia")
    f(f"Término de interferencia: 2·Re(αβ*) = {2*inter:+.4f}")
    if inter > 1e-9:
        r("cos φ > 0  →  interferencia CONSTRUCTIVA hacia |0⟩ y destructiva hacia |1⟩")
    elif inter < -1e-9:
        r("cos φ < 0  →  interferencia DESTRUCTIVA hacia |0⟩ y constructiva hacia |1⟩")
    else:
        r("cos φ = 0  →  sin interferencia: queda 50 % / 50 %")
    f(f"Casos límite:  φ = 0°   → P'(0) = {0.5+k:.4f}")
    f(f"               φ = ±90° → P'(0) = 0.5000")
    f(f"               φ = 180° → P'(0) = {0.5-k:.4f}")
    f("Antes de H las probabilidades eran siempre 1/3 y 2/3 sin importar φ:")
    f("la fase solo se hace visible DESPUÉS de interferir con H.")
    return T


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("P1 · Qubit del sensor y compuerta Hadamard")
        self.geometry("1100x720")
        self.configure(bg=BG)

        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=10, pady=6)
        tk.Label(top, text="Fase φ (grados):", bg=BG, fg=FG, font=("Arial", 12)).pack(side="left")
        self.escala = tk.Scale(top, from_=-180, to=180, orient="horizontal", length=600,
                               bg=BG, fg=FG, highlightthickness=0, troughcolor=PANEL)
        self.escala.set(45)
        self.escala.pack(side="left", padx=10)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Pestaña 1: gráficas
        tab1 = tk.Frame(nb, bg=BG)
        nb.add(tab1, text="  Gráficas  ")
        self.fig = Figure(figsize=(10, 4.5), facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab1)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Pestaña 2: procedimiento
        tab2 = tk.Frame(nb, bg=BG)
        nb.add(tab2, text="  Procedimiento matemático  ")
        self.texto = tk.Text(tab2, bg=BG, fg=FG, font=("Consolas", 12), wrap="none",
                             padx=14, pady=10, borderwidth=0)
        sb = ttk.Scrollbar(tab2, command=self.texto.yview)
        self.texto.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.texto.pack(fill="both", expand=True)
        self.texto.tag_configure("h", foreground=CIAN, font=("Consolas", 13, "bold"))
        self.texto.tag_configure("f", foreground=FG)
        self.texto.tag_configure("r", foreground=NARANJA, font=("Consolas", 12, "bold"))

        self.escala.config(command=self.actualizar)
        self.actualizar()

    def actualizar(self, _=None):
        fase = int(self.escala.get())
        _, _, sv = construir_estado(fase)
        antes = calcular_probabilidades(sv)
        despues = calcular_probabilidades(sv.evolve(HGate()))

        self.fig.clear()
        gs = self.fig.subplots(1, 3)
        for ax, datos, tit in [(gs[0], antes, "Antes de H"), (gs[1], despues, "Después de H")]:
            ax.set_facecolor(PANEL)
            bars = ax.bar(["|0⟩", "|1⟩"], datos, color=[CIAN, NARANJA], edgecolor="white")
            for bar, v in zip(bars, datos):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v*100:.1f}%",
                        ha="center", color=FG, fontweight="bold")
            ax.set_ylim(0, 1.15)
            ax.set_title(tit, color=FG)
            ax.tick_params(colors=FG)

        ax = gs[2]
        ax.set_facecolor(PANEL)
        fases = np.linspace(-180, 180, 361)
        curva = 0.5 + (np.sqrt(2) / 3) * np.cos(np.deg2rad(fases))
        ax.plot(fases, curva, color=CIAN, lw=2.5, label="P'(0)")
        ax.plot(fases, 1 - curva, color=NARANJA, lw=2.5, label="P'(1)")
        ax.axvline(fase, color=MAGENTA, ls="--")
        ax.set_title("Interferencia vs fase", color=FG)
        ax.set_xlabel("φ (°)", color=FG)
        ax.tick_params(colors=FG)
        ax.legend(facecolor=BG, labelcolor=FG, fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

        self.texto.config(state="normal")
        self.texto.delete("1.0", "end")
        for txt, tag in procedimiento(fase):
            self.texto.insert("end", txt, tag)
        self.texto.config(state="disabled")


if __name__ == "__main__":
    App().mainloop()