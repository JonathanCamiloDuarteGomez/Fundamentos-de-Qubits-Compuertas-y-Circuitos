
import numpy as np
import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector
from qiskit.circuit.library import HGate


# ==========================================================
# 1. CONFIGURACIÓN MATEMÁTICA DEL QUBIT
# ==========================================================

def construir_estado(fase_grados):
    """
    Construye el estado:
        |psi> = (1/sqrt(3))|0> + sqrt(2/3)*e^(i*phi)|1>

    La fase se puede modificar sin cambiar las probabilidades
    iniciales.
    """

    phi = np.deg2rad(fase_grados)

    alpha = 1 / np.sqrt(3)
    beta = np.sqrt(2 / 3) * np.exp(1j * phi)

    estado = Statevector([alpha, beta])

    if not estado.is_valid():
        raise ValueError("El estado no está normalizado.")

    return alpha, beta, estado


# ==========================================================
# 2. CÁLCULO DE PROBABILIDADES
# ==========================================================

def calcular_probabilidades(estado):
    """
    Aplica la regla de Born:
        P(0) = |alpha|²
        P(1) = |beta|²
    """

    probabilidades = estado.probabilities()

    return probabilidades[0], probabilidades[1]


# ==========================================================
# 3. FORMATO DE NÚMEROS COMPLEJOS
# ==========================================================

def mostrar_complejo(z):
    return f"{z.real:+.4f} {z.imag:+.4f}j"


# ==========================================================
# 4. VISUALIZACIÓN DINÁMICA
# ==========================================================

def visualizar_experimento(fase_grados=45, shots=1000):

    # Construcción del estado
    alpha, beta, estado = construir_estado(fase_grados)

    # Probabilidades antes de Hadamard
    p0, p1 = calcular_probabilidades(estado)

    # Aplicación de la compuerta Hadamard
    estado_h = estado.evolve(HGate())

    # Probabilidades después de Hadamard
    p0_h, p1_h = calcular_probabilidades(estado_h)

    # Comprobación matemática independiente
    phi = np.deg2rad(fase_grados)

    p0_teorica = 0.5 + (np.sqrt(2) / 3) * np.cos(phi)
    p1_teorica = 0.5 - (np.sqrt(2) / 3) * np.cos(phi)

    assert np.allclose(
        [p0_h, p1_h],
        [p0_teorica, p1_teorica]
    )

    # Muestreo de mediciones después de Hadamard
    conteos = estado_h.sample_counts(shots=shots)

    # ======================================================
    # INFORMACIÓN NUMÉRICA
    # ======================================================

    print("=" * 58)
    print(" SIMULADOR DE SENSOR CUÁNTICO - ANÁLISIS DE FASE")
    print("=" * 58)

    print(f"\nFase configurada: {fase_grados} grados")

    print("\nAMPLITUDES INICIALES")
    print(f"Alpha (|0>): {mostrar_complejo(alpha)}")
    print(f"Beta  (|1>): {mostrar_complejo(beta)}")

    print("\nMEDICIÓN INICIAL")
    print(f"P(0): {p0:.4f} ({p0 * 100:.2f}%)")
    print(f"P(1): {p1:.4f} ({p1 * 100:.2f}%)")

    print("\nDESPUÉS DE APLICAR HADAMARD")

    print(f"Amplitud de |0>: {mostrar_complejo(estado_h.data[0])}")
    print(f"Amplitud de |1>: {mostrar_complejo(estado_h.data[1])}")

    print(f"P(0): {p0_h:.4f} ({p0_h * 100:.2f}%)")
    print(f"P(1): {p1_h:.4f} ({p1_h * 100:.2f}%)")

    print("\nMUESTREO DE MEDICIONES")
    print(f"Cantidad de mediciones: {shots}")
    print(f"Resultados: {conteos}")

    print("\nCOMPROBACIÓN")
    print(f"Norma del estado: {np.linalg.norm(estado.data):.4f}")
    print(f"Norma después de H: {np.linalg.norm(estado_h.data):.4f}")

    print("La simulación coincide con la fórmula teórica.")
    print("=" * 58)

    # ======================================================
    # 5. GRÁFICAS DE ALTO CONTRASTE
    # ======================================================

    COLOR_FONDO = "#101820"
    COLOR_PANEL = "#192936"
    COLOR_CIAN = "#00E5FF"
    COLOR_NARANJA = "#FFB000"
    COLOR_MAGENTA = "#FF4FA3"
    COLOR_VERDE = "#7CFF6B"
    COLOR_TEXTO = "#FFFFFF"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fig.patch.set_facecolor(COLOR_FONDO)

    for ax in axes.flat:
        ax.set_facecolor(COLOR_PANEL)
        ax.tick_params(colors=COLOR_TEXTO)

        for spine in ax.spines.values():
            spine.set_color("#8395A7")

        ax.xaxis.label.set_color(COLOR_TEXTO)
        ax.yaxis.label.set_color(COLOR_TEXTO)

        ax.title.set_color(COLOR_TEXTO)
        ax.grid(color="#526575", alpha=0.35)

    fig.suptitle(
        "COMPORTAMIENTO DE UN QUBIT SEGÚN SU FASE",
        color=COLOR_TEXTO,
        fontsize=17,
        fontweight="bold"
    )

    # ------------------------------------------------------
    # GRÁFICA 1: REPRESENTACIÓN DE AMPLITUDES COMPLEJAS
    # ------------------------------------------------------

    ax = axes[0, 0]

    circulo = np.linspace(0, 2 * np.pi, 300)

    ax.plot(
        np.cos(circulo),
        np.sin(circulo),
        "--",
        color="#748A9E",
        alpha=0.6
    )

    # Amplitud alpha
    ax.annotate(
        "",
        xy=(alpha.real, alpha.imag),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="->",
            color=COLOR_VERDE,
            lw=3
        )
    )

    # Amplitud beta
    ax.annotate(
        "",
        xy=(beta.real, beta.imag),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="->",
            color=COLOR_MAGENTA,
            lw=3
        )
    )

    ax.text(
        alpha.real,
        alpha.imag + 0.08,
        "Alpha |0>",
        color=COLOR_VERDE,
        ha="center"
    )

    ax.text(
        beta.real,
        beta.imag + 0.08,
        "Beta |1>",
        color=COLOR_MAGENTA,
        ha="center"
    )

    ax.axhline(0, color="white", alpha=0.5)
    ax.axvline(0, color="white", alpha=0.5)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_aspect("equal")

    ax.set_xlabel("Parte real")
    ax.set_ylabel("Parte imaginaria")

    ax.set_title("1. Amplitudes en el plano complejo")

    # ------------------------------------------------------
    # GRÁFICA 2: PROBABILIDADES INICIALES
    # ------------------------------------------------------

    ax = axes[0, 1]

    barras = ax.bar(
        ["|0>", "|1>"],
        [p0, p1],
        color=[COLOR_CIAN, COLOR_NARANJA],
        edgecolor="white",
        linewidth=1.5
    )

    for barra, valor in zip(barras, [p0, p1]):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            valor + 0.03,
            f"{valor * 100:.2f}%",
            ha="center",
            color=COLOR_TEXTO,
            fontweight="bold"
        )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Probabilidad")
    ax.set_title("2. Probabilidades antes de Hadamard")

    # ------------------------------------------------------
    # GRÁFICA 3: PROBABILIDADES DESPUÉS DE HADAMARD
    # ------------------------------------------------------

    ax = axes[1, 0]

    barras = ax.bar(
        ["|0>", "|1>"],
        [p0_h, p1_h],
        color=[COLOR_CIAN, COLOR_NARANJA],
        edgecolor="white",
        linewidth=1.5
    )

    for barra, valor in zip(barras, [p0_h, p1_h]):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            valor + 0.03,
            f"{valor * 100:.2f}%",
            ha="center",
            color=COLOR_TEXTO,
            fontweight="bold"
        )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Probabilidad")
    ax.set_title("3. Probabilidades después de Hadamard")

    # ------------------------------------------------------
    # GRÁFICA 4: VARIACIÓN DE PROBABILIDAD SEGÚN LA FASE
    # ------------------------------------------------------

    ax = axes[1, 1]

    fases = np.linspace(-180, 180, 721)
    radianes = np.deg2rad(fases)

    curva_p0 = 0.5 + (np.sqrt(2) / 3) * np.cos(radianes)
    curva_p1 = 0.5 - (np.sqrt(2) / 3) * np.cos(radianes)

    ax.plot(
        fases,
        curva_p0,
        color=COLOR_CIAN,
        linewidth=2.8,
        label="P(0) después de H"
    )

    ax.plot(
        fases,
        curva_p1,
        color=COLOR_NARANJA,
        linewidth=2.8,
        label="P(1) después de H"
    )

    # Marca la fase seleccionada
    ax.axvline(
        fase_grados,
        color=COLOR_MAGENTA,
        linestyle="--",
        linewidth=2,
        label=f"Fase actual: {fase_grados}°"
    )

    ax.scatter(
        [fase_grados, fase_grados],
        [p0_h, p1_h],
        color=[COLOR_CIAN, COLOR_NARANJA],
        edgecolors="white",
        s=100,
        zorder=5
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 1.05)

    ax.set_xlabel("Fase (grados)")
    ax.set_ylabel("Probabilidad")

    ax.set_title("4. Interferencia según la fase")

    ax.legend(
        facecolor=COLOR_FONDO,
        labelcolor=COLOR_TEXTO,
        fontsize=8
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


if __name__ == "__main__":
    visualizar_experimento()
