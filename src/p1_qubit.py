import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import HGate


def estado_sensor(fase=True):
    """Qubit del sensor: a|0> + beta|1>, con a hallado por normalizacion."""
    beta = (1 + 1j) / np.sqrt(3) if fase else np.sqrt(2 / 3)
    a = np.sqrt(1 - abs(beta) ** 2)          # |a|^2 + |beta|^2 = 1
    return a, Statevector([a, beta])


def probabilidades(sv):
    p = sv.probabilities()                    # regla de Born: |amplitud|^2
    return p[0], p[1]


def main():
    a, sv = estado_sensor(fase=True)
    print(f"a = {a:.4f}  (1/sqrt(3) = {1/np.sqrt(3):.4f})")
    print("Estado valido:", sv.is_valid())
    p0, p1 = probabilidades(sv)
    print(f"P(0) = {p0:.4f}   P(1) = {p1:.4f}")
    print("Muestreo 1000 shots:", dict(sv.sample_counts(shots=1000)))

    # Parte c): mismas probabilidades, distinta fase -> distinto resultado tras H
    _, sv_sin_fase = estado_sensor(fase=False)
    for nombre, estado in [("con fase (1+i)", sv), ("sin fase (real)", sv_sin_fase)]:
        antes = probabilidades(estado)
        despues = probabilidades(estado.evolve(HGate()))
        print(f"{nombre}: antes {antes[0]:.3f}/{antes[1]:.3f} -> despues de H {despues[0]:.3f}/{despues[1]:.3f}")


if __name__ == "__main__":
    main()