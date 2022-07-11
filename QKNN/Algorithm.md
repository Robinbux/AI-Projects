# Quantum K-Nearest Neighbors

## Variable Descriptions

1. **test states** -- $\{u_n\}$: Collection of vectors of unknown labels
2. **train states** -- $\{v_n\}$: Collection of vectors of known labels
---
1. **$\mathcal{H}$**: n-qubit Hilbert space of dimensions $N = 2^n$
2. **$\ket{\psi}\in\mathcal{H}$**: Unknown test state whose label is to be determined.
3. **$\{\ket{\phi_j}:j\in\{0,\dots, M-1\}\}\subset\mathcal{H}$**: Collection of $M = 2^m$ train states whose labels are known to us.
---
1. **$F_j = F(\psi,\phi_j)=|\braket{\psi|\phi_j}|^2$**: Fidelity between the test state $\ket\psi$ and the j state $\ket{\phi_j}$
2. **$F=[F_0,\dots,F_{M-1}]$**: Table of length $M$ containing the fidelities with the test state $\ket{\psi}$ and all train states $\{\ket{\phi_j}\}$
3. 
