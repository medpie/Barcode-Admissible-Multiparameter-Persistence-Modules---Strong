# Persistence Module Barcode Admissibility Checker and Barcode Finder

This project provides algorithms to handle **persistence modules (pModules)**, focusing on verifying **barcode admissibility** and finding barcodes.

---

## Copyright

Copyright (c) 2024 Mehdi Nategh
This code is licensed under the MIT License. See the LICENSE file for details.

---

## Overview

This program is designed to find a **minimal set of generators `S`** for a given persistence module and then verify if the module is **barcode admissible**. Assuming the admissibility is verified, it also provides a dictionary of barcodes.

### pModule Structure

A `pModule` is a Python dictionary where:
* **Keys** are `d`-dimensional tuples representing indices (grades) `z = (z_1, z_2, ..., z_d)`.
* **Values** are lists containing two elements:
    1.  The **basis matrix** (`numpy.array`) of dimension `dim(M_z)`, representing the vector space at that node `M_z`.
    2.  A **list of linear transformation matrices** (`numpy.array`), representing the maps to the next adjacent nodes along each dimension:
        * `M_z -> M_{z + (1, 0, 0, ..., 0)}`
        * `M_z -> M_{z + (0, 1, 0, ..., 0)}`
        * ...
        * `M_z -> M_{z + (0, 0, ..., 0, 1)}`

---

## Functionality

### `pModuleMaps(pModule, x, y)`

This function computes the **composed linear transformation matrix** from a node `x` to a node `y` within the `pModule`, following a lexicographical path. This is crucial for tracking how basis vectors evolve across the module's grid.

### `minimal_generator(pModule)`

This function identifies a **minimal set of generators `S`** for the persistence module. These generators are conceptually similar to "birth points" in single-parameter persistence, marking where fundamental topological features emerge.

### `barcode_admissible(pModule)`

This is the core function that checks whether a `pModule` is **barcode admissible**. This property is significant because it determines if the persistence module can be fully decomposed into a direct sum of interval modules, which in turn means it can be represented by a standard barcode. This greatly simplifies its interpretation and stability analysis in Topological Data Analysis.

The admissibility check is based on **Proposition 2.38**:

> **[Proposition 2.38]** An $\mathbb{N}^d$-indexed persistence module $M$ has a barcode if and only if $M$ has a set $S$ of generators such that for every $x \in \mathbb{N}^d$, the set $\{ M_{gr(s),x}(s) \mid s \in S, gr(s) \leqslant x \text{ and } M_{gr(s),x}(s) \neq 0 \}$ is a linear basis of $M_x$.
>
> (Nategh M., Qin Z., and Wang Sh. *Multiparameter Persistence Modules*, Doctoral Dissertation, University of Missouri-Columbia, ProQuest ID: 12577, (2025)).

### `barcodes` (Conceptual)

The description mentions a `barcodes` function. This (currently conceptual) function would, for barcode-admissible modules, generate a dictionary where keys correspond to individual generators that are vectors born at an index s, and values are sets of indices `z` indicating the nodes up to which each generator's topological feature survives. This function's utility is contingent on the decomposability of the module being verified by the `is_barcode_admissible` function.
