# Persistence Module Barcode Admissibility Checker

"""
This repository contains Python code for analyzing **persistence modules** and determining if they are **barcode admissible**. Persistence modules are fundamental structures in **Persistent Homology**, a powerful tool in **Topological Data Analysis (TDA)** used to study the "shape" of data. Barcode admissibility is a key property that relates to the stability and interpretability of persistence barcodes.

---

## What is a Persistence Module?

In the context of this code, a persistence module is represented as a dictionary (`pModule`). Each key in the dictionary is a **tuple representing a node** (e.g., `(0, 0)`, `(1, 0)`) in a multi-dimensional grid. Each value associated with a node is a list containing two elements:

1.  A **basis matrix** (`numpy.array`) representing the vector space at that node.
2.  A **list of linear transformation matrices** (`numpy.array`), where each matrix represents the map to the next node along a specific dimension. The order of these matrices corresponds to the dimensions.

---

## `pModuleMaps(pModule, x, y)`

This function computes the **composed linear transformation matrix** along a lexicographical path from a starting node `x` to an ending node `y` within the `pModule`.

### How it Works:

* It takes a `pModule` (the persistence module), a starting node `x` (as a tuple), and an ending node `y` (as a tuple).
* It navigates from `x` to `y` by incrementing one dimension at a time, following a lexicographical order.
* At each step, it multiplies the current accumulated transformation matrix by the map matrix associated with the current dimension step.
* **Error Handling:** Includes robust checks for dimension mismatches, missing keys, and malformed `pModule` entries to ensure valid matrix multiplications and paths.

---

## `minimal_generator(pModule)`

This function identifies a **minimal set of generators** for the persistence module. These generators are analogous to "birth points" in traditional 1-parameter persistence homology, representing fundamental topological features that "appear" at specific nodes.

### How it Works:

* It iterates through all nodes `z` in the `pModule` in a sorted order.
* For each node `z`, it collects the images of all basis vectors from "previous" nodes (nodes `s` such that `s <= z` element-wise) under the composed maps `pModuleMaps(pModule, s, z)`.
* It then finds a minimal set of basis vectors at node `z` that are linearly independent of these incoming images. These linearly independent vectors constitute the generators born at `z`.
* The function returns a dictionary where keys are nodes and values are the `numpy.array` of generator vectors born at that node.

---

## `is_barcode_admissible(pModule)`

This is the main function that determines if the given persistence module is **barcode admissible**. A persistence module is barcode admissible if, at every node `z`, the vectors in the basis at `z` that are images of generators born at nodes `s <= z` span the entire vector space at `z`. This property is crucial for the module to be "barcode-like" and for its persistence barcode to accurately represent the underlying topological features.

### How it Works:

* It first calls `minimal_generator(pModule)` to find all generators for the module.
* For each node `z` in the `pModule`:
    * It identifies all "birth points" `s` (nodes where generators are born) that are "less than or equal to" `z` (i.e., within its lower cone).
    * It computes the images of the generators born at these `s` nodes under the composed maps `pModuleMaps(pModule, s, z)`.
    * It then checks if the span of these "incoming" generator images has a rank equal to the dimension of the basis at node `z`.
    * If this condition holds for all nodes `z`, the module is deemed barcode admissible. Otherwise, it is not.

---

## Example Usage

To use this code, ensure you have `numpy` installed (`pip install numpy`). You can then run the script directly. The code includes three examples to demonstrate the usage and output of the `barcode_admissible` function:

* **Example 1:** This is a barcode admissible module.
* **Example 2:** This is NOT a barcode admissible module.
* **Example 3:** This is NOT a barcode admissible module.

These examples are commented out in the provided code, but you can uncomment them to see the output.

```python

"""

import numpy as np 


def pModuleMaps(pModule, x, y):
    """
    Computes the composed linear transformation matrix along a lexicographical path
    from node x to node y in a persistence module.
    """
    x_arr = np.array(x)
    y_arr = np.array(y)

    if x_arr.shape != y_arr.shape:
        raise ValueError("Input points x and y must have the same number of dimensions.")
    if not np.all(x_arr <= y_arr):
        raise ValueError("Point x must be less than or equal to point y element-wise for this path algorithm.")

    d = len(x_arr) 

    try:
        current_transformation_matrix = pModule[tuple(x_arr)][0]
    except KeyError:
        raise ValueError(f"Starting point {tuple(x_arr)} not found in pModule.")
    except IndexError:
        raise ValueError(f"pModule entry for {tuple(x_arr)} is malformed. Expected [matrix, [list_of_maps]].")

    current_node = np.copy(x_arr)

    for dim_idx in range(d):
        while current_node[dim_idx] < y_arr[dim_idx]:
            source_node_key = tuple(current_node)
            
            try:
                step_matrix = pModule[source_node_key][1][dim_idx]
            except KeyError:
                raise ValueError(f"Map from {source_node_key} along dimension {dim_idx} not found in pModule. Missing key or map for this dimension.")
            except IndexError:
                raise ValueError(f"pModule entry for {source_node_key} is malformed. Expected [matrix_at_point, [list_of_maps]] where list has at least {dim_idx+1} elements.")

            if step_matrix.shape[1] != current_transformation_matrix.shape[0]:
                    raise ValueError(
                        f"Dimension mismatch for map from {source_node_key} along dim {dim_idx}:\n"
                        f"Step matrix {step_matrix.shape} (expected cols to match current matrix rows {current_transformation_matrix.shape[0]})"
                        f" vs. Current accumulated matrix {current_transformation_matrix.shape}.\n"
                        f"This means the map from the current node's space isn't compatible with the accumulated transformation."
                    )
            
            try:
                current_transformation_matrix = step_matrix @ current_transformation_matrix
            except ValueError as e:
                raise ValueError(f"Dimension mismatch during multiplication: step_matrix (shape {step_matrix.shape}) @ current_transformation_matrix (shape {current_transformation_matrix.shape}) at node {source_node_key} in dimension {dim_idx}. Error: {e}")

            current_node[dim_idx] += 1

    return current_transformation_matrix

#===========================================================================
#===========================================================================

def minimal_generator(pModule):
    generators_by_node = {}
    first_key = next(iter(pModule))
    d = len(first_key)
    origin_key = tuple(np.zeros(d, dtype=int))
    if origin_key not in pModule:
        raise ValueError("The pModule must contain the origin (0,0,...) node for this generator algorithm.")
    indices = sorted(pModule.keys())
    for z_tuple in indices:
        z_arr = np.array(z_tuple)
        S_z_incoming_images = [] 
        if np.all(z_arr == 0):
            basis_at_z = pModule[z_tuple][0]
            current_node_generators = []
            for j in range(basis_at_z.shape[1]):
                vec = basis_at_z[:, j]
                if vec.ndim == 0: 
                    vec = np.array([vec.item()])
                if not np.all(vec == 0):
                    current_node_generators.append(vec)
            
            if current_node_generators: 
                generators_by_node[z_tuple] = np.array(current_node_generators)
            continue
        
        for i in range(d): 
            prev_z_arr = np.copy(z_arr)
            prev_z_arr[i] -= 1 
            prev_z_tuple = tuple(prev_z_arr)

            if np.all(prev_z_arr >= 0) and prev_z_tuple in pModule:
                try:
                    map_from_prev_to_z = pModuleMaps(pModule, prev_z_tuple, z_tuple)
                except ValueError as e:
                    print(f"Warning: Could not compute map from {prev_z_tuple} to {z_tuple}. Skipping this incoming map. Error: {e}")
                    continue 

                basis_at_prev_z = pModule[prev_z_tuple][0]

                try:
                    image_vectors_matrix = map_from_prev_to_z @ basis_at_prev_z
                    
                    for col_idx in range(image_vectors_matrix.shape[1]):
                        vec = image_vectors_matrix[:, col_idx]
                        if vec.ndim == 0:
                            vec = np.array([vec.item()])
                        
                        if not np.all(vec == 0):
                            S_z_incoming_images.append(vec)

                except ValueError as e:
                    print(f"Warning: Dimension mismatch when applying map {map_from_prev_to_z.shape} to basis {basis_at_prev_z.shape} at {prev_z_tuple}. Skipping. Error: {e}")
                    continue

        basis_at_z = pModule[z_tuple][0]
        expected_dim_at_z = basis_at_z.shape[0]
        filtered_incoming_images = []
        for vec in S_z_incoming_images:
            if vec.shape[0] == expected_dim_at_z: 
                filtered_incoming_images.append(vec)
            else:
                print(f"Warning: Incoming image vector {vec.shape} from a previous step does not match current basis dimension {expected_dim_at_z} at {z_tuple}. Skipping this vector.")

        if len(filtered_incoming_images) > 0:
            incoming_span_matrix = np.vstack(filtered_incoming_images).T 
            incoming_rank = np.linalg.matrix_rank(incoming_span_matrix)
        else:
            incoming_span_matrix = np.empty((expected_dim_at_z, 0)) 
            incoming_rank = 0

        current_node_generators = [] 

        for j in range(basis_at_z.shape[1]): 
            current_basis_vector = basis_at_z[:, j]
            if current_basis_vector.ndim == 0:
                current_basis_vector = np.array([current_basis_vector.item()])

            if np.all(current_basis_vector == 0):
                continue
            test_matrix = np.hstack((incoming_span_matrix, current_basis_vector.reshape(-1, 1)))
            new_rank = np.linalg.matrix_rank(test_matrix)
            if new_rank > incoming_rank:
                current_node_generators.append(current_basis_vector)
                incoming_span_matrix = np.hstack((incoming_span_matrix, current_basis_vector.reshape(-1, 1)))
                incoming_rank = new_rank 
        if current_node_generators:
            generators_by_node[z_tuple] = np.array(current_node_generators)

    return generators_by_node
#============================================================================================
#============================================================================================
#============================================================================================

def is_barcode_admissible(pModule):
    generators = minimal_generator(pModule)
    S_keys = list(generators.keys())
    def lower_cone(z):
        z_arr = np.array(z)
        l_cone = []
        for s_keys in S_keys:
            s_arr = np.array(s_keys)
            if np.all(s_arr <= z_arr):
                l_cone.append(s_arr)
        return l_cone
    
    j = 0
    result = f'The module is barcode admissible.'
    while j < len(sorted(pModule.keys())):
        z = sorted(pModule.keys())[j]
        S_z = []
        birth_points = lower_cone(z)
        for index in birth_points:
            vectors_born_at_index = generators.get(tuple(index), None)
            map_s_to_z = pModuleMaps(pModule, index, z)
            matrix = map_s_to_z @ vectors_born_at_index.T
            for column in range(matrix.shape[1]):
                vec = matrix[:, column]
                if not np.all(vec == 0):
                    S_z.append(vec)
        S_z_unique = set(tuple(arr) for arr in S_z) 
        basis_at_z = pModule[z][0]
        rank_at_z = np.linalg.matrix_rank(basis_at_z)
        if  (rank_at_z != 0) and (len(S_z_unique) == basis_at_z.shape[0]):
            j += 1
        elif rank_at_z == 0:
            j += 1
        elif (rank_at_z != 0) and (len(S_z_unique) != basis_at_z.shape[0]):     
            result = f'The module is not barcode admissible.' 
            break
    return result


#=======================================================================

def barcodes(pModule):
    if is_barcode_admissible(pModule) == f'The module is barcode admissible.':
        generators = minimal_generator(pModule)
        birth_points = list(generators.keys())
        points = np.array(list(pModule.keys()))
        barcodes_dictionary = {}
        for s in birth_points:
            Vectors = generators.get(tuple(s), None)
            mask = np.all(points >= s, axis=1)
            upper_points = points[mask]
            for v in Vectors:
                values = []
                for index in upper_points:
                    image_of_v = pModuleMaps(pModule, s, index) @ v
                    if np.any(image_of_v) != 0:
                        value = tuple(int(coord) for coord in index)
                        values.append(value)
                        unique_values = list(set(values))
                        barcodes_dictionary.update({(s, f'{v}'): unique_values})
        result = barcodes_dictionary 
    elif f'The module is not barcode admissible':
        result = f'The module is not barcode admissible hence, there is no well-defined barcode.'
    return result


"""
#============================================= Examples


# Example 1: This is a barcode admissible module.

p_Module = {}
p_Module.update({(0, 0): [np.eye(3,3), [np.eye(3, 3), np.eye(3, 3)]]})
p_Module.update({(1, 0): [np.eye(3,3), [np.eye(1,3), np.array([[1, 0, 0], [0, 0, 1]])]]})
p_Module.update({(2, 0): [np.eye(1,1), [np.zeros((1,1)), np.eye(1,1)]]})
p_Module.update({(0, 1): [np.eye(3,3), [np.array([[1, 0, 0], [0, 0, 1]]), np.array([[1, 0, 0], [0, 0, 1]])]]})
p_Module.update({(0, 2): [np.eye(2,2), [np.array([[0, 1]]), np.zeros((1,2))]]})
p_Module.update({(1, 1): [np.eye(2,2), [np.array([[1, 0]]), np.array([[0, 1]])]]})
p_Module.update({(1, 2): [np.eye(1,1), [np.zeros((1,1)), np.zeros((1,1))]]})
p_Module.update({(2, 2): [np.zeros((1,1)), [np.zeros((1,1)), np.zeros((1,1))]]})
p_Module.update({(2, 1): [np.eye(1,1), [np.zeros((1,1)), np.zeros((1,1))]]})

#print(is_barcode_admissible(p_Module))
#print(barcodes(p_Module))



#==================================================================================

#Example 2: This is NOT a barcode admissible module
p_Module = {}
p_Module.update({(0, 0): [np.zeros((1,1)), [np.zeros((1,1)), np.zeros((0,1))]]})
p_Module.update({(1, 0): [np.eye(1,1), [np.zeros((1,1)), np.array([[1]])]]})
p_Module.update({(0, 1): [np.eye(1,1), [np.array([[2]]), np.zeros((1,1))]]})
p_Module.update({(1, 1): [np.eye(1,1), [np.zeros((1,1)), np.zeros((1,1))]]})

#print(is_barcode_admissible(p_Module))
#print(barcodes(p_Module))


#============================================================================
#Example 3: This is NOT a barcode admissible module

p_Module = {}
p_Module.update({(0, 0): [np.zeros((1,1)), [np.zeros((1,1)), np.zeros((0,1))]]})
p_Module.update({(1, 0): [np.eye(2,2), [np.zeros((1,1)), np.array([[1, 0]])]]})
p_Module.update({(0, 1): [np.eye(2,2), [np.array([[0, 2]]) , np.zeros((1,1))]]})
p_Module.update({(1, 1): [np.eye(1,1), [np.zeros((1,1)), np.zeros((1,1))]]})

print(is_barcode_admissible(p_Module))

"""
