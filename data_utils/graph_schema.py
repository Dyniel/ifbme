# data_utils/graph_schema.py

"""
Defines the schema (node types and edge types) for heterogeneous patient graphs.
"""

# Node Types
# T_t – Time-slice (e.g., 1 h bin)
# V_i – Vital/Lab concept (HR, WBC, Lactate …)
# D_k – Diagnosis code (ICD-10 / SNOMED)
# M_j – Medication class (ATC-4)
# P_l – Procedure (CPT / OPS)
NODE_TYPES = [
    'timeslice',  # Represents a binned moment in time for a patient (T_t)
    'vital',      # Represents a type of vital sign or lab test (V_i)
    'diagnosis',  # Represents an ICD code (D_k)
    'medication', # Represents an ATC medication code/class (M_j)
    'procedure',  # Represents a procedure code (P_l)
]

# Edge Types
# Format: (source_node_type, relation_name, destination_node_type)
# E_temp: (T_t → T_{t+1}), feature = Δt, kolejność
# E_meas: (T_t ↔ V_i) jeśli dana zmierzona w tym binie (edge feature = z-score value)
# E_diag / med / proc: (T_t ↔ D_k / M_j / P_l) jeśli kod aktywny
# (opc.) E_ont: (D_k ↔ D_parent) z ontologii, tylko w pre-trainingu (Not included initially)

EDGE_TYPES = [
    # Temporal connections between time slices
    ('timeslice', 'temporally_precedes', 'timeslice'),

    # Connections between time slices and vital/lab concepts for measurements
    ('timeslice', 'has_vital_measurement', 'vital'),
    ('vital', 'vital_measured_in', 'timeslice'), # Reverse of has_vital_measurement

    # Connections between time slices and diagnosis codes
    ('timeslice', 'has_diagnosis', 'diagnosis'),
    ('diagnosis', 'diagnosis_active_in', 'timeslice'), # Reverse of has_diagnosis

    # Connections between time slices and medication codes/classes
    ('timeslice', 'has_medication', 'medication'),
    ('medication', 'medication_given_in', 'timeslice'), # Reverse of has_medication

    # Connections between time slices and procedure codes
    ('timeslice', 'has_procedure', 'procedure'),
    ('procedure', 'procedure_performed_in', 'timeslice'), # Reverse of has_procedure

    # Potentially, relationships between concept nodes themselves if using an ontology
    # e.g., ('diagnosis', 'is_a_parent_of', 'diagnosis') - for future extension
]

# Example of how to access a specific edge type string for PyG:
# edge_key = ('timeslice', 'temporally_precedes', 'timeslice')
# data[edge_key].edge_index = ...

if __name__ == '__main__':
    print("Node Types:")
    for nt in NODE_TYPES:
        print(f"- {nt}")

    print("\nEdge Types (Canonical PyG Format):")
    for et in EDGE_TYPES:
        print(f"- {et}")
