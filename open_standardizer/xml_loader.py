import xml.etree.ElementTree as ET
from .cpu_ops import cpu_execute


# --------------------------------------------------------------------
# MAP CHEMAXON ACTION NAMES → RDKit CPU OP KEYS (no "op_" prefix)
# --------------------------------------------------------------------
ACTION_MAP = {
    "ClearStereo": "clear_stereo",
    "RemoveFragment": "remove_fragment_keeplargest",  # attrs can refine this
    "StripSalts": "strip_salts_default_keep_last",    # attrs refine this
    "RemoveAttachedData": "remove_attached_data",
    "RemoveAtomValues": "remove_atom_values",
    "RemoveExplicitH": "remove_explicit_h",
    "ClearIsotopes": "clear_isotopes",
    "ClearAtomValues": "clear_atom_values",
    "CleanGeometry": "clean_geometry",
    "Neutralize": "neutralize",
    "Mesomerize": "mesomerize",
    "Tautomerize": "tautomerize",
    "Aromatize": "aromatize",
}


# --------------------------------------------------------------------
# Convert XML <Action> → execution dictionary
# --------------------------------------------------------------------
def _convert_action(xml_action):
    tag = xml_action.tag  # e.g. "ClearStereo"

    if tag not in ACTION_MAP:
        raise ValueError(f"Unsupported XML action: {tag}")

    attrs = {k: v for k, v in xml_action.attrib.items()}
    op_name = ACTION_MAP[tag]

    # Special handling for RemoveFragment variants
    if tag == "RemoveFragment":
        # Example ChemAxon attrs:
        #   Measure="heavyAtomCount" Method="keepLargest"
        method = attrs.get("Method", "").lower()
        measure = attrs.get("Measure", "").lower()

        if method == "keeplargest" and measure == "heavyatomcount":
            op_name = "remove_fragment_keeplargest"
        elif method == "removesmallest" and measure == "heavyatomcount":
            op_name = "remove_fragment_smallest"
        else:
            # Fallback to keeplargest if we don't recognize the combo
            op_name = "remove_fragment_keeplargest"

    elif tag == "StripSalts":
        # ChemAxon sample:
        #   dontremovelastcomponent="true" usedefaultsalts="true"
        dont_last = attrs.get("dontremovelastcomponent", "true").lower()
        use_default = attrs.get("usedefaultsalts", "true").lower()

        if use_default == "true":
            if dont_last == "true":
                op_name = "strip_salts_default_keep_last"
            else:
                op_name = "strip_salts_default_allow_empty"
        else:
            # usedefaultsalts="false" → placeholder ops (currently same behavior
            # as default until you plug in a custom salt list)
            if dont_last == "true":
                op_name = "strip_salts_custom_keep_last"
            else:
                op_name = "strip_salts_custom_allow_empty"

    return {
        "op": op_name,
        "attrs": attrs,    # saved for future extensions
        "xml_tag": tag,
    }


# --------------------------------------------------------------------
# LOAD XML FILE → ordered list of operations
# --------------------------------------------------------------------
def load_xml_actions(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # <StandardizerConfiguration><Actions>...</Actions>
    actions_block = root.find("Actions")
    if actions_block is None:
        raise ValueError("XML missing <Actions> block")

    ops = []
    for elem in actions_block:
        ops.append(_convert_action(elem))

    return ops


# --------------------------------------------------------------------
# EXECUTE list of ChemAxon-equivalent RDKit operations
# --------------------------------------------------------------------
def run_xml_actions(smiles: str, actions):
    """
    SMILES → SMILES standardization using CPU ops only.
    """
    current = smiles
    for entry in actions:
        op_key = entry["op"]  # e.g. "strip_salts_default_keep_last"
        mol = Chem.MolFromSmiles(current)
        if mol is None:
            return None
        current = cpu_execute(op_key, mol)  # returns SMILES
    return current


# --------------------------------------------------------------------
# HELPER: load XML + standardize in one call
# --------------------------------------------------------------------
def standardize_with_xml_smiles(smiles: str, xml_path: str) -> str | None:
    actions = load_xml_actions(xml_path)
    return run_xml_actions(smiles, actions)

