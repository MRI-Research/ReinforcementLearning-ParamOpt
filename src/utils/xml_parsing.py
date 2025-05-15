
import xmltodict
import xml.etree.ElementTree as ET

def strip_ns(elem):
    """Recursively remove namespace prefixes so tags become local names."""
    for e in elem.iter():
        if "}" in e.tag:
            e.tag = e.tag.split("}", 1)[1]

def pick_relaxation_constants(field_T):
    """
    Return (T1_ms: longitudinal relaxation times, T2_ms: transverse relaxation time) for either 1.5 T or 3 T for FastMRI Knee, based on field strength.
    Thresholds chosen to split the two clusters in fastMRI knee data.
    """
    if 1.3 <= field_T <= 1.7:
        # 1.5 T class
        T1, T2 = 950.0, 45.0  
    elif 2.6 <= field_T <= 3.2:
        # 3 T class
        T1, T2 = 1250.0, 38.0  
    else:
        raise ValueError(f"Field strength {field_T} T outside expected fastMRI ranges")
    return T1, T2

def parse_header(xml_bytes):
    """
    Parse the ISMRMRD header of FastMRI knee to extract TR, TE, T1c, T2c.
    """
    xml_str = (
        xml_bytes.decode("utf-8")
        if isinstance(xml_bytes, (bytes, bytearray))
        else b"".join(xml_bytes).decode("utf-8")
    )
    field_T = float(xmltodict.parse(xml_str)
                ["ismrmrdHeader"]["acquisitionSystemInformation"]
                ["systemFieldStrength_T"])
    T1c, T2c = pick_relaxation_constants(field_T)

    root = ET.fromstring(xml_str)
    strip_ns(root)
    seqp = root.find(".//sequenceParameters")
    if seqp is None:
        all_tags = {e.tag for e in root.iter()}
        raise KeyError(f"<sequenceParameters> not found! Available tags: {all_tags}")
    tr = float(seqp.findtext("TR"))
    te = float(seqp.findtext("TE"))
    
    return tr, te,  T1c, T2c