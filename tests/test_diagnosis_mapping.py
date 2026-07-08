from ecg_analyzer import _canonical_diagnosis, CLASSES


def test_rule_names_map_to_canonical_classes():
    assert _canonical_diagnosis("Normal Sinus Rhythm") == "Normal"
    assert _canonical_diagnosis("Atrial Fibrillation") == "AFib"
    assert _canonical_diagnosis("Sinus Tachycardia") == "Tachycardia"
    assert _canonical_diagnosis("Bradycardia") == "Bradycardia"


def test_mapped_names_are_valid_classes():
    for verbose in ("Normal Sinus Rhythm", "Atrial Fibrillation",
                    "Sinus Tachycardia", "Bradycardia"):
        assert _canonical_diagnosis(verbose) in CLASSES


def test_non_canonical_labels_pass_through():
    # VT is rules-only with no ML class; INCONCLUSIVE is a non-diagnosis.
    assert _canonical_diagnosis("Ventricular Tachycardia") == "Ventricular Tachycardia"
    assert _canonical_diagnosis("INCONCLUSIVE") == "INCONCLUSIVE"
    assert _canonical_diagnosis("PVC") == "PVC"
