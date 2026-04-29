def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    ref_total = sum(reference_counts)
    prod_total = sum(production_counts)

    tvd = 0.5 * sum(
        abs((ref / ref_total) - (prod / prod_total))
        for ref, prod in zip(reference_counts, production_counts)
    )

    return {
        "score": tvd,
        "drift_detected": tvd > threshold
    }