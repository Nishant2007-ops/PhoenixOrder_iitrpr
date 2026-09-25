def generate_defense_recommendations(
    behavior_evidence,
    stage="Unknown",
    mitre_tactic="Unknown",
):
    recommendations = []

    evidence_text = " ".join(
        behavior_evidence
    ).lower()

    if "flow_count increase" in evidence_text:
        recommendations.append(
            "Monitor connection-rate spikes and investigate unusual source concentration."
        )

    if "total_packets_sum increase" in evidence_text:
        recommendations.append(
            "Inspect packet-volume changes and identify hosts generating abnormal traffic."
        )

    if "total_bytes_sum increase" in evidence_text:
        recommendations.append(
            "Monitor bandwidth consumption and investigate unusual high-volume flows."
        )

    if "forward_packets_sum increase" in evidence_text:
        recommendations.append(
            "Inspect outbound packet activity for unexpected source-to-destination patterns."
        )

    if "backward_packets_sum increase" in evidence_text:
        recommendations.append(
            "Inspect inbound response traffic and identify unusual destination activity."
        )

    if "syn_count_sum increase" in evidence_text:
        recommendations.append(
            "Inspect connection attempts and exposed services for abnormal SYN activity."
        )

    if "rst_count_sum increase" in evidence_text:
        recommendations.append(
            "Investigate abnormal connection resets and possible service disruption."
        )

    if "psh_count_sum increase" in evidence_text:
        recommendations.append(
            "Inspect application-level traffic for unusual bursts of data transmission."
        )

    if not recommendations:
        recommendations.append(
            "Continue monitoring the network state for persistent behavioral changes."
        )

    return {
        "stage": stage,
        "mitre_tactic": mitre_tactic,
        "recommendations": recommendations,
    }
