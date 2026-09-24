
import pandas as pd


def safe_value(dataframe, feature):
    row = dataframe[
        dataframe["feature"] == feature
    ]

    if len(row) == 0:
        return None

    return float(
        row.iloc[0]["normalized_change"]
    )


def build_behavioral_indicators(
    transition_dataframe
):

    indicators = {}

    total_packets = safe_value(
        transition_dataframe,
        "total_packets_sum"
    )

    total_bytes = safe_value(
        transition_dataframe,
        "total_bytes_sum"
    )

    flow_count = safe_value(
        transition_dataframe,
        "flow_count"
    )

    syn_count = safe_value(
        transition_dataframe,
        "syn_count_sum"
    )

    rst_count = safe_value(
        transition_dataframe,
        "rst_count_sum"
    )

    ack_count = safe_value(
        transition_dataframe,
        "ack_count_sum"
    )

    forward_packets = safe_value(
        transition_dataframe,
        "forward_packets_sum"
    )

    backward_packets = safe_value(
        transition_dataframe,
        "backward_packets_sum"
    )

    down_up_ratio = safe_value(
        transition_dataframe,
        "down_up_ratio_mean"
    )

    indicators["traffic_volume_change"] = (
        total_packets
        if total_packets is not None
        else 0.0
    )

    indicators["byte_volume_change"] = (
        total_bytes
        if total_bytes is not None
        else 0.0
    )

    indicators["flow_volume_change"] = (
        flow_count
        if flow_count is not None
        else 0.0
    )

    indicators["syn_activity_change"] = (
        syn_count
        if syn_count is not None
        else 0.0
    )

    indicators["rst_activity_change"] = (
        rst_count
        if rst_count is not None
        else 0.0
    )

    indicators["ack_activity_change"] = (
        ack_count
        if ack_count is not None
        else 0.0
    )

    indicators["forward_packet_change"] = (
        forward_packets
        if forward_packets is not None
        else 0.0
    )

    indicators["backward_packet_change"] = (
        backward_packets
        if backward_packets is not None
        else 0.0
    )

    indicators["directional_ratio_change"] = (
        down_up_ratio
        if down_up_ratio is not None
        else 0.0
    )

    return indicators


def infer_attack_stage(
    transition_dataframe,
    minimum_score=2
):

    indicators = build_behavioral_indicators(
        transition_dataframe
    )

    scores = {
        "Reconnaissance": 0,
        "Discovery": 0,
        "Impact": 0
    }

    evidence = {
        "Reconnaissance": [],
        "Discovery": [],
        "Impact": []
    }

    syn_change = indicators[
        "syn_activity_change"
    ]

    flow_change = indicators[
        "flow_volume_change"
    ]

    packet_change = indicators[
        "traffic_volume_change"
    ]

    byte_change = indicators[
        "byte_volume_change"
    ]

    rst_change = indicators[
        "rst_activity_change"
    ]

    directional_change = indicators[
        "directional_ratio_change"
    ]

    if syn_change >= 0.50:

        scores["Reconnaissance"] += 1

        evidence["Reconnaissance"].append(
            "Significant increase in SYN activity"
        )

    if flow_change >= 0.50:

        scores["Reconnaissance"] += 1

        evidence["Reconnaissance"].append(
            "Significant increase in flow volume"
        )

    if packet_change >= 0.50:

        scores["Reconnaissance"] += 1

        evidence["Reconnaissance"].append(
            "Significant increase in packet volume"
        )

    if syn_change >= 0.30 and flow_change >= 0.30:

        scores["Discovery"] += 1

        evidence["Discovery"].append(
            "Increased connection activity "
            "with increased SYN activity"
        )

    if flow_change >= 0.50 and packet_change >= 0.50:

        scores["Impact"] += 1

        evidence["Impact"].append(
            "Large increase in flow and "
            "packet volume"
        )

    if byte_change >= 0.50 and packet_change >= 0.50:

        scores["Impact"] += 1

        evidence["Impact"].append(
            "Large increase in byte and "
            "packet volume"
        )

    if rst_change >= 0.50 and flow_change >= 0.50:

        scores["Impact"] += 1

        evidence["Impact"].append(
            "Large increase in reset activity "
            "and flow volume"
        )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_stage, best_score = ranked[0]

    if best_score < minimum_score:

        return {
            "stage": "Unknown",
            "score": best_score,
            "confidence": 0.0,
            "evidence": [],
            "indicators": indicators
        }

    confidence = min(
        best_score / 3.0,
        1.0
    )

    return {
        "stage": best_stage,
        "score": best_score,
        "confidence": confidence,
        "evidence": evidence[best_stage],
        "indicators": indicators
    }


def map_stage_to_mitre(stage_result):

    stage = stage_result["stage"]

    mapping = {

        "Reconnaissance": {
            "tactic_id": "TA0043",
            "tactic": "Reconnaissance",
            "techniques": [
                {
                    "id": "T1595",
                    "name": "Active Scanning"
                }
            ]
        },

        "Discovery": {
            "tactic_id": "TA0007",
            "tactic": "Discovery",
            "techniques": [
                {
                    "id": "T1046",
                    "name": "Network Service Discovery"
                }
            ]
        },

        "Impact": {
            "tactic_id": "TA0040",
            "tactic": "Impact",
            "techniques": [
                {
                    "id": "T1498",
                    "name": "Network Denial of Service"
                }
            ]
        },

        "Unknown": {
            "tactic_id": None,
            "tactic": None,
            "techniques": []
        }
    }

    result = mapping.get(
        stage,
        mapping["Unknown"]
    ).copy()

    result["stage"] = stage

    result["confidence"] = (
        stage_result["confidence"]
    )

    result["evidence"] = (
        stage_result["evidence"]
    )

    return result


def analyze_transition(
    transition_dataframe
):

    stage_result = infer_attack_stage(
        transition_dataframe
    )

    mitre_result = map_stage_to_mitre(
        stage_result
    )

    return {
        "stage_analysis": stage_result,
        "mitre_mapping": mitre_result
    }

