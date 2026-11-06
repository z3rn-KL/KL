/** Ordered exactly as CandidateEncoder.FEATURE_NAMES in python/rl/proposed/candidate_encoder.py.
 * The Python trace endpoint currently maps the SAME numerical vector onto different
 * feature_dict names. Use features[] instead; never infer semantics from the bad keys.
 */
export const RL_ENCODER_FEATURES = [
  { key: "elapsed_ratio", label: "Thời gian ca đã trôi qua" },
  { key: "remaining_ratio", label: "Tỷ lệ đơn còn lại" },
  { key: "distance_ratio", label: "Cự ly chuẩn hóa" },
  { key: "travel_time_ratio", label: "Thời gian đi chuẩn hóa" },
  { key: "slack_ratio", label: "Đệm hạn giao chuẩn hóa" },
  { key: "deadline_valid", label: "Có hạn giao (0/1)" },
  { key: "lateness_ratio", label: "Trễ hạn chuẩn hóa" },
  { key: "cluster_switch", label: "Đổi cụm (0/1)" },
] as const;

export function formattedRlFeatures(features: number[] | undefined) {
  return RL_ENCODER_FEATURES.map(({ key, label }, index) => ({
    key,
    label,
    value: typeof features?.[index] === "number" && Number.isFinite(features[index])
      ? features[index].toFixed(3) : "—",
  }));
}
