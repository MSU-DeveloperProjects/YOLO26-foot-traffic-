#include "tracker.h"
#include <algorithm>
#include <cmath>
#include <limits>

Tracker::Tracker(float iou_threshold,
                 double center_distance_threshold,
                 int max_missed_frames,
                 int min_hits)
    : next_id(0),
      iou_threshold(iou_threshold),
      center_distance_threshold(center_distance_threshold),
      max_missed_frames(max_missed_frames),
      min_hits(min_hits) {}

cv::Point Tracker::centerOf(const cv::Rect& r) {
    return cv::Point(r.x + r.width / 2, r.y + r.height / 2);
}

float Tracker::iou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    if (inter.area() <= 0) return 0.0f;

    float union_area = static_cast<float>(a.area() + b.area() - inter.area());
    if (union_area <= 0.0f) return 0.0f;

    return static_cast<float>(inter.area()) / union_area;
}

std::vector<TrackResult> Tracker::update(const std::vector<Detection>& detections) {
    std::vector<TrackResult> results;

    for (auto& t : tracks) {
        t.matched_this_frame = false;
    }

    // Process stronger detections first. This is not full ByteTrack, but it uses
    // the same useful idea: do not throw away low-confidence detections too early.
    std::vector<int> order(detections.size());
    for (int i = 0; i < static_cast<int>(order.size()); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return detections[a].confidence > detections[b].confidence;
    });

    for (int det_index : order) {
        const Detection& det = detections[det_index];
        cv::Point det_center = centerOf(det.box);

        int best_track = -1;
        float best_iou = 0.0f;
        double best_dist = std::numeric_limits<double>::max();

        for (int i = 0; i < static_cast<int>(tracks.size()); ++i) {
            if (tracks[i].matched_this_frame) continue;

            float current_iou = iou(det.box, tracks[i].box);
            double current_dist = std::hypot(det_center.x - tracks[i].center.x,
                                             det_center.y - tracks[i].center.y);

            bool better_iou = current_iou > best_iou;
            bool similar_iou_better_dist = std::abs(current_iou - best_iou) < 1e-6 && current_dist < best_dist;

            if (better_iou || similar_iou_better_dist) {
                best_iou = current_iou;
                best_dist = current_dist;
                best_track = i;
            }
        }

        bool matched = false;
        if (best_track >= 0) {
            bool good_iou_match = best_iou >= iou_threshold;
            bool good_center_match = best_iou > 0.0f && best_dist <= center_distance_threshold;
            bool near_center_match = best_iou == 0.0f && best_dist <= center_distance_threshold * 0.55;

            if (good_iou_match || good_center_match || near_center_match) {
                Track& t = tracks[best_track];
                t.box = det.box;
                t.center = det_center;
                t.confidence = det.confidence;
                t.missed = 0;
                t.hits++;
                t.matched_this_frame = true;
                matched = true;
            }
        }

        if (!matched) {
            Track t;
            t.box = det.box;
            t.center = det_center;
            t.id = next_id++;
            t.missed = 0;
            t.hits = 1;
            t.confidence = det.confidence;
            t.matched_this_frame = true;
            tracks.push_back(t);
        }
    }

    // Age unmatched tracks but keep them briefly, so one missed YOLO frame does
    // not immediately create a new person ID.
    std::vector<Track> kept;
    kept.reserve(tracks.size());
    for (auto& t : tracks) {
        if (!t.matched_this_frame) {
            t.missed++;
        }

        if (t.missed <= max_missed_frames) {
            kept.push_back(t);
        }
    }
    tracks.swap(kept);

    for (const auto& t : tracks) {
        // Return active/confirmed tracks. Counting code can still decide whether
        // to use missed tracks; for now we only output tracks visible this frame.
        if (t.hits >= min_hits && t.missed == 0) {
            results.push_back({t.box, t.id, t.confidence, t.missed, t.hits, t.matched_this_frame});
        }
    }

    return results;
}
