#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct Detection {
    cv::Rect box;
    float confidence = 0.0f;
    int class_id = 0;
};

struct TrackResult {
    cv::Rect box;
    int id = -1;
    float confidence = 0.0f;
    int missed = 0;
    int hits = 0;
    bool matched_this_frame = false;
};

class Tracker {
public:
    Tracker(float iou_threshold = 0.25f,
            double center_distance_threshold = 45.0,
            int max_missed_frames = 12,
            int min_hits = 1);

    // ByteTrack-inspired lightweight tracker:
    // - accepts low-confidence detections from YOLO
    // - keeps IDs alive for a few missed frames
    // - matches detections to existing tracks by IoU first, then center distance
    std::vector<TrackResult> update(const std::vector<Detection>& detections);

private:
    struct Track {
        cv::Rect box;
        cv::Point center;
        int id = -1;
        int missed = 0;
        int hits = 0;
        float confidence = 0.0f;
        bool matched_this_frame = false;
    };

    std::vector<Track> tracks;
    int next_id;
    float iou_threshold;
    double center_distance_threshold;
    int max_missed_frames;
    int min_hits;

    static cv::Point centerOf(const cv::Rect& r);
    static float iou(const cv::Rect& a, const cv::Rect& b);
};
