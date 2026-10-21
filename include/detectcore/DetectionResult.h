#pragma once
#include <vector>
#include <string>
#include <opencv2/core.hpp>

namespace detectcore {

struct Detection {
    int classId = -1;
    std::string className;
    float confidence = 0.f;
    cv::Rect2f bbox;
};

// --- ĐỊNH NGHĨA OPTIONS ĐẦY ĐỦ TẠI ĐÂY ---
struct Options {
    float confidence = 0.25f;
    float iou = 0.45f;
    int threads = 4;
    std::vector<std::string> allowedClasses = {};
};

struct DetectionTiming {
    double preprocessMs = 0.0;
    double inferenceMs = 0.0;
    double postprocessMs = 0.0;
};

struct DetectionResult {
    std::vector<Detection> detections;
    DetectionTiming timing;
};

struct BirdKiteEvidence {
    cv::Rect2f bbox;
    int predictedClassId = -1;
    int candidateIndex = -1;
    float birdScore = -1.f;
    float kiteScore = -1.f;
};

} // namespace detectcore