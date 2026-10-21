#pragma once
#include <detectcore/IInferenceEngine.h>
#include "Preprocessor.h"
#include <vector>
#include <string>

namespace detectcore {

// 1. Hàm tương thích ngược (dùng cho tests và legacy code)
std::vector<Detection> postprocessRaw(const Tensor& raw, const LetterboxResult& prep,
                                      const ModelInfo& model, float confidence, float iou,
                                      std::vector<BirdKiteEvidence>* evidence = nullptr);

// 2. Hàm mở rộng hỗ trợ lọc nhãn qua allowedClasses
std::vector<Detection> postprocessRaw(const Tensor& raw, const LetterboxResult& prep,
                                      const ModelInfo& model, float confidence, float iou,
                                      const std::vector<std::string>& allowedClasses,
                                      std::vector<BirdKiteEvidence>* evidence = nullptr);

} // namespace detectcore