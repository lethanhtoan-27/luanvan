#include "FrameWorker.h"
#include <QThread>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
std::filesystem::path pathOf(const QString& path) {
#ifdef _WIN32
    return std::filesystem::path(path.toStdWString());
#else
    return std::filesystem::path(path.toStdString());
#endif
}

cv::Mat openImage(const QString& path) {
    std::ifstream stream(pathOf(path), std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Không thể mở ảnh");
    const std::streamoff size = stream.tellg();
    if (size <= 0) throw std::runtime_error("Ảnh rỗng hoặc không xác định được kích thước");
    if (static_cast<std::uintmax_t>(size) >
        static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()) ||
        size > std::numeric_limits<std::streamsize>::max())
        throw std::runtime_error("Kích thước file ảnh vượt giới hạn bộ nhớ");
    std::vector<uchar> bytes(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    if (!stream.read(reinterpret_cast<char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("Không đọc được toàn bộ dữ liệu ảnh");
    return cv::imdecode(bytes, cv::IMREAD_COLOR);
}

QImage drawDetections(const cv::Mat& bgr,
                      const std::vector<const detectcore::Detection*>& detections,
                      cv::Mat& previewScratch, std::string& labelScratch) {
    constexpr int maxPreviewSide = 1920;
    const cv::Mat* preview = &bgr;
    if (std::max(bgr.cols, bgr.rows) > maxPreviewSide) {
        const double ratio = static_cast<double>(maxPreviewSide) /
                             static_cast<double>(std::max(bgr.cols, bgr.rows));
        const cv::Size previewSize(std::max(1, static_cast<int>(std::lround(bgr.cols * ratio))),
                                   std::max(1, static_cast<int>(std::lround(bgr.rows * ratio))));
        cv::resize(bgr, previewScratch, previewSize, 0.0, 0.0, cv::INTER_AREA);
        preview = &previewScratch;
    }
    QImage image(preview->cols, preview->rows, QImage::Format_RGB888);
    if (image.isNull()) throw std::bad_alloc();
    cv::Mat out(preview->rows, preview->cols, CV_8UC3, image.bits(),
                static_cast<size_t>(image.bytesPerLine()));
    cv::cvtColor(*preview, out, cv::COLOR_BGR2RGB);
    const float sx = static_cast<float>(out.cols) / static_cast<float>(bgr.cols);
    const float sy = static_cast<float>(out.rows) / static_cast<float>(bgr.rows);

    for (const auto* d : detections) {
        const cv::Rect2f box(d->bbox.x * sx, d->bbox.y * sy,
                             d->bbox.width * sx, d->bbox.height * sy);
        const cv::Rect rect = cv::Rect(box) & cv::Rect(0, 0, out.cols, out.rows);
        if (rect.empty()) continue;

        // Phân biệt màu sắc viền theo loại đối tượng (BGR)
        cv::Scalar boxColor(85, 205, 0); // Xanh lá mặc định
        if (d->className == "drone" || d->className == "Drone" || d->className == "uav" || d->className == "UAV") {
            boxColor = cv::Scalar(0, 0, 255); // Đỏ cho Drone
        } else if (d->className == "kite" || d->className == "Kite") {
            boxColor = cv::Scalar(255, 0, 255); // Hồng/Tím cho Diều
        } else if (d->className == "bird" || d->className == "Bird") {
            boxColor = cv::Scalar(0, 165, 255); // Cam cho Chim
        }

        cv::rectangle(out, rect, boxColor, 2);
        char confidenceText[32];
        std::snprintf(confidenceText, sizeof(confidenceText), " %.2f", d->confidence);
        labelScratch.assign(d->className, 0, std::min(d->className.size(), std::size_t{96}));
        labelScratch.append(confidenceText);
        int base = 0;
        const auto size = cv::getTextSize(labelScratch, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        const int tx = std::clamp(rect.x, 0, std::max(0, out.cols - size.width - 7));
        const int ty = std::max(rect.y - 5, size.height + 5);
        cv::rectangle(out, cv::Rect(tx, ty - size.height - 4,
                      std::min(size.width + 7, out.cols - tx), size.height + base + 7),
                      cv::Scalar(37, 49, 66), cv::FILLED);
        cv::putText(out, labelScratch, {tx + 3, ty}, cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    return image;
}
} // namespace

FrameWorker::FrameWorker(Request req, std::shared_ptr<std::atomic_bool> stop,
                         std::shared_ptr<std::atomic_bool> pending)
    : req_(std::move(req)), stop_(std::move(stop)), pending_(std::move(pending)) {}

void FrameWorker::run() {
    try {
        if (stop_->load()) { emit finished(); return; }

        detectcore::DetectCore corePrimary;
        detectcore::DetectCore coreSecondary;

        std::filesystem::path primaryPath = pathOf(req_.modelDirectory);
        corePrimary.loadModel(primaryPath, req_.threads);

        // Tự động nạp mô hình thứ 2 trong thư mục models/
        std::filesystem::path parentDir = primaryPath.parent_path();
        std::filesystem::path dronePath = parentDir / "drone_ncnn_model";
        std::filesystem::path yolo26Path = parentDir / "yolo26n_ncnn_model";

        bool hasSecondaryModel = false;
        std::string primaryFolderName = primaryPath.filename().string();
        if (primaryFolderName == "yolo26n_ncnn_model" && std::filesystem::exists(dronePath)) {
            try {
                coreSecondary.loadModel(dronePath, req_.threads);
                hasSecondaryModel = true;
            } catch (...) {}
        } else if (primaryFolderName == "drone_ncnn_model" && std::filesystem::exists(yolo26Path)) {
            try {
                coreSecondary.loadModel(yolo26Path, req_.threads);
                hasSecondaryModel = true;
            } catch (...) {}
        }

        if (stop_->load()) { emit finished(); return; }

        // Hàm kiểm tra chỉ lấy đúng 3 đối tượng mong muốn
        auto isTargetClass = [](const std::string& name) {
            return name == "kite"  || name == "Kite"  ||
                   name == "drone" || name == "Drone" ||
                   name == "bird"  || name == "Bird"  ||
                   name == "uav"   || name == "UAV";
        };

        // Lọc danh sách nhãn hiển thị lên GUI
        QStringList classes;
        for (const auto& name : corePrimary.modelInfo().names) {
            if (isTargetClass(name)) {
                QString qName = QString::fromStdString(name);
                if (!classes.contains(qName)) classes.push_back(qName);
            }
        }
        if (hasSecondaryModel) {
            for (const auto& name : coreSecondary.modelInfo().names) {
                if (isTargetClass(name)) {
                    QString qName = QString::fromStdString(name);
                    if (!classes.contains(qName)) classes.push_back(qName);
                }
            }
        }
        emit classesReady(classes, req_.generation);

        quint64 dropped = 0;
        std::vector<detectcore::Detection> combinedDetections;
        std::vector<const detectcore::Detection*> selected;
        cv::Mat previewScratch;
        std::string labelScratch;
        labelScratch.reserve(128);

        // Thuật toán Cross-Model NMS loại bỏ khung trùng giữa 2 mô hình
        auto applyCrossModelNMS = [](std::vector<detectcore::Detection>& detections, float iouThreshold) {
            if (detections.empty()) return;
            std::sort(detections.begin(), detections.end(), [](const detectcore::Detection& a, const detectcore::Detection& b) {
                return a.confidence > b.confidence;
            });

            std::vector<detectcore::Detection> result;
            for (const auto& det : detections) {
                bool keep = true;
                for (const auto& kept : result) {
                    float x1 = std::max(det.bbox.x, kept.bbox.x);
                    float y1 = std::max(det.bbox.y, kept.bbox.y);
                    float x2 = std::min(det.bbox.x + det.bbox.width, kept.bbox.x + kept.bbox.width);
                    float y2 = std::min(det.bbox.y + det.bbox.height, kept.bbox.y + kept.bbox.height);

                    float interW = std::max(0.0f, x2 - x1);
                    float interH = std::max(0.0f, y2 - y1);
                    float interArea = interW * interH;

                    float areaA = det.bbox.width * det.bbox.height;
                    float areaB = kept.bbox.width * kept.bbox.height;
                    float unionArea = areaA + areaB - interArea;

                    float iou = (unionArea > 0.0f) ? (interArea / unionArea) : 0.0f;
                    if (iou > iouThreshold) {
                        keep = false;
                        break;
                    }
                }
                if (keep) result.push_back(det);
            }
            detections = std::move(result);
        };

        const auto process = [&](const cv::Mat& frame, double fps) {
            if (stop_->load()) return false;
            const auto begin = Clock::now();
            const detectcore::Options options{req_.settings->confidence.load(),
                                               req_.settings->iou.load(), req_.threads};

            combinedDetections.clear();

            // 1. Chạy Model 1 -> Lọc Diều, Drone, Chim
            auto resultPrimary = corePrimary.detect(frame, options);
            for (auto& d : resultPrimary.detections) {
                if (isTargetClass(d.className)) {
                    combinedDetections.push_back(std::move(d));
                }
            }

            // 2. Chạy Model 2 -> Lọc Diều, Drone, Chim
            if (hasSecondaryModel) {
                auto resultSecondary = coreSecondary.detect(frame, options);
                for (auto& d : resultSecondary.detections) {
                    if (isTargetClass(d.className)) {
                        combinedDetections.push_back(std::move(d));
                    }
                }
            }

            // 3. Thực thi NMS khử trùng lặp khung trùng giữa 2 mô hình
            applyCrossModelNMS(combinedDetections, options.iou);

            if (stop_->load()) return false;
            if (pending_->exchange(true)) { ++dropped; return false; }

            selected.clear();
            selected.reserve(combinedDetections.size());
            for (const auto& d : combinedDetections) {
                selected.push_back(&d);
            }

            const int limit = req_.settings->maxDetections.load();
            if (static_cast<int>(selected.size()) > limit) {
                std::partial_sort(selected.begin(), selected.begin() + limit, selected.end(),
                    [](const auto* a, const auto* b) { return a->confidence > b->confidence; });
                selected.resize(limit);
            }

            auto image = drawDetections(frame, selected, previewScratch, labelScratch);
            if (stop_->load()) { pending_->store(false); return false; }
            const double total = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
            FrameMetrics metrics{resultPrimary.timing.preprocessMs, resultPrimary.timing.inferenceMs,
                                 resultPrimary.timing.postprocessMs, total, fps, dropped,
                                 static_cast<int>(selected.size())};
            emit frameReady(std::move(image), metrics, req_.generation);
            return true;
        };

        if (req_.source == Source::Image) {
            auto image = openImage(req_.path);
            if (image.empty()) throw std::runtime_error("Không giải mã được ảnh");
            process(image, 0.0);
        } else {
            cv::VideoCapture cap;
            if (req_.source == Source::Camera) {
#ifdef _WIN32
                if (req_.cameraIndex < 0 || req_.cameraId.isEmpty() ||
                    !cap.open(req_.cameraIndex, cv::CAP_DSHOW))
                    throw std::runtime_error("Không thể mở camera bằng DirectShow");
#elif defined(__linux__)
                if (!req_.cameraId.startsWith("/dev/video") ||
                    !cap.open(req_.cameraId.toStdString(), cv::CAP_V4L2))
                    throw std::runtime_error("Không thể mở thiết bị camera V4L2");
#else
                throw std::runtime_error("Chưa hỗ trợ liệt kê camera trên hệ điều hành này");
#endif
            } else {
                cap.open(req_.path.toUtf8().constData());
            }
            if (!cap.isOpened()) throw std::runtime_error("Không thể mở camera/video");
            cv::Mat frame;
            int failures = 0;
            auto last = Clock::now();
            if (req_.source == Source::Video) {
                const double rawFps = cap.get(cv::CAP_PROP_FPS);
                const bool fpsValid = std::isfinite(rawFps) && rawFps > 0.1 && rawFps <= 240.0;
                const double fps = fpsValid ? rawFps : 30.0;
                const double rawCount = cap.get(cv::CAP_PROP_FRAME_COUNT);
                const bool countValid = std::isfinite(rawCount) && rawCount >= 1.0 &&
                                        rawCount <= 1.0e10;
                const bool seekable = fpsValid && countValid;
                const std::int64_t durationMs = seekable
                    ? static_cast<std::int64_t>(std::ceil(rawCount * 1000.0 / fps)) : 0;
                emit videoOpened(durationMs, seekable, req_.generation);
                if (!req_.playback) throw std::logic_error("Missing video playback controller");
                if (req_.initialVideoPositionMs > 0 && seekable)
                    req_.playback->seek(std::min(req_.initialVideoPositionMs, durationMs));

                const auto framePeriod = std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(1.0 / fps));
                auto nextFrame = Clock::now();
                std::int64_t shownMs = 0;
                while (!stop_->load()) {
                    const auto cmd = req_.playback->next(*stop_, nextFrame);
                    if (cmd.stop) break;
                    if (cmd.seek && seekable) {
                        const auto frameNumber = std::clamp(
                            std::floor(static_cast<long double>(cmd.seekMs) * fps / 1000.0L),
                            0.0L, static_cast<long double>(rawCount - 1.0));
                        if (!cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(frameNumber))) {
                            emit playbackNotice("Video backend does not support seeking", req_.generation);
                            emit videoOpened(durationMs, false, req_.generation);
                        } else {
                            shownMs = static_cast<std::int64_t>(frameNumber * 1000.0L / fps);
                        }
                    }
                    if (cmd.seek) {
                        while (pending_->load() && !stop_->load()) QThread::msleep(5);
                    }
                    if (stop_->load()) break;
                    if (!cap.read(frame) || frame.empty()) {
                        if (stop_->load()) break;
                        emit videoPosition(durationMs > 0 ? durationMs : shownMs, req_.generation);
                        break;
                    }
                    const double actualFrame = cap.get(cv::CAP_PROP_POS_FRAMES);
                    if (fpsValid && std::isfinite(actualFrame) && actualFrame >= 1.0) {
                        shownMs = static_cast<std::int64_t>((actualFrame - 1.0) * 1000.0 / fps);
                    } else if (!cmd.seek) {
                        shownMs += static_cast<std::int64_t>(std::llround(1000.0 / fps));
                    }
                    if (durationMs > 0) shownMs = std::clamp<std::int64_t>(shownMs, 0, durationMs);
                    const auto now = Clock::now();
                    const double seconds = std::chrono::duration<double>(now - last).count();
                    last = now;
                    if (process(frame, seconds > 0.0 ? 1.0 / seconds : 0.0))
                        emit videoPosition(shownMs, req_.generation);
                    nextFrame = Clock::now() + framePeriod;
                }
            } else {
                while (!stop_->load()) {
                    if (!cap.read(frame) || frame.empty()) {
                        if (++failures >= 10)
                            throw std::runtime_error("Camera ngắt kết nối hoặc không có frame");
                        QThread::msleep(30);
                        continue;
                    }
                    failures = 0;
                    const auto now = Clock::now();
                    const double seconds = std::chrono::duration<double>(now - last).count();
                    last = now;
                    process(frame, seconds > 0.0 ? 1.0 / seconds : 0.0);
                }
            }
            cap.release();
        }
    } catch (const std::exception& e) {
        pending_->store(false);
        if (!stop_->load()) emit error(QString::fromUtf8(e.what()), req_.generation);
    }
    emit finished();
}