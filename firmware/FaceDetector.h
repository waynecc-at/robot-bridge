#pragma once
/**
 * Face detection + crop using ESP-DL on ESP32-S3.
 *
 * Detects faces in the camera frame, crops the face region,
 * and encodes it as a small JPEG for upload.
 *
 * ESP-DL must be added to platformio.ini:
 *   lib_deps = espressif/esp-dl
 */

#include <vector>
#include <esp_camera.h>
#include <dl_image.hpp>
#include <face_detect.hpp>

class FaceDetector {
public:
    using FaceCallback = void (*)(const uint8_t *jpeg, size_t len,
                                   int x, int y, int w, int h);

    FaceDetector() : _cb(nullptr), _detector(nullptr), _skip(0) {}

    bool begin() {
        _detector = new dl::detect::FaceDetector();
        if (!_detector) {
            log_e("FaceDetector: allocation failed");
            return false;
        }
        log_i("FaceDetector: ready (MTMN model, ~3-5 FPS)");
        return true;
    }

    void setFaceCallback(FaceCallback cb) { _cb = cb; }

    /**
     * Process a camera frame buffer. Detects faces and sends crops via callback.
     * Call this from CameraDriver::FrameCallback.
     * Skips every N frames for performance (runs at ~3-5 FPS out of ~15 raw).
     */
    void process(const uint8_t *jpeg, size_t len, int width, int height) {
        if (!_detector || !_cb) return;

        // Skip frames to throttle to ~3-5 FPS
        _skip++;
        if (_skip < 4) return;
        _skip = 0;

        // Decode JPEG to RGB888 for ESP-DL
        dl::image::ImageRGB *img = dl::image::from_jpeg(jpeg, len);
        if (!img) return;

        auto faces = _detector->detect(img);
        if (faces.empty()) {
            delete img;
            return;
        }

        for (auto &face : faces) {
            int fx = max(0, face.box[0]);
            int fy = max(0, face.box[1]);
            int fw = min(face.box[2], width - fx);
            int fh = min(face.box[3], height - fy);

            if (fw < 40 || fh < 40) continue;  // too small

            // Crop face region from RGB888 image
            uint8_t *cropStart = img->data + (fy * img->w * 3) + (fx * 3);
            size_t cropLen = fw * fh * 3;

            if (cropLen < 1024) { delete img; continue; }

            // Encode crop as JPEG
            uint8_t *jpegBuf = nullptr;
            size_t jpegLen = 0;
            bool ok = fmt2jpg(cropStart, cropLen, fw, fh,
                              PIXFORMAT_RGB888, 15, &jpegBuf, &jpegLen);

            if (ok && jpegBuf && jpegLen > 0) {
                _cb(jpegBuf, jpegLen, fx, fy, fw, fh);
                free(jpegBuf);
            }
        }
        delete img;
    }

private:
    FaceCallback _cb;
    dl::detect::FaceDetector *_detector;
    int _skip;  // frame skip counter
};
