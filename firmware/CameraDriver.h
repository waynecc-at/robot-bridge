#pragma once
/**
 * Camera driver for M5Stack CoreS3 GC0308.
 *
 * Initializes the camera, captures JPEG frames, and provides
 * motion detection + frame upload via WebSocket.
 *
 * Usage:
 *   CameraDriver cam;
 *   cam.begin();
 *   // In loop():
 *   cam.loop();  // auto-captures every 1s, sends if motion
 *   cam.onFrame([](const uint8_t *jpeg, size_t len) {
 *       ws.sendFrame(jpeg, len);
 *   });
 */

#include <esp_camera.h>
#include <Arduino.h>

// CoreS3 GC0308 pin mapping
#define CAM_PWDN_GPIO    -1
#define CAM_RESET_GPIO   -1
#define CAM_XCLK_GPIO    15
#define CAM_SIOD_GPIO     4
#define CAM_SIOC_GPIO     5
#define CAM_Y9_GPIO       3
#define CAM_Y8_GPIO      16
#define CAM_Y7_GPIO      11
#define CAM_Y6_GPIO      10
#define CAM_Y5_GPIO       9
#define CAM_Y4_GPIO       8
#define CAM_Y3_GPIO       7
#define CAM_Y2_GPIO      20
#define CAM_VSYNC_GPIO    6
#define CAM_HREF_GPIO    18
#define CAM_PCLK_GPIO    17

class CameraDriver {
public:
    using FrameCallback = void (*)(const uint8_t *jpeg, size_t len,
                                    int width, int height);
    using MotionCallback = void (*)();

    CameraDriver()
        : _cb(nullptr), _motionCb(nullptr),
          _lastCapture(0), _interval(1000),
          _prevFrame(nullptr), _prevLen(0) {}

    bool begin(framesize_t size = FRAMESIZE_QVGA,
               pixformat_t fmt = PIXFORMAT_JPEG,
               int quality = 12) {
        camera_config_t cfg = {};
        cfg.pin_pwdn    = CAM_PWDN_GPIO;
        cfg.pin_reset   = CAM_RESET_GPIO;
        cfg.pin_xclk    = CAM_XCLK_GPIO;
        cfg.pin_sscb_sda = CAM_SIOD_GPIO;
        cfg.pin_sscb_scl = CAM_SIOC_GPIO;
        cfg.pin_d7      = CAM_Y9_GPIO;
        cfg.pin_d6      = CAM_Y8_GPIO;
        cfg.pin_d5      = CAM_Y7_GPIO;
        cfg.pin_d4      = CAM_Y6_GPIO;
        cfg.pin_d3      = CAM_Y5_GPIO;
        cfg.pin_d2      = CAM_Y4_GPIO;
        cfg.pin_d1      = CAM_Y3_GPIO;
        cfg.pin_d0      = CAM_Y2_GPIO;
        cfg.pin_vsync   = CAM_VSYNC_GPIO;
        cfg.pin_href    = CAM_HREF_GPIO;
        cfg.pin_pclk    = CAM_PCLK_GPIO;

        cfg.xclk_freq_hz = 20000000;
        cfg.ledc_timer   = LEDC_TIMER_0;
        cfg.ledc_channel = LEDC_CHANNEL_0;

        cfg.pixel_format = fmt;
        cfg.frame_size   = size;
        cfg.jpeg_quality = quality;
        cfg.fb_count     = 1;
        cfg.fb_location  = CAMERA_FB_IN_PSRAM;
        cfg.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

        esp_err_t err = esp_camera_init(&cfg);
        if (err != ESP_OK) {
            log_e("Camera init failed: 0x%x", err);
            return false;
        }

        // Fix for GC0308 auto-exposure
        sensor_t *s = esp_camera_sensor_get();
        if (s) {
            s->set_hmirror(s, 1);  // mirror for front-facing
            s->set_vflip(s, 1);
        }

        log_i("Camera OK: %dx%d %s",
              cfg.frame_size, cfg.frame_size, "QVGA");
        return true;
    }

    void setInterval(unsigned long ms) { _interval = ms; }
    void setFrameCallback(FrameCallback cb) { _cb = cb; }
    void setMotionCallback(MotionCallback cb) { _motionCb = cb; }

    void loop() {
        unsigned long now = millis();
        if (now - _lastCapture < _interval) return;
        _lastCapture = now;

        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            log_w("Camera capture failed");
            return;
        }

        // Motion detection (skip sending if no motion)
        bool hasMotion = true;
        if (_prevFrame) {
            hasMotion = _detectMotion(fb->buf, fb->len);
        } else {
            _savePrevFrame(fb);
        }

        if (hasMotion && _cb) {
            _cb(fb->buf, fb->len, fb->width, fb->height);
        }

        if (hasMotion && _motionCb) {
            _motionCb();
        }

        if (hasMotion) {
            _savePrevFrame(fb);
        }

        esp_camera_fb_return(fb);
    }

private:
    FrameCallback  _cb;
    MotionCallback _motionCb;
    unsigned long  _lastCapture;
    unsigned long  _interval;
    uint8_t       *_prevFrame;
    size_t         _prevLen;

    bool _detectMotion(const uint8_t *frame, size_t len) {
        if (!_prevFrame || _prevLen == 0) return true;

        // Simple pixel-difference motion detection at 8:1 stride
        int diffCount = 0;
        size_t step = 64;  // check every 64th byte
        for (size_t i = 0; i < min(len, _prevLen); i += step) {
            if (abs((int)frame[i] - (int)_prevFrame[i]) > 25) {
                diffCount++;
            }
        }
        // ~2% of sampled pixels changed → motion
        return diffCount > (len / step / 50);
    }

    void _savePrevFrame(const camera_fb_t *fb) {
        free(_prevFrame);
        _prevFrame = (uint8_t *)malloc(fb->len);
        if (_prevFrame) {
            memcpy(_prevFrame, fb->buf, fb->len);
            _prevLen = fb->len;
        }
    }
};
