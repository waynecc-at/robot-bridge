# ESP32-S3 摄像头固件参考

StackChan (CoreS3) 摄像头集成代码。添加 `Faces` 目录下到您的 PlatformIO 项目。

## 依赖

```ini
# platformio.ini
lib_deps =
    espressif/esp-dl        # 深度学习 (人脸检测)
    esphome/esp-who         # 摄像头 + 人脸检测示例
```

## 摄像头初始化 (camera.h)

```cpp
#pragma once
#include <esp_camera.h>

// CoreS3 GC0308 摄像头引脚
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM      4
#define SIOC_GPIO_NUM      5
#define Y9_GPIO_NUM        3
#define Y8_GPIO_NUM       16
#define Y7_GPIO_NUM       11
#define Y6_GPIO_NUM       10
#define Y5_GPIO_NUM        9
#define Y4_GPIO_NUM        8
#define Y3_GPIO_NUM        7
#define Y2_GPIO_NUM       20
#define VSYNC_GPIO_NUM     6
#define HREF_GPIO_NUM     18
#define PCLK_GPIO_NUM     17

static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,
    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,   // 320x240
    .jpeg_quality = 12,
    .fb_count = 1,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

esp_err_t init_camera() {
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE("CAM", "Camera init failed: %x", err);
    }
    return err;
}
```

## 定时拍照 + WebSocket 发送

在现有 WebSocket 循环中添加定时拍照逻辑（约 1-2fps，不阻塞主循环）：

```cpp
// 现有 WebSocket 客户端变量: WebSocketClient ws

static uint32_t last_capture = 0;

void loop() {
    uint32_t now = millis();

    // 每隔 1 秒拍一帧
    if (now - last_capture > 1000) {
        last_capture = now;

        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) {
            // 发送 JSON 格式  视觉帧
            size_t b64_len = 4 * ((fb->len + 2) / 3);
            char *b64_buf = (char *)malloc(b64_len + 1);
            if (b64_buf) {
                base64_encode(fb->buf, fb->len, b64_buf);
                // 构造 JSON 消息
                DynamicJsonDocument doc(4096);
                doc["type"] = "vision_frame";
                doc["data"] = b64_buf;
                doc["format"] = "jpeg";
                doc["width"] = fb->width;
                doc["height"] = fb->height;

                String payload;
                serializeJson(doc, payload);
                ws.sendTXT(payload);

                free(b64_buf);
            }
            esp_camera_fb_return(fb);
        }
    }
}
```

## 运动检测触发 (可选, 省带宽)

Espressif `motion_detect` 组件，在画面变化时才拍照发送：

```cpp
#include "esp_camera.h"
#include "dl_image.hpp"

static uint8_t *prev_frame = NULL;
static size_t prev_frame_len = 0;

bool motion_detected(camera_fb_t *fb) {
    if (!prev_frame) {
        prev_frame = (uint8_t *)malloc(fb->len);
        memcpy(prev_frame, fb->buf, fb->len);
        prev_frame_len = fb->len;
        return true;  // first frame = motion
    }

    // 简单差分检测
    int diff = 0;
    int step = fb->width * 2;  // stride
    for (int i = 0; i < fb->len; i += step) {
        if (abs((int)fb->buf[i] - (int)prev_frame[i]) > 30) diff++;
    }

    memcpy(prev_frame, fb->buf, fb->len);
    prev_frame_len = fb->len;

    return diff > 50;  // >50 different pixels → motion
}

// 使用:
// if (motion_detected(fb)) { 编码 + 发送 }
```

## 人脸裁切 + 发送 (ESP-WHO)

用 ESP-WHO 在 ESP32 上做人脸检测，只发送裁切区域而非整帧：

```cpp
#include "esp_face_detect.hpp"

static dl::face::FaceDetector *detector = nullptr;

void init_face_detection() {
    detector = new dl::face::FaceDetector();
}

void process_frame_with_ai(camera_fb_t *fb) {
    if (!detector) return;

    // 转换为 RGB888 (ESP-WHO 需要的格式)
    dl::image::ImageRGB *img = dl::image::from_jpeg(fb->buf, fb->len);
    if (!img) return;

    // 检测人脸
    std::list<dl::result::FaceResult> faces = detector->detect(img);
    if (faces.empty()) {
        delete img;
        return;
    }

    // 对每个检测到的人脸
    for (auto &face : faces) {
        int x = face.box[0], y = face.box[1];
        int w = face.box[2], h = face.box[3];

        // 裁切人脸区域 → JPEG → base64 → 发送
        camera_fb_t crop;
        crop.buf = img->data + y * img->w * 3 + x * 3;
        crop.width = w;
        crop.height = h;
        crop.format = PIXFORMAT_RGB888;
        crop.len = w * h * 3;

        size_t jpeg_len = 0;
        uint8_t *jpeg_buf = NULL;
        esp_camera_fb_return(fb);  // 释放原帧
        bool ok = fmt2jpg(crop.buf, crop.len, crop.width, crop.height,
                          PIXFORMAT_RGB888, 20, &jpeg_buf, &jpeg_len);
        if (ok && jpeg_len > 0) {
            send_vision_frame(jpeg_buf, jpeg_len);
            free(jpeg_buf);
        }
    }
    delete img;
}
```

## WebSocket 消息格式

栈机器人发送:
```json
{
  "type": "vision_frame",
  "data": "<base64_jpeg>",
  "format": "jpeg",
  "width": 320,
  "height": 240
}
```

服务器回复 (当识别到人物时):
```json
{
  "type": "person_detected",
  "name": "小明",
  "relationship": "主人"
}
```

## 舵机跟随 (人脸追踪)

人脸检测到后, 计算中心偏移 → PID → 舵机:

```cpp
// 在 WebSocket JSON 回调中或 ESP-WHO 检测后执行
void track_face(int face_x, int face_y, int face_w, int face_h,
                int frame_w, int frame_h) {

    // 人脸中心
    int face_cx = face_x + face_w / 2;
    int face_cy = face_y + face_h / 2;
    int frame_cx = frame_w / 2;
    int frame_cy = frame_h / 2;

    // 偏移量 (归一化到 -1..1)
    float dx = (float)(face_cx - frame_cx) / frame_cx;
    float dy = (float)(face_cy - frame_cy) / frame_cy;

    // P控制器
    float kp = 30.0;  // 度/偏移
    float pan_angle = current_pan + dx * kp;
    float tilt_angle = current_tilt + dy * kp;

    // 限幅
    pan_angle = constrain(pan_angle, 0, 180);
    tilt_angle = constrain(tilt_angle, 0, 180);

    // 驱动舵机
    servo_pan.write(pan_angle);
    servo_tilt.write(tilt_angle);

    current_pan = pan_angle;
    current_tilt = tilt_angle;
}
```

## 调试提示

1. **帧率平衡**: 1fps 足够人脸检测, 降低 CPU 占用
2. **弱光**: GC0308 在 <50 lux 时质量下降明显, 可加补光灯
3. **PSRAM**: `fb_location = CAMERA_FB_IN_PSRAM` 避免碎片化
4. **JPEG 质量**: `jpeg_quality = 12` (0-63, 越低越好) 平衡大小和画质
