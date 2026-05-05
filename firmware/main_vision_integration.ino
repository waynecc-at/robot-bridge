/**
 * StackChan firmware — vision module integration example.
 *
 * Drop this into your existing StackChan.ino or merge into setup()/loop().
 *
 * Dependencies (platformio.ini):
 *   lib_deps =
 *     bblanchon/ArduinoJson @ ^6.18
 *     espressif/esp-dl
 *     espressif/esp32-camera
 */

#include "CameraDriver.h"
#include "FaceDetector.h"
#include "FaceTracker.h"
#include "WebSocketSender.h"

// ── Global instances ──────────────────────────────────────────
CameraDriver    camera;
FaceDetector    faceDetector;
FaceTracker     tracker;
WebSocketSender wsSender;

// ── Callbacks ─────────────────────────────────────────────────

// Called every ~1s when motion is detected
void onCameraFrame(const uint8_t *jpeg, size_t len,
                   int width, int height) {

    // Send full frame as vision_frame (iteration 1)
    wsSender.sendFrame(jpeg, len, width, height);

    // Run face detection on the same frame (iteration 2)
    faceDetector.process(jpeg, len, width, height);
}

// Called when a face is detected and cropped (iteration 2)
void onFaceCrop(const uint8_t *jpeg, size_t len,
                int x, int y, int w, int h) {

    // Send face crop to server for recognition
    wsSender.sendFaceCrop(jpeg, len, x, y, w, h);

    // Update face tracker for servo following (iteration 3)
    tracker.onFace(x, y, w, h);
}

// Called by your existing WebSocket onMessage handler
void onWebSocketMessage(const String &message) {
    StaticJsonDocument<512> doc;
    DeserializationError err = deserializeJson(doc, message);
    if (err) return;

    const char *type = doc["type"];

    if (strcmp(type, "person_detected") == 0) {
        // Server recognized a person — update display/behavior
        const char *name = doc["name"] | "unknown";
        const char *rel  = doc["relationship"] | "";
        log_i("Person detected: %s (%s)", name, rel);

        // TODO: show name on LVGL display
        // lv_label_set_text(person_label, name);
    }
}

// ── setup / loop ──────────────────────────────────────────────

void setup_vision() {
    // 1. Camera
    if (!camera.begin()) {
        log_e("Camera init failed — vision disabled");
        return;
    }
    camera.setFrameCallback(onCameraFrame);
    camera.setInterval(1000);  // 1 FPS

    // 2. Face detector
    if (!faceDetector.begin()) {
        log_w("Face detector init failed — continuing without it");
    }
    faceDetector.setFaceCallback(onFaceCrop);

    // 3. Face tracker (servos)
    tracker.attach(/* panPin= */ 12, /* tiltPin= */ 13);
    tracker.setPID(0.3, 0.01, 0.05);
    tracker.setFrameSize(320, 240);

    // 4. WebSocket sender
    wsSender.setSendFunction([](const String &msg) {
        // Replace `ws` with your WebSocket client instance
        // ws.sendTXT(msg);
    });

    log_i("Vision subsystem ready");
}

void loop_vision() {
    camera.loop();   // capture + motion detect
    tracker.loop();  // servo follow
}
