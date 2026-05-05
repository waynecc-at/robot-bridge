#pragma once
/**
 * vision_frame WebSocket message builder.
 *
 * Encodes JPEG data as base64 and sends it through the existing
 * WebSocket connection as a JSON message.
 *
 * Integrates with CameraDriver, FaceDetector, and FaceTracker.
 */

#include <Arduino.h>
#include <ArduinoJson.h>
#include <base64.h>  // ESP32 built-in

class WebSocketSender {
public:
    using SendFunction = void (*)(const String &message);

    WebSocketSender() : _send(nullptr) {}

    void setSendFunction(SendFunction fn) { _send = fn; }

    void sendFrame(const uint8_t *jpeg, size_t len,
                   int width, int height) {
        if (!_send || len == 0) return;

        String b64 = base64::encode(jpeg, len);
        if (b64.length() == 0) return;

        StaticJsonDocument<4096> doc;
        doc["type"]   = "vision_frame";
        doc["data"]   = b64;
        doc["format"] = "jpeg";
        doc["width"]  = width;
        doc["height"] = height;

        String output;
        serializeJson(doc, output);
        _send(output);
    }

    void sendFaceCrop(const uint8_t *jpeg, size_t len,
                      int x, int y, int w, int h) {
        if (!_send || len == 0) return;

        String b64 = base64::encode(jpeg, len);
        if (b64.length() == 0) return;

        StaticJsonDocument<2048> doc;
        doc["type"]   = "vision_frame";
        doc["data"]   = b64;
        doc["format"] = "jpeg";
        doc["face"]   = true;
        doc["bx"]     = x;
        doc["by"]     = y;
        doc["bw"]     = w;
        doc["bh"]     = h;

        String output;
        serializeJson(doc, output);
        _send(output);
    }

    void sendPersonDetected(const char *name, const char *relationship) {
        if (!_send) return;

        StaticJsonDocument<256> doc;
        doc["type"]         = "person_detected";
        doc["name"]         = name;
        doc["relationship"] = relationship;

        String output;
        serializeJson(doc, output);
        _send(output);
    }

private:
    SendFunction _send;
};
