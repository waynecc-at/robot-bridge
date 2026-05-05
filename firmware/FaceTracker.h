#pragma once
/**
 * PID face tracker for StackChan pan/tilt servos.
 *
 * Receives face bounding box from FaceDetector, computes
 * offset from frame center, runs PID to calculate servo angles.
 *
 * Handles:
 *  - Smooth face following with configurable PID gains
 *  - "Search" sweep when face is lost for >3s
 *  - Angle limiting to prevent servo overrotation
 */

#include <Arduino.h>

// Uncomment the servo library you use:
// #include <ESP32Servo.h>
// #include <Servo.h>

class FaceTracker {
public:
    FaceTracker()
        : _servoPan(-1), _servoTilt(-1),
          _panAngle(90), _tiltAngle(90),
          _targetPan(90), _targetTilt(90),
          _kp(0.3), _ki(0.01), _kd(0.05),
          _integralPan(0), _integralTilt(0),
          _lastErrorPan(0), _lastErrorTilt(0),
          _lastFaceTime(0), _faceLost(false),
          _frameW(320), _frameH(240),
          _searchDir(1), _searchTime(0) {}

    /**
     * Attach servo pins. Call once in setup().
     */
    void attach(int pinPan, int pinTilt) {
        _servoPan = pinPan;
        _servoTilt = pinTilt;
        // _pan.attach(pinPan);
        // _tilt.attach(pinTilt);
        log_i("FaceTracker: servos on pan=%d tilt=%d", pinPan, pinTilt);
    }

    void setFrameSize(int w, int h) { _frameW = w; _frameH = h; }

    /**
     * Set PID gains.
     * Defaults: kp=0.3 ki=0.01 kd=0.05 — good for small servos.
     */
    void setPID(float kp, float ki, float kd) {
        _kp = kp; _ki = ki; _kd = kd;
    }

    /**
     * Update face position from detector callback.
     * Call from FaceDetector::FaceCallback.
     */
    void onFace(int x, int y, int w, int h) {
        int faceCx = x + w / 2;
        int faceCy = y + h / 2;
        int frameCx = _frameW / 2;
        int frameCy = _frameH / 2;

        // Normalize error to [-1, 1]
        float errPan  = (float)(faceCx - frameCx) / frameCx;
        float errTilt = (float)(faceCy - frameCy) / frameCy;

        _updatePID(errPan, errTilt);
        _lastFaceTime = millis();
        _faceLost = false;
    }

    /**
     * Call every loop() iteration.
     * Computes servo angles and applies them.
     */
    void loop() {
        unsigned long now = millis();

        if (_faceLost) {
            // Search mode: sweep slowly
            if (now - _searchTime > 50) {
                _targetPan += 0.5 * _searchDir;
                if (_targetPan > 150) { _targetPan = 150; _searchDir = -1; }
                if (_targetPan < 30)  { _targetPan = 30;  _searchDir = 1;  }
                _searchTime = now;
            }
        } else {
            // Face tracking: check if lost
            if (now - _lastFaceTime > 3000) {
                _faceLost = true;
                _searchTime = now;
                log_i("FaceTracker: face lost, entering search");
            }
        }

        _applyAngles();
    }

    bool isFaceLost() const { return _faceLost; }

private:
    int _servoPan, _servoTilt;
    float _panAngle, _tiltAngle;
    float _targetPan, _targetTilt;

    // PID
    float _kp, _ki, _kd;
    float _integralPan, _integralTilt;
    float _lastErrorPan, _lastErrorTilt;

    unsigned long _lastFaceTime;
    bool _faceLost;
    int _frameW, _frameH;

    // Search state
    int _searchDir;
    unsigned long _searchTime;

    // Servo objects (uncomment your library)
    // Servo _pan;
    // Servo _tilt;

    void _updatePID(float errPan, float errTilt) {
        float dt = 0.05;  // ~50ms per PID update (fixed step)

        // Pan
        _integralPan = constrain(_integralPan + errPan * dt, -5, 5);
        float derivPan = (errPan - _lastErrorPan) / dt;
        float outputPan = _kp * errPan + _ki * _integralPan + _kd * derivPan;
        _lastErrorPan = errPan;

        // Tilt
        _integralTilt = constrain(_integralTilt + errTilt * dt, -5, 5);
        float derivTilt = (errTilt - _lastErrorTilt) / dt;
        float outputTilt = _kp * errTilt + _ki * _integralTilt + _kd * derivTilt;
        _lastErrorTilt = errTilt;

        // Convert normalized output to angle delta
        const float maxDelta = 30.0;  // max degrees per update
        _targetPan  = constrain(_targetPan  + outputPan  * maxDelta, 0, 180);
        _targetTilt = constrain(_targetTilt + outputTilt * maxDelta, 0, 180);
    }

    void _applyAngles() {
        // Smooth interpolation (ease-in)
        _panAngle  += (_targetPan  - _panAngle)  * 0.3;
        _tiltAngle += (_targetTilt - _tiltAngle) * 0.3;

        // _pan.write((int)_panAngle);
        // _tilt.write((int)_tiltAngle);
    }
};
