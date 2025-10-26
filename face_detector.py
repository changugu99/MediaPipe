#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

# =========================================
# 🙂 MediaPipe 초기화 (Face Detection)
# =========================================
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# model_selection:
#   0 = 근거리(웹캠) 최적화, 1 = 원거리(전신/멀리서) 최적화
detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# =========================================
# 📸 카메라/영상 입력
# =========================================
# cap = cv2.VideoCapture(0)                 # 기본 카메라
cap = cv2.VideoCapture("face.mp4")          # 동영상 파일 사용 시

print("📷 얼굴 감지 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 프레임을 읽지 못했습니다. 입력 소스를 확인하세요.")
        break

    # 셀카 뷰(좌우 반전) — 필요 없으면 주석 처리
    frame = cv2.flip(frame, 1)

    # BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 감지
    results = detector.process(rgb)

    # 감지 결과 그리기 (바운딩박스 + 6개 키포인트)
    if results.detections:
        for det in results.detections:
            mp_draw.draw_detection(frame, det)

    # 화면 표시
    cv2.imshow("🙂 MediaPipe Face Detector", frame)

    # ESC로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()
