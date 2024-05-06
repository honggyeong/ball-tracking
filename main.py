import streamlit as st
import numpy as np
import cv2
import imutils
from collections import deque
import time
import tempfile

st.title("물체 추적 앱")

video_file = st.file_uploader("또는 비디오 파일 선택", type=["mp4", "avi", "mov"])

buffer_size = 1000  # 최대 버퍼 크기를 1000으로 늘렸습니다.

whiteLower = (0, 0, 200)
whiteUpper = (255, 50, 255)

pts = deque(maxlen=buffer_size)

if video_file is not None:
    # 파일을 임시로 저장
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        temp_file_path = temp_file.name

    video_stream = cv2.VideoCapture(temp_file_path)
    time.sleep(2.0)

    last_frame_with_ball = None  # 마지막으로 공이 화면에 있는 프레임을 저장할 변수 추가

    while True:
        grabbed, frame = video_stream.read()

        if not grabbed:
            st.warning("비디오 파일을 모두 재생했습니다.")
            break

        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, whiteLower, whiteUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                last_frame_with_ball = frame.copy()  # 공이 화면에 있는 경우 해당 프레임을 저장합니다.
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        pts.appendleft(center)

        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 5)  # 라인의 굵기를 고정값인 5로 설정합니다.
            # 공이 화면에 있을 때에만 라인을 그립니다.
            if last_frame_with_ball is not None:
                cv2.line(last_frame_with_ball, pts[i - 1], pts[i], (0, 0, 255), 5)

    # 임시 파일 삭제
    try:
        del temp_file_path
    except Exception as e:
        st.error(f"임시 파일 삭제 중 오류 발생: {e}")

    if last_frame_with_ball is not None:
        st.image(last_frame_with_ball, channels="BGR")  # 마지막으로 공이 화면에 있는 프레임을 표시합니다.
    else:
        st.write("비디오에서 공이 발견되지 않았습니다.")
else:
    st.write("비디오 파일을 선택해주세요.")
