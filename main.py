"""CLI runtime entrypoint for ANTARDRISHTI.

For product mode use Flask:
flask --app app run --host 0.0.0.0 --port 5000 --no-reload
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from runtime import AntardrishtiRuntime, RuntimeConfig


def run_cli(args) -> None:
    config = RuntimeConfig(
        webcam=args.webcam,
        pi_url=args.pi_url,
        state_file=args.state_file,
        break_seconds=args.break_seconds,
        obstacle_area=args.obstacle_area,
        voice_cooldown=args.voice_cooldown,
        mute=args.mute,
        output_dir=args.output_dir,
        record_annotated=args.record_annotated,
        auto_start_pi=args.auto_start_pi,
        pi_host=args.pi_host,
        pi_username=args.pi_user,
        pi_password=args.pi_password,
        pi_port=args.pi_port,
        pi_start_cmd=args.pi_start_cmd,
    )

    runtime = AntardrishtiRuntime(config)
    runtime.start()

    try:
        while True:
            state = runtime.get_state()
            if args.show_preview:
                frame_bytes = runtime.latest_frame_jpeg()
                if frame_bytes:
                    np_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
                    if frame is not None:
                        cv2.imshow("ANTARDRISHTI", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(
                    f"[{state.get('runtime', 'n/a')}] "
                    f"state={state.get('state', 'n/a')} "
                    f"activity={state.get('activity', 'n/a')} "
                    f"score={state.get('score', {}).get('skill_score', 0)}"
                )
                time.sleep(1.0)

            if state.get("runtime") == "error":
                break

    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANTARDRISHTI adaptive AI coach")
    parser.add_argument("--webcam", type=int, default=0, help="Laptop camera source index")
    parser.add_argument("--pi-url", default="udp://@:8080", help="Pi UDP stream URL")
    parser.add_argument("--state-file", default="state.json", help="State JSON file")
    parser.add_argument("--break-seconds", type=int, default=7, help="Break duration")
    parser.add_argument("--obstacle-area", type=int, default=7000, help="Obstacle contour area threshold")
    parser.add_argument("--voice-cooldown", type=float, default=2.4, help="Voice cooldown")
    parser.add_argument("--mute", action="store_true", help="Disable speech output")
    parser.add_argument("--show-preview", action="store_true", help="Show OpenCV preview")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for recordings")
    parser.add_argument("--record-annotated", dest="record_annotated", action="store_true", help="Record annotated video")
    parser.add_argument("--no-record-annotated", dest="record_annotated", action="store_false", help="Disable recording")
    parser.add_argument("--auto-start-pi", action="store_true", help="Auto-start Pi camera stream over SSH")
    parser.add_argument("--pi-host", default="", help="Pi host")
    parser.add_argument("--pi-user", default="", help="Pi username")
    parser.add_argument("--pi-password", default="", help="Pi password")
    parser.add_argument("--pi-port", type=int, default=22, help="Pi SSH port")
    parser.add_argument(
        "--pi-start-cmd",
        default="cd ~ && ./start_cam.sh > /tmp/drishti_stream.log 2>&1 &",
        help="Pi command to start stream",
    )
    parser.set_defaults(record_annotated=True)
    args = parser.parse_args()

    run_cli(args)
