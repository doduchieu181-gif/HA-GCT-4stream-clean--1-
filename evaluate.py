"""Đánh giá riêng sau train, không dùng test để chọn checkpoint."""
from hagct.engine.inference import evaluate_main

if __name__ == "__main__":
    evaluate_main()
