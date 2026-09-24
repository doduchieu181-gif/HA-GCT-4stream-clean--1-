"""Điểm vào masked reconstruction; chỉ train encoder trên split train."""
from hagct.engine.trainer import main

if __name__ == "__main__":
    main(pretrain=True)
