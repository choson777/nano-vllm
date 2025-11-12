#!/bash/bin

export CUDA_VISIBLE_DEVICES=2,3
python tests/test_spec_engine.py > record.txt