
TEACHER_DIR=data/dpo/MiniLLM/SFT-OPT-6.7B
STUDENT_DIR=data/dpo/facebook/opt-1.3b
OUTPUT_DIR=distillm-2/data/reformatted/opt

python distillm-2/generate/reformat.py --teacher_file $TEACHER_DIR --student_file $STUDENT_DIR --output_dir $OUTPUT_DIR