
TEACHER_DIR=data/dpo/MiniLLM/teacher-gpt2-1.5B
STUDENT_DIR=data/dpo/openai-community/gpt2
OUTPUT_DIR=distillm-2/data/reformatted/gpt2

python distillm-2/generate/reformat.py --teacher_file $TEACHER_DIR --student_file $STUDENT_DIR --output_dir $OUTPUT_DIR