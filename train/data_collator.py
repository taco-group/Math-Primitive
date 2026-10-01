import torch


class SelfDistillationDataCollator:
    """
    Data collator for on-policy self-distillation: builds a student and a teacher prompt per row.

    Student: sees only the problem (with chat template).
    Teacher: the SAME model, additionally given privileged context about the problem.

    privilege_style
      "primitive" (Absorb, ours): the privileged context is the problem's mathematical primitive
                   (one-sentence core concept, no computation, no answer); the teacher is told it is
                   correct and to build on it, and may still reflect / verify / reconsider.
      "solution"  (original OPSD): the privileged context is a full reference solution.

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=True,
        student_thinking=False,
        teacher_thinking=True,
        privilege_style="solution",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        if privilege_style not in ("primitive", "solution"):
            raise ValueError(f"privilege_style must be 'primitive' or 'solution', got {privilege_style!r}")
        self.privilege_style = privilege_style

        if privilege_style == "primitive":
            # Absorb: privilege = the primitive's core concept (single-sentence key idea).
            self.privilege_header = (
                "Here is the mathematical primitive for this problem — the essential idea that makes it solvable:\n"
                "=== Primitive Begin ===\n{privilege}\n=== Primitive End ===\n"
            )
            self.transition_prompt = (
                "\n\nThis primitive is correct for this problem. Solve the problem by "
                "building on it: let it guide your choice of approach, and work out "
                "the solution to its conclusion. You may reflect, verify, or "
                "reconsider your steps whenever you find it necessary.\n"
            )
            self.reason_first_prompt = None  # reason_first is not supported for this style
        else:
            # Original OPSD: the privileged info is a full reference solution.
            self.privilege_header = (
                "Here is a reference solution to this problem:\n"
                "=== Reference Solution Begin ===\n{privilege}\n=== Reference Solution End ===\n"
            )
            self.reason_first_prompt = (
                "\n\nThe reference reasoning above arrives at the correct answer. "
                "Please analyze this solution and explain the key reasoning steps and problem-solving strategies employed. "
                "Do NOT use <think> tags. Do NOT derive your own solution. "
                "Simply analyze and explain the reference solution provided above.\n"
            )
            self.transition_prompt = (
                "\n\nAfter reading the reference solution above, make sure you truly understand "
                "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
                "own words and independent reasoning, derive the same final answer to the problem above. "
                "Think step by step, explore different approaches, and don't be afraid to backtrack "
                "or reconsider if something doesn't work out:\n"
            )

        # Per-row task tail: boxed answer for answer-bearing rows, rigorous proof otherwise.
        # (plain strings, concatenated — never passed through .format())
        self.tail_answer = "Please reason step by step, and put your final answer within \\boxed{}."
        self.tail_proof = "Please reason step by step, and provide a complete, rigorous proof."

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # for reason_first mode

        for feature in features:
            problem = feature["problem"]
            solution = feature["solution"]  # the privileged context (primitive or reference solution)
            # Per-row tail: answer-bearing rows ask for \boxed{}, proof rows ask for a proof
            tail = self.tail_answer if feature.get("final_answer") else self.tail_proof

            # Student prompt: just the problem with instruction (matching evaluation format)
            student_user_message = f"Problem: {problem}\n\n{tail}"
            student_messages = [{"role": "user", "content": student_user_message}]
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.student_thinking
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                assert self.reason_first_prompt is not None, "reason_first is only supported with privilege_style='solution'"
                reasoning_user_message = (
                    f"Problem: {problem}\n\n"
                    f"{self.privilege_header.format(privilege=solution)}"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.teacher_thinking
                )
                teacher_reasoning_prompts.append(reasoning_prompt)
                teacher_prompts.append("")  # placeholder, replaced in training_step
            else:
                # Teacher prompt: problem + privileged context.
                # primitive style: no step-by-step/boxed tail on the teacher side (the transition
                # already carries the instruction); the student keeps the per-row tail.
                if self.privilege_style == "primitive":
                    teacher_user_message = (
                        f"Problem: {problem}\n\n"
                        f"{self.privilege_header.format(privilege=solution)}"
                        f"{self.transition_prompt}"
                    )
                else:
                    teacher_user_message = (
                        f"Problem: {problem}\n\n"
                        f"{self.privilege_header.format(privilege=solution)}"
                        f"{self.transition_prompt}\n"
                        f"{tail}"
                    )
                teacher_messages = [{"role": "user", "content": teacher_user_message}]
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.teacher_thinking
                )
                teacher_prompts.append(teacher_prompt)

        # Tokenize WITHOUT padding first to get true lengths
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

        # Find max lengths in this batch
        max_student_prompt_len = max(student_prompt_lengths)

        # Tokenize WITH padding to max length in batch
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }

        if self.reason_first:
            # Tokenize reasoning prompts
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text. Per-row tail (boxed vs proof).
            transition_texts = [
                f"\n{self.transition_prompt}\n{self.tail_answer if f_.get('final_answer') else self.tail_proof}"
                for f_ in features
            ]
            # padding="longest" because tails differ per row (boxed vs proof); pad tokens are
            # attention-masked downstream (training_step masks == pad_token_id).
            transition_encoded = self.tokenizer(
                transition_texts,
                padding="longest",
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            # Normal mode: tokenize teacher prompts
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)

            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )

        return result
