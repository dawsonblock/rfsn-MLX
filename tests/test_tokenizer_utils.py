from __future__ import annotations

import unittest

import mlx.core as mx

from rfsn_v10_5.tokenizer_utils import (
    decode_token_ids,
    encode_prompt_text,
    materialize_generated_sequence,
    prompt_ids_from_list,
)


class FakeTokenizer:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids

    def __call__(self, text: str, add_special_tokens: bool = True):
        return {"input_ids": list(self.token_ids)}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "|".join(str(token_id) for token_id in token_ids)


class TokenizerUtilsTest(unittest.TestCase):
    def test_encode_prompt_text_returns_batch_int32_tensor(self) -> None:
        tokenizer = FakeTokenizer([4, 5, 6])

        prompt_ids = encode_prompt_text(tokenizer, "hello", vocab_size=32)

        self.assertEqual(prompt_ids.shape, (1, 3))
        self.assertEqual(prompt_ids.tolist(), [[4, 5, 6]])

    def test_encode_prompt_text_rejects_out_of_vocab_ids(self) -> None:
        tokenizer = FakeTokenizer([1, 9])

        with self.assertRaisesRegex(ValueError, "outside model vocab_size"):
            encode_prompt_text(tokenizer, "hello", vocab_size=8)

    def test_decode_token_ids_accepts_batch_one_tensor_shape(self) -> None:
        tokenizer = FakeTokenizer([1])

        decoded = decode_token_ids(tokenizer, [[7, 8, 9]])

        self.assertEqual(decoded, "7|8|9")

    def test_decode_token_ids_rejects_multi_batch(self) -> None:
        tokenizer = FakeTokenizer([1])

        with self.assertRaisesRegex(ValueError, "Only batch_size=1 decode"):
            decode_token_ids(tokenizer, [[1], [2]])

    def test_prompt_ids_from_list_validates_and_batches(self) -> None:
        prompt_ids = prompt_ids_from_list([2, 3, 4], vocab_size=8)

        self.assertEqual(prompt_ids.tolist(), [[2, 3, 4]])

    def test_materialize_generated_sequence_concatenates_generated_steps(self) -> None:
        prompt_ids = prompt_ids_from_list([1, 2], vocab_size=8)

        token_ids = materialize_generated_sequence(
            prompt_ids,
            [mx.array([3], dtype=mx.int32), mx.array([4], dtype=mx.int32)],
        )

        self.assertEqual(token_ids.tolist(), [[1, 2, 3, 4]])


if __name__ == "__main__":
    unittest.main()