from __future__ import annotations

import unittest

from rfsn_v10_5.tokenizer_utils import (
    apply_chat_template,
    encode_messages,
    get_tokenizer_capabilities,
    tokenizer_supports_chat_templates,
)


class ChatTemplateTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if add_generation_prompt:
            rendered = f"{rendered}\nassistant:"
        if tokenize:
            return {"input_ids": [11, 12, 13]}
        return rendered


class PlainTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = True):
        return {"input_ids": [1, 2]}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return "|".join(str(token_id) for token_id in token_ids)


class ChatTemplateTest(unittest.TestCase):
    def test_capability_detection_reports_chat_template_support(self) -> None:
        supported = get_tokenizer_capabilities(ChatTemplateTokenizer())
        unsupported = get_tokenizer_capabilities(PlainTokenizer())

        self.assertTrue(supported["chat_template"])
        self.assertFalse(unsupported["chat_template"])
        self.assertTrue(tokenizer_supports_chat_templates(ChatTemplateTokenizer()))
        self.assertFalse(tokenizer_supports_chat_templates(PlainTokenizer()))

    def test_apply_chat_template_renders_generation_prompt(self) -> None:
        rendered = apply_chat_template(
            ChatTemplateTokenizer(),
            [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Hello"},
            ],
            add_generation_prompt=True,
        )

        self.assertIn("system: You are concise.", rendered)
        self.assertTrue(rendered.endswith("assistant:"))

    def test_encode_messages_uses_chat_template_tokens(self) -> None:
        prompt_ids = encode_messages(
            ChatTemplateTokenizer(),
            [{"role": "user", "content": "Hello"}],
            vocab_size=64,
        )

        self.assertEqual(prompt_ids.tolist(), [[11, 12, 13]])

    def test_unsupported_tokenizer_raises_for_chat_templates(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support chat templates"):
            apply_chat_template(PlainTokenizer(), [{"role": "user", "content": "Hello"}])


if __name__ == "__main__":
    unittest.main()