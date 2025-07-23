import json
from typing import List
from openai import OpenAI
from anthropic import Anthropic
from jinja2 import Template
import json_repair


class QuestionReformat:
    def __init__(self):
        self.client = Anthropic()
        self.model = "claude-sonnet-4-20250514"
        with open("restructure_0_hop.md", "r") as f:
            template_str = f.read()
        self.zero_hop_template = Template(template_str)

        with open("restructure_1_2_hop.md", "r") as f:
            template_str = f.read()
        self.one_two_hop_template = Template(template_str)

        with open("update.md", "r") as f:
            template_str = f.read()
        self.update_template = Template(template_str)

    def reformat(self, user, personal_info, questions, is_zero=False) -> list:
        if is_zero:
            sys = self.zero_hop_template.render(
                user=user, personal_info=personal_info, questions=questions
            )
        else:
            sys = self.one_two_hop_template.render(
                user=user, personal_info=personal_info, questions=questions
            )

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            temperature=1,
            system=sys,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": json.dumps(questions)}],
                }
            ],
        )
        try:
            return json_repair.loads(message.content[0].text)
        except:
            raise RuntimeError(
                f"✖ LLM returned invalid JSON: {message.content}"
            ) from None

    def reformat_update(self, user, path) -> List[str]:
        sys = self.update_template.render(user=user, path=json.dumps(path, indent=2))

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            temperature=1,
            messages=[{"role": "user", "content": [{"type": "text", "text": sys}]}],
        )
        # format as list
        try:
            return json_repair.loads(message.content[0].text)
        except:
            raise RuntimeError(
                f"✖ LLM returned invalid JSON: {message.content}"
            ) from None


class LLM:
    def __init__(self, model="gpt-4o"):
        self.client = OpenAI()
        self.model = model

    def create_text(self, system: str, prompt: str) -> str:
        return (
            self.client.chat.completions.create(
                model="o3",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.0,
            )
            .choices[0]
            .message.content
        )

    def create_json(self, system: str, user: str, format) -> str:
        return (
            self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=1.0,
                response_format=format,
            )
            .choices[0]
            .message.parsed
        )
