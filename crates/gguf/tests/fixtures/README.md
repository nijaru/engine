# Qwen chat-template fixtures

`qwen38-chat.jinja` is the exact `tokenizer.chat_template` string extracted from
`Qwen3.8-27B-UD-Q4_K_M.gguf`, SHA-256
`322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`.
The artifact identifies its license as Apache-2.0.

The `.txt` outputs were rendered with Python Jinja2 3.1.6 using the same message
arrays and options as `pinned_template_matches_python_jinja_rendering`.
Case 0 is a non-thinking user turn. Case 1 enables thinking and includes merged
system/developer instructions and assistant history. No whitespace or escape
normalization was applied. The rendered fixtures contain actual newline bytes,
matching the template's escaped newline literals.

Regenerate using `jinja2.Environment().from_string(template).render(messages=...,
add_generation_prompt=True, enable_thinking=...)`. Keep fixture changes tied to
an identified artifact and compare exact bytes, not displayed text.
