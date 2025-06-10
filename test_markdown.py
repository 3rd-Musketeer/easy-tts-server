from src.easy_tts_server.utils import normalize_text

# Test with complex markdown
test_md = '''
# Main Title
## Subtitle
- List item 1
- List item 2
  - Nested item

**Bold text** and *italic text*

[Link text](https://example.com)

`inline code` and:

```python
def hello():
    print('world')
```

| Table | Header |
|-------|--------|
| Cell 1| Cell 2 |

> Blockquote text
'''

result = normalize_text(test_md)
print('Result:')
print(repr(result))
print()
print('Cleaned text:')
print(result) 