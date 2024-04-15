# Style Guide

We recommend referring to the provided text and code style for all the articles.

## Text Style

### Length

Examples should be clear and detailed, but not overly verbose. You can add as much text content as you want, as
long as each additional sentence / paragraph provides useful information that helps with understanding the example.
Never use any "filler" content.

### Style

- Use present tense ("We present... we implement...")
- Always define abbreviations / acronyms the first time you use them ("We implement a Graph Attention Network (GAT)...")
- All and any sentence should convey a useful idea; avoid filler at all costs.

### Proofreading

Make sure to proofread your text paragraphs to avoid typos.
Every sentence should start with a capital letter and should end with a period. This applies to code comments as well.

### Introduction and conclusion

There should be an introduction that explains what the reader should expect to find in the example,
and why it is useful/interesting.
If the example presents a specific technique,
the introduction should also include an overview of the technique as well as links to external references.
There should be a conclusion section that recapitulates key takeaways from the example, and offers pointers to next steps.

### Code elements

All code keywords should be formatted with backticks, e.g. `like_this` (standard Markdown code formatting).

When referring to a function or method name, it should be followed with parens, like this: `my_function()` or `my_method()`.

### Line length

Keep text lines relatively short (about 80 characters), unless it's a link.

### Markdown links

Each markdown link should fit on a single line, unbroken, like this:

```md
Here's a link:

[This is the link text](https://github.com/keras-team/keras-io/blob/master/contributor_guide.md)
```

Do not break the link like this (or in any other way):

```md
[This is the link text](
    https://github.com/keras-team/keras-io/blob/master/contributor_guide.md)
```

### Markdown lists

There should be a line break before the first item in any list, e.g.

This is good:

```md
Here's a list:

- First item
- Second item
```

This is bad:

```md
Here's a badly formatted list:
- First item
- Second item
```

## Code Style


We prefer [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for code.

### Variable names

Make sure to use fully-spelled out variable names. Do not use single-letter variable names.
Do not use abbreviations unless they're completely obvious (e.g. `num_layers` is ok).

This is bad:

```python
m = get_model(u=32, d=0.5)
```

This is good:

```python
model = get_model(units=32, dropout_rate=0.5)
```

### Imports

Import modules, not individual objects. In particular, don't import individual layers. Typically
you should import the following:

```python
import tensorflow as tf
import keras
from keras import layers
```

Then access objects from these modules:

```python
tf.Variable(...)
tf.reshape(...)
keras.Input(...)
keras.Model(...)
keras.optimizers.Adam(...)
layers.Layer(...)
layers.Conv2D(...)
```

Read [Imports formatting](https://google.github.io/styleguide/pyguide.html#313-imports-formatting) section for more details on ordering the imports.

### Docstrings

A docstring should give enough information to write a call to the function without reading the function’s code. 

A docstring is mandatory for every function that has one or more of the following properties:

* being part of the public API
* nontrivial size
* non-obvious logic

The docstring may be descriptive-style (`"""Fetches rows from a Bigtable."""`) or imperative-style (`"""Fetch rows from a Bigtable."""`), but the style should be consistent within a file.


Certain aspects of a function should be documented in special sections such as `Args`, `Returns` and `Raises`. These sections can be omitted in cases where the function’s name and signature are informative enough that it can be aptly described using a one-line docstring.

```python
def fetch_smalltable_rows(
    table_handle: smalltable.Table,
    keys: Sequence[bytes | str],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {b'Serak': ('Rigel VII', 'Preparer'),
         b'Zim': ('Irk', 'Invader'),
         b'Lrrr': ('Omicron Persei 8', 'Emperor')}

        Returned keys are always bytes.  If a key from the keys argument is
        missing from the dictionary, then that row was not found in the
        table (and require_all_keys must have been False).

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
```

### Block and Inline Comments

Complicated operations get a few lines of comments before the operations commence. Non-obvious ones get comments at the end of the line.

```python
# We use a weighted dictionary search to find out where i is in
# the array.  We extrapolate position based on the largest num
# in the array and the array size and then do binary search to
# get the exact number.

if i & (i-1) == 0:  # True if i is 0 or a power of 2.
```

On the other hand, never describe the code. Assume the person reading the code knows Python (though not what you’re trying to do) better than you do

```python
# BAD COMMENT: Now go through the b array and make sure whenever i occurs
# the next element is i+1
```

### Punctuation, Spelling, and Grammar

Pay attention to punctuation, spelling, and grammar; it is easier to read well-written comments than badly written ones.

Comments should be as readable as narrative text, with proper capitalization and punctuation. In many cases, complete sentences are more readable than sentence fragments. Shorter comments, such as comments at the end of a line of code, can sometimes be less formal, but you should be consistent with your style.