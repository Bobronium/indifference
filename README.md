# indifference

Track and compare differences between Python objects.

## Installation
```bash
uv pip install indifference
```

## Usage
```python
import pickle
from copy import deepcopy
from indifference import diff, at


class Person:
    def __init__(self, name):
        self.name = name
        self.cache = {}
        self.secret = "hidden"

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        # Pickle excludes secret and modifies name
        state = {"name": "Bob", "cache": {}}
        return (self.__class__, (self.name,), state)

    def __setstate__(self, state):
        self.__dict__.update(state)


original = Person("Alice")

deepcopy_version = deepcopy(original)
pickle_version = pickle.loads(pickle.dumps(original))

deepcopy_story = diff(original, deepcopy_version)
pickle_story = diff(original, pickle_version)


# Use set operations to find differences
differences = deepcopy_story ^ pickle_story  # symmetric difference
common = deepcopy_story & pickle_story  # intersection
unique_to_pickle = pickle_story - deepcopy_story
all_changes = deepcopy_story | pickle_story

# Check what's different between the stories
assert differences == [at.name > "Bob"]
```

## Development
- Install Task: https://taskfile.dev
- First run: `task setup`
- All checks: `task`
