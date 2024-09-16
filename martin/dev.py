from llm.cache.disk import get_disk
from llm.loader import load_data

cache = get_disk()
cache.clear()

data = load_data(name='karamazov', train_ratio=0.5)
data = load_data(name='prince', train_ratio=0.97)
print(data)