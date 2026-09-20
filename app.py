
from src.model import WorldModel
import mesa

model = WorldModel(44, 44, 50, 42)
while True:
    model.step()