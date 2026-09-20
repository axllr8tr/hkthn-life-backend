from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import mesa
from mesa.space import MultiGrid

@dataclass
class CellData:
    health: float = 100.0
    strength: float = 1.0
    intelligence: float = 1.0
    saturation: float = 100.0

class PersonAgent(mesa.Agent):
    """
    Agent living on the grid.

    Attributes:
        hp: Health points.
        strength: Attack strength.
        intelligence: Intelligence value.
        saturation: Hunger/energy/saturation value.
        message_queue: Messages received from other agents.
    """

    def __init__(
        self,
        model: "WorldModel",
        hp: float = 100.0,
        strength: float = 10.0,
        intelligence: float = 10.0,
        saturation: float = 100.0,
    ):
        super().__init__(model)
        self.hp = hp
        self.strength = strength
        self.intelligence = intelligence
        self.saturation = saturation

        self.message_queue: List[dict] = []
        self.message_read: List[dict] = []

    @property
    def x(self):
        return self.cell.coordinate[0]

    @property
    def y(self):
        return self.cell.coordinate[1]

    def get_nearby_agents(self, radius: int = 10) -> List["PersonAgent"]:
        """
        Return all agents within radius cells of this agent.

        Moore neighborhood is used, so diagonal cells are included.
        The agent itself is excluded.
        """
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            radius=radius,
            include_center=False,
        )

        return [
            agent
            for agent in neighbors
            if isinstance(agent, PersonAgent) and agent.hp > 0
        ]

    def intelligence_tick(self):
        nearby_agents = self.get_nearby_agents(radius=10)
        

    def step(self):
        self.tick()

    def tick(self):
        self.saturation -= self.model.SATURATION_DECAY

        if self.saturation < 10:
            damage = (10 - self.saturation) * 0.2
            self.hp -= damage

        if self.hp <= 0:
            self.die()

    def hit(self, other: "PersonAgent"):
        if self.hp <= 0:
            return

        if other.hp <= 0:
            return

        if other is self:
            return

        # Avoid division by zero.
        if other.strength <= 0:
            damage = self.strength * 10
        else:
            damage = (self.strength / other.strength) * 10

        other.hp -= damage

        if other.hp <= 0:
            other.die()

        return damage

    def walk(self, d) -> bool:
        x, y = self.pos
        if d == "left":
            dest = (x - 1, y)
        elif d == "top":
            dest = (x, y + 1)
        elif d == "right":
            dest = (x + 1, y)
        elif d == "bottom":
            dest = (x, y - 1)
        else:
            return

        if self.model.grid.out_of_bounds(dest):
            return
        if not self.model.grid.is_cell_empty(dest):
            return
        self.model.grid.move_agent(self, dest)

    def talk(self, message: str):
        """
        Send a message to every agent within 10 cells.

        The message is placed in each recipient's message queue.
        """

        nearby_agents = self.get_nearby_agents(radius=10)

        for other in nearby_agents:
            other.message_queue.append(
                {
                    "sender": self.unique_id,
                    "message": message,
                }
            )

    def read_messages(self) -> List[dict]:
        """
        Return and clear all messages currently in the queue.
        """
        messages = self.message_queue.copy()
        self.message_read.extend(messages)
        self.message_queue.clear()
        return messages

    def die(self):
        """Remove the agent from the simulation."""
        if self.pos is not None:
            self.model.grid.remove_agent(self)

        self.remove()



class WorldModel(mesa.Model):
    """
    Grid-based Mesa world.
    """

    SATURATION_DECAY = 1.0
    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        num_agents: int = 50,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height

        # MultiGrid allows multiple agents in the same cell.
        self.grid = MultiGrid(
            width=width,
            height=height,
            torus=False,
        )

        # Create agents.
        for _ in range(num_agents):
            agent = PersonAgent(
                self,
                hp=self.random.uniform(80, 100),
                strength=self.random.uniform(5, 15),
                intelligence=self.random.uniform(5, 15),
                saturation=self.random.uniform(50, 100),
            )

            x = self.random.randrange(width)
            y = self.random.randrange(height)

            self.grid.place_agent(agent, (x, y))

        self.running = True

    def select_for_intelligence_tick(self, count: int = 10):
        agents = [
            agent
            for agent in self.agents
            if agent.pos is not None and agent.hp > 0
        ]
        if not agents:
            return

        count = min(count, len(agents))
        weights = np.array(
            [max(0.0, agent.intelligence) for agent in agents],
            dtype=float,
        )
        if weights.sum() == 0:
            probabilities = None
        else:
            probabilities = weights / weights.sum()

        rng = np.random.default_rng(seed=self.random.randint(0, 100))
        indices = rng.choice(
            len(agents),
            size=count,
            replace=False,
            p=probabilities,
        )
        return [agents[i] for i in indices]


    def step(self):
        for agent in self.select_for_intelligence_tick():
            agent.intelligence_tick()

        
        agents = list(self.agents)
        self.random.shuffle(agents)

        for agent in agents:
            if agent.pos is not None and agent.hp > 0:
                agent.tick()

        self.steps += 1


if __name__ == "__main__":

    model = WorldModel(
        width=50,
        height=50,
        num_agents=100,
        seed=42,
    )

    for step in range(20):
        print(f"\n--- Step {step} ---")

        model.step()

        alive_agents = [
            agent
            for agent in model.agents
            if agent.hp > 0
        ]

        print(f"Alive agents: {len(alive_agents)}")

        if alive_agents:
            agent = alive_agents[0]

            print(
                f"Agent {agent.unique_id}: "
                f"pos={agent.pos}, "
                f"hp={agent.hp:.2f}, "
                f"strength={agent.strength:.2f}, "
                f"intelligence={agent.intelligence:.2f}, "
                f"saturation={agent.saturation:.2f}"
            )