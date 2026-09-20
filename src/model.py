from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Deque, Literal
from langchain.chat_models import BaseChatModel
from langchain_gigachat import GigaChat
import numpy as np
import mesa
from langchain.agents import create_agent
from langchain.tools import tool
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
        llm: str | BaseChatModel,
        model: "WorldModel",
        hp: float = 100.0,
        strength: float = 10.0,
        intelligence: float = 10.0,
        saturation: float = 100.0
    ):
        super().__init__(model)
        self.hp = hp
        self.strength = strength
        self.intelligence = intelligence
        self.saturation = saturation

        self.history: Deque[str] = deque(maxlen=10)
        self.message_queue: List[dict] = []
        self.message_read: List[dict] = []
        self.llm_tools = self._build_tools()
        self.talk_tools = [t for t in self.llm_tools if t.name == "talk"]
        # Tools must be passed at create_agent time. Empty list = no tool loop.
        # invoke(..., tools=...) is ignored by the compiled graph.
        self.llm_agent = create_agent(llm, tools=self.llm_tools)
        self.intel_agent = create_agent(llm, tools=self.talk_tools)

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    def get_system_prompt(self, is_intel: bool = False) -> str:
        return f"""You are an agent (ID: {self.unique_id}), a living entity in a grid world.
Attributes:
- Health: {self.hp:.2f}/100
- Saturation: {self.saturation:.2f}/100
- Intelligence: {self.intelligence:.2f}
- Strength: {self.strength:.2f}

Grid Location: ({self.x}, {self.y})
World Bounds: ({self.model.width - 1}, {self.model.height - 1})
Entities in Vision Field:
- Nearby Agents: {'\n'.join(f"ID {i.unique_id} at ({i.x}, {i.y}) hp={i.hp:.0f}" for i in self.get_nearby_agents()) if self.get_nearby_agents() else 'No one nearby.'}

Recent Memories:
{chr(10).join(self.history) if self.history else "No recent memories."}

Recent Messages from Other Agents:
{chr(10).join([f"ID {x.get('sender')}: {x.get('message')}" for x in self.message_queue]) if self.message_queue else "No recent messages."}
""" + (
"""
Goal: Survive, thrive, eat when hungry, interact with others, and stay safe.
You MUST take exactly ONE action by calling a tool. Do not reply with plain text.
If you have nothing else to do, call talk with a short thought. 
"""
        if not is_intel
        else
"""
Goal: Assess the situation and say something to other agents around you.
You MUST call the talk tool. Do not reply with plain text.
""")

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

    def _run_llm(self, *, intel: bool = False):
        graph = self.intel_agent if intel else self.llm_agent
        result = graph.invoke(
            {"messages": [{"role": "user", "content": self.get_system_prompt(intel)}]},
            {"recursion_limit": 8},
        )
        # print(result)
        return result

    def intelligence_tick(self):
        self._run_llm(intel=True)

    def step(self):
        self.tick()

    def tick(self):
        self.saturation -= self.model.SATURATION_DECAY

        if self.saturation < 10:
            damage = (10 - self.saturation) * 0.2
            self.hp -= damage

        if self.hp <= 0:
            self.die()
        self.read_messages()

    def hit(self, other: "PersonAgent"):
        if self.hp <= 0:
            return

        self._run_llm(intel=False)

    def _build_tools(self):
        agent = self

        @tool(return_direct=True)
        def hit(target_id: int) -> str:
            """Attack a nearby living agent by unique ID. Target must be adjacent (1 cell)."""
            if agent.hp <= 0:
                return "You are dead and cannot attack."
            if target_id == agent.unique_id:
                return "You cannot attack yourself."

            other = next(
                (a for a in agent.get_nearby_agents(radius=1) if a.unique_id == target_id),
                None,
            )
            if other is None:
                return f"No living adjacent agent with ID {target_id}."

            if other.strength <= 0:
                damage = agent.strength * 10
            else:
                damage = (agent.strength / other.strength) * 10

            other.hp -= damage
            agent.history.append(
                f"Entity ID {agent.unique_id} attacked {other.unique_id} and dealt {damage:.2f} damage"
            )
            if other.hp <= 0:
                other.die()
            return f"Dealt {damage:.2f} damage to agent {target_id}."

        @tool(return_direct=True)
        def walk(direction: Literal["left", "right", "top", "bottom"]) -> str:
            """Move one cell. direction must be left, right, top, or bottom."""
            x, y = agent.pos
            dest = {
                "left": (x - 1, y),
                "right": (x + 1, y),
                "top": (x, y + 1),
                "bottom": (x, y - 1),
            }.get(direction)
            if dest is None:
                return "Unknown direction. Use left, right, top, or bottom."
            if agent.model.grid.out_of_bounds(dest):
                return "Cannot walk out of bounds."
            if not agent.model.grid.is_cell_empty(dest):
                return "Destination cell is occupied."
            agent.model.grid.move_agent(agent, dest)
            agent.history.append(f"Entity ID {agent.unique_id} walked {direction} to {dest}")
            return f"Moved {direction} to {dest}."

        @tool(return_direct=True)
        def talk(message: str) -> str:
            """Send a message to every agent within 10 cells."""
            nearby_agents = agent.get_nearby_agents(radius=10)
            for other in nearby_agents:
                other.message_queue.append(
                    {
                        "sender": agent.unique_id,
                        "message": message,
                    }
                )
            agent.history.append(f"Entity ID {agent.unique_id} said: {message}")
            if not nearby_agents:
                return "No one nearby heard you."
            return f"Sent message to {len(nearby_agents)} nearby agent(s)."

        @tool(return_direct=True)
        def eat() -> str:
            """Forage in place and restore saturation. Use this when hungry."""
            gained = 20.0
            agent.saturation = min(100.0, agent.saturation + gained)
            agent.history.append(
                f"Entity ID {agent.unique_id} ate and saturation is {agent.saturation:.2f}"
            )
            return f"Saturation is now {agent.saturation:.2f}/100."

        return [hit, walk, talk, eat]

    def read_messages(self) -> List[dict]:
        """Return and clear all messages currently in the queue."""
        messages = self.message_queue.copy()
        self.message_read.extend(messages)
        self.message_queue.clear()
        return messages

    def terminate(self):
        """Removes you, the agent, from the simulation"""
        self.die()

    def die(self):
        """Remove the agent from the simulation."""
        if self.pos is not None:
            self.model.grid.remove_agent(self)
        self.history.append(f"Entity ID {self.unique_id} died")
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
        agent_llm: Optional[str | BaseChatModel] = "ollama:qwen2.5:1.5b-instruct-q4_0"
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
                model=self,
                llm=agent_llm,
                hp=self.random.uniform(80, 100),
                strength=self.random.uniform(5, 90),
                intelligence=self.random.uniform(5, 90),
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
            return []

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
        num_agents=10,
        seed=42,
        agent_llm=GigaChat(credentials='MDFhMGI5OTAtNWVlMy03ZjI3LWJiNDktMDk5ZGQxMjI3N2E4OjQyOGM3NTViLWEwZTctNDEyYi05M2U5LTc2MGNhYjM2NDNkNw==', model="GigaChat-3-Ultra", verify_ssl_certs=False)
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

            messages = chr(10).join(
                f"ID {m.get('sender')}: {m.get('message')}"
                for m in agent.message_queue
            )
            print(
                f"Agent {agent.unique_id}: "
                f"pos={agent.pos}, "
                f"hp={agent.hp:.2f}, "
                f"strength={agent.strength:.2f}, "
                f"intelligence={agent.intelligence:.2f}, "
                f"saturation={agent.saturation:.2f}\n"
                f"\tmessages={messages}\n"
                f"\tevents={chr(10).join(agent.history)}\n"
            )