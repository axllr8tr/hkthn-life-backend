# main.py
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from model import PersonAgent, WorldModel
from dataclasses import dataclass

app = FastAPI(title="Simulation API")

@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        message = await websocket.receive_json()
        model = WorldModel(width=44, height=44, num_agents=message["num_agents"], seed=message["seed"])  
        items = list(map(lambda x: Item.parse_json(x), message["items"]))
        craft_connections = list(map(lambda x: CraftConnection.parse_json(x), message["crafts"]))
        print(items)
        print(craft_connections)
        
        while True:
            model.step()            

            await websocket.send_json([serialize_agent(agent) for agent in model.agents])
            
            message = await websocket.receive_json()
            if message.get("ack") is not True:
                return

            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        pass

def serialize_agent(agent: PersonAgent) -> dict[str, any]:
    d =  {
        "id": agent.unique_id,
        "x": agent.x,
        "y": agent.y,
        "strength": agent.strength,
        "intelligence": agent.intelligence,
        "saturation": agent.saturation,
        "messages": agent.message_read.copy()
    }
    agent.message_read.clear()
    return d


@dataclass
class Item:
    id: int
    name: str
    item_type: str
    damage: int
    mining: int

    @staticmethod
    def parse_json(json: dict[str, any]) -> 'Item':
        return Item(id=json["id"], name=json["name"], item_type=json["item_type"], damage=json["damage"], mining=json["mining"])
    

@dataclass
class CraftConnection:
    source: int
    target: int
    count: int

    @staticmethod
    def parse_json(json: dict[str, any]) -> 'CraftConnection':
        return CraftConnection(source=json["source"], target=json["target"], count=json["count"])
