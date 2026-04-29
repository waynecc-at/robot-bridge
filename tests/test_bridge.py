"""Test script for Robot Bridge"""
import asyncio
import base64
import json
import httpx
import websockets
from pathlib import Path


BASE_URL = "http://127.0.0.1:8081"
WS_URL = "ws://127.0.0.1:8081/ws/robot"


async def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 40)
    print("Testing /health endpoint")
    print("=" * 40)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")


async def test_voices():
    """Test voice listing"""
    print("\n" + "=" * 40)
    print("Testing /api/voices endpoint")
    print("=" * 40)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/api/voices?language=zh")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Found {data['count']} Chinese voices:")
        for voice in data["voices"][:5]:
            print(f"  - {voice['short_name']} ({voice['gender']})")


async def test_tts():
    """Test TTS synthesis"""
    print("\n" + "=" * 40)
    print("Testing /api/tts endpoint")
    print("=" * 40)
    
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{BASE_URL}/api/tts",
            json={"text": "你好，我是小Stack，很高兴认识你！"}
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            audio_data = response.content
            output_file = Path("tests/test_output.mp3")
            output_file.write_bytes(audio_data)
            print(f"Audio saved to {output_file}")
            print(f"Audio size: {len(audio_data)} bytes")


async def test_chat():
    """Test chat endpoint"""
    print("\n" + "=" * 40)
    print("Testing /api/chat endpoint")
    print("=" * 40)
    
    test_messages = [
        "你好，你叫什么名字？",
        "今天天气怎么样？",
        "给我讲个笑话吧"
    ]
    
    async with httpx.AsyncClient(timeout=60) as client:
        for msg in test_messages:
            print(f"\nUser: {msg}")
            
            response = await client.post(
                f"{BASE_URL}/api/chat",
                json={"message": msg, "stream": False}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"Bot: {data['text']}")
            else:
                print(f"Error: {response.status_code} - {response.text}")


async def test_websocket():
    """Test WebSocket connection"""
    print("\n" + "=" * 40)
    print("Testing WebSocket /ws/robot")
    print("=" * 40)
    
    async with websockets.connect(WS_URL) as ws:
        print("Connected to WebSocket")
        
        # Send a text message
        test_msg = {
            "type": "text",
            "text": "你好！"
        }
        await ws.send(json.dumps(test_msg))
        print(f"Sent: {test_msg}")
        
        # Receive responses
        audio_chunks = []
        while True:
            response = await ws.recv()
            data = json.loads(response)
            
            msg_type = data.get("type")
            
            if msg_type == "response_text":
                print(f"Bot text: {data['text']}")
            elif msg_type == "tts_audio":
                if data.get("data"):
                    audio_chunks.append(base64.b64decode(data["data"]))
                if data.get("final"):
                    break
            elif msg_type == "status":
                print(f"Status: {data.get('message')}")
        
        # Save audio
        if audio_chunks:
            audio_data = b"".join(audio_chunks)
            output_file = Path("tests/ws_test_output.mp3")
            output_file.write_bytes(audio_data)
            print(f"Audio saved to {output_file} ({len(audio_data)} bytes)")


async def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Robot Bridge Test Suite")
    print("=" * 60)
    
    try:
        await test_health()
        await test_voices()
        await test_tts()
        await test_chat()
        # await test_websocket()  # Uncomment when server is running
    except httpx.ConnectError:
        print("\n❌ Cannot connect to server. Make sure Robot Bridge is running.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
